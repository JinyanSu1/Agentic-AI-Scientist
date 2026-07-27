"""Shared openai-agents (Agent/Runner/handoff) setup for fugu, plus tools wrapping
our search + knowledge-bank utilities so agents can search/read papers as genuine
tool calls (the agent decides when to read something, not a Python pre-fetch step).

Call configure_fugu_as_default() once per process before constructing any Agent.
"""

import json
import os
import os.path as osp
import time
from dataclasses import dataclass
from typing import Any, Optional

from agents import (
    AsyncOpenAI,
    RunContextWrapper,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.exceptions import MaxTurnsExceeded

from ai_scientist.tools.knowledge_bank import get_paper_knowledge
from ai_scientist.tools.paper_search import search_for_papers
from ai_scientist.tools.codex_worker import (
    describe_plot,
    list_workdir_files,
    read_text_file,
)
from ai_scientist.tools.coding_worker import run_coding_task


def configure_fugu_as_default(model: str = "fugu") -> None:
    client = AsyncOpenAI(
        api_key=os.environ["SAKANA_API_KEY"],
        base_url="https://api.sakana.ai/v1",
        # fugu can be slow to respond on very large research-agent contexts; a
        # short default timeout there causes "Request timed out" -> a full
        # from-scratch re-run. Give it generous headroom.
        timeout=600,
    )
    # Tracing defaults to uploading run traces to OpenAI's platform using
    # OPENAI_API_KEY, which we don't have a valid one for and don't want anyway
    # since we're not running against OpenAI's actual models.
    set_default_openai_client(client, use_for_tracing=False)
    set_tracing_disabled(True)
    set_default_openai_api("chat_completions")


def run_agent_with_retry(
    agent: Any,
    input_data: Any,
    context: Optional[Any] = None,
    max_turns: int = 20,
    retries: int = 3,
) -> Any:
    """Runner.run_sync, but a single garbled/corrupted model response (a real,
    observed failure mode -- e.g. an occasional token-level glitch that breaks
    structured-output JSON parsing) raises agents.exceptions.ModelBehaviorError
    and would otherwise kill an entire multi-hour pipeline run. Retry a few
    times before giving up for real."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if context is not None:
                return Runner.run_sync(agent, input_data, context=context, max_turns=max_turns)
            return Runner.run_sync(agent, input_data, max_turns=max_turns)
        except MaxTurnsExceeded:
            # Hitting the turn cap is not a transient glitch -- re-running the
            # whole conversation from scratch would just hit it again (at 3x the
            # cost). Propagate immediately so the caller can salvage partial work
            # (see run_research_agent's distill-from-workdir fallback) instead.
            raise
        except Exception as e:
            last_exc = e
            print(
                f"Agent run failed (attempt {attempt + 1}/{retries + 1}) for "
                f"agent {getattr(agent, 'name', '?')}: {e}"
            )
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise last_exc


@dataclass
class ResearchContext:
    """Passed as Runner.run(..., context=...) so tools know where to cache
    knowledge-bank entries and what model to use for paper summarization.

    knowledge_bank_dir, when set, is where the paper cache lives; it should be a
    directory shared across every stage of one research loop (pilots, novelty
    check, development rounds, writeup) so the same paper is downloaded and
    summarized once, not re-fetched per stage. If None, it falls back to workdir
    (the old per-stage behavior)."""

    workdir: str
    model: str = "fugu"
    knowledge_bank_dir: Optional[str] = None
    # The loop directory holding round_XX_outcome.json for THIS idea's prior
    # development rounds, so recall_prior_rounds can retrieve their details on
    # demand instead of us stuffing the whole history into the kickoff prompt.
    loop_dir: Optional[str] = None
    # Wall-clock timeout (seconds) for a single coding-worker sub-task. Configurable
    # because a genuinely heavy sub-task (e.g. real training) can need more than
    # the 1h default.
    codex_timeout: int = 3600
    # Which CLI coding agent run_experiment_task delegates to: "codex" or
    # "claude-code" (see ai_scientist/tools/coding_worker.py).
    worker: str = "codex"
    # Codex CLI profile to use when worker == "codex" (None/"" for Codex's own
    # default profile/login instead of a named profile).
    codex_profile: Optional[str] = "fugu"


@function_tool
def search_literature(query: str) -> str:
    """Search Semantic Scholar, OpenAlex, and arXiv (merged and deduplicated) for
    papers matching a query. Returns title/authors/venue/year/abstract for each."""
    papers = search_for_papers(query, result_limit=5)
    if not papers:
        return "No papers found."
    return "\n\n".join(
        f"{p.get('title', 'Unknown Title')}. {p.get('authors', 'Unknown')}. "
        f"{p.get('venue', 'Unknown Venue')}, {p.get('year', 'Unknown Year')}.\n"
        f"Abstract: {p.get('abstract', 'No abstract available.')}"
        for p in papers
    )


def _read_paper_in_depth_impl(workdir: str, model: str, query_or_path: str, why: str) -> str:
    if osp.isfile(query_or_path) and query_or_path.lower().endswith(".pdf"):
        title = osp.splitext(osp.basename(query_or_path))[0].replace("_", " ").replace("-", " ")
        paper = {"title": title, "abstract": "", "pdf_url": osp.abspath(query_or_path)}
    else:
        papers = search_for_papers(query_or_path, result_limit=1)
        if not papers:
            return f"No paper found for {query_or_path!r}."
        paper = papers[0]
    return get_paper_knowledge(workdir, paper, model, idea_context=why)


@function_tool
def read_paper_in_depth(
    ctx: RunContextWrapper[ResearchContext], query_or_path: str, why: str
) -> str:
    """Read a paper in full (from a local PDF path or by searching for it by
    title/topic) and return a summary of what's useful for the given reason
    ('why' -- what you're trying to figure out, e.g. your current research
    direction or hypothesis). Cached by title, so reading the same paper again
    (from any query that resolves to it) is instant and doesn't re-download it."""
    base_folder = ctx.context.knowledge_bank_dir or ctx.context.workdir
    return _read_paper_in_depth_impl(base_folder, ctx.context.model, query_or_path, why)


@function_tool
def run_experiment_task(ctx: RunContextWrapper[ResearchContext], task: str) -> str:
    """Give Codex a concrete coding/experiment task to carry out (write code,
    run it, fix errors, report results) in your working directory. Codex can
    read/write any file there and run shell commands. Be specific: what to
    implement, what data/model to use, what to measure, what file(s) to save
    results/plots to. Codex has no memory of previous calls -- restate any
    context it needs (e.g. what to fix if the last attempt failed)."""
    return run_coding_task(
        task,
        ctx.context.workdir,
        worker=ctx.context.worker,
        timeout=ctx.context.codex_timeout,
        codex_profile=ctx.context.codex_profile,
    )


@function_tool
def list_experiment_files(ctx: RunContextWrapper[ResearchContext]) -> str:
    """List all files Codex has produced so far in your working directory
    (results, plots, logs), so you know what's available to inspect."""
    return list_workdir_files(ctx.context.workdir)


@function_tool
def read_experiment_file(ctx: RunContextWrapper[ResearchContext], relative_path: str) -> str:
    """Read a text file (results.json, a log, etc.) from your working
    directory, given a path relative to it."""
    return read_text_file(osp.join(ctx.context.workdir, relative_path))


@function_tool
def inspect_plot(ctx: RunContextWrapper[ResearchContext], relative_path: str, question: str) -> str:
    """Ask a question about a plot/figure image Codex produced (e.g. 'does the
    training loss converge?', 'is there anything wrong with this figure?').
    relative_path is relative to your working directory."""
    return describe_plot(osp.join(ctx.context.workdir, relative_path), question, model=ctx.context.model)


def _recall_prior_rounds_impl(loop_dir: str, query: str, field_chars: int = 500) -> str:
    outcomes = sorted(
        f for f in os.listdir(loop_dir)
        if f.startswith("round_") and f.endswith("_outcome.json")
    )
    if not outcomes:
        return "No prior development rounds recorded yet."

    def _clip(text: str) -> str:
        text = str(text)
        return text if len(text) <= field_chars else text[:field_chars] + " …"

    terms = [t for t in query.lower().split() if len(t) > 2]
    digests = []
    for fname in outcomes:
        try:
            with open(osp.join(loop_dir, fname)) as f:
                payload = json.load(f)
        except Exception:
            continue
        report = payload.get("report", {}) or {}
        verdict = payload.get("verdict", {}) or {}
        blob_for_match = json.dumps(payload).lower()
        matched = any(t in blob_for_match for t in terms) if terms else False
        lines = [
            f"### {fname.replace('_outcome.json', '')}  (verdict: {verdict.get('decision', '?')})",
            f"Summary: {_clip(report.get('summary', ''))}",
        ]
        if report.get("key_results"):
            lines.append("Key results: " + _clip("; ".join(map(str, report["key_results"]))))
        if report.get("completed_steps"):
            lines.append("Completed: " + _clip("; ".join(map(str, report["completed_steps"]))))
        if report.get("dead_ends"):
            lines.append("Dead ends (do NOT repeat): " + _clip("; ".join(map(str, report["dead_ends"]))))
        if report.get("next_steps"):
            lines.append("Next steps it suggested: " + _clip("; ".join(map(str, report["next_steps"]))))
        if verdict.get("reasoning"):
            lines.append("Evaluator reasoning: " + _clip(verdict["reasoning"]))
        digests.append((matched, "\n".join(lines)))

    # Rounds matching the query first, so a targeted recall surfaces them on top.
    digests.sort(key=lambda d: not d[0])
    header = (
        f"Prior development rounds relevant to {query!r} (matches first):\n"
        if terms else "All prior development rounds:\n"
    )
    return header + "\n\n".join(d[1] for d in digests)


@function_tool
def recall_prior_rounds(ctx: RunContextWrapper[ResearchContext], query: str) -> str:
    """Look up what happened in earlier development rounds of THIS idea: their
    results, what already worked, what was tried and failed (dead ends), the next
    steps they suggested, and the evaluator's verdict. Your kickoff only carries a
    short headline, so use this whenever you need the detail of prior work instead
    of assuming or redoing it. `query` is what you're trying to recall (a method, a
    metric, 'why did stage 2 fail', etc.); leave it empty to list every round."""
    loop_dir = ctx.context.loop_dir
    if not loop_dir or not osp.isdir(loop_dir):
        return "No prior-round history is available for this run."
    return _recall_prior_rounds_impl(loop_dir, query or "")
