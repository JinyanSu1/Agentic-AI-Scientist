"""Replaces BFTS's hand-rolled tree-search (fixed stages, fixed max_iters,
Node/Journal machinery) with a single agent that owns the whole experiment
lifecycle: implement, run, look at real results, and decide what to do next --
fix a bug, try a baseline, scale up, try an ablation, or stop -- entirely on its
own judgment, with no fixed stage structure or iteration cap baked in by us.

The actual code-writing/running/debugging is delegated to a CLI coding worker
(Codex or Claude Code -- see ai_scientist/tools/coding_worker.py) via
run_experiment_task; this agent is the "brain" deciding what to ask the worker
to do and when enough evidence has been gathered.
"""

import json
import os.path as osp
from typing import Any, Dict, List

from agents import Agent
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel

from ai_scientist.agents_common import (
    ResearchContext,
    configure_model_provider,
    inspect_plot,
    list_experiment_files,
    read_experiment_file,
    read_paper_in_depth,
    recall_prior_rounds,
    run_agent_with_retry,
    run_experiment_task,
    search_literature,
)
from ai_scientist.tools.codex_worker import list_workdir_files, read_text_file


class ExperimentReport(BaseModel):
    status: str  # "completed", "abandoned", or "incomplete" (ran out of step budget)
    summary: str  # what was done and found -- this feeds the paper writeup
    key_results: List[str]  # specific, concrete findings/metrics, one per item
    files_of_interest: List[str]  # paths (relative to workdir) worth citing/plotting in the paper
    # --- Handoff to the next development round (and the reviser). The next round
    # continues in the SAME working directory, so these carry the useful residue
    # of this round's work forward without replaying the whole transcript. ---
    working_assets: List[str] = []  # files that already work + one line on what each is/does
    completed_steps: List[str] = []  # stages/experiments actually finished with real results
    dead_ends: List[str] = []  # approaches tried that failed, and why (so the next round won't repeat them)
    next_steps: List[str] = []  # concrete next actions for whoever continues


RESEARCH_AGENT_INSTRUCTIONS = """You are an experienced empirical ML researcher actually carrying out a research idea from implementation through to a defensible conclusion, entirely on your own judgment. There are no fixed stages, and no iteration count you must follow -- that decision is yours.

Your idea:
```json
{idea_json}
```

You have a working directory where Codex (via run_experiment_task) will actually write and run code for you -- Codex has no memory between calls, so give it a complete, self-contained task each time, including what to fix if the last attempt failed. Inspect what it produces with list_experiment_files / read_experiment_file / inspect_plot before deciding your next move. Use search_literature / read_paper_in_depth if you need to check something in the literature.

This may be a continuation: your working directory can already contain code, data, and results from previous rounds. Your kickoff carries only a short headline (where the last round left off and the immediate next steps) -- it deliberately does NOT dump the full prior history. When you need the detail of earlier rounds (their full results, what already works, what was tried and failed, the evaluator's verdicts), call recall_prior_rounds to retrieve it on demand, and list/read the working directory to confirm what code and result files are actually there. BUILD ON the existing work: reuse assets that already work, do not re-implement or re-run completed steps, do not repeat approaches recorded as dead ends, and pick up from the next steps.

Work iteratively: implement, run, look at the actual results, and decide what to do next based on what you see -- fix bugs, add baselines, scale up, try alternative approaches, run ablations, whatever the results actually call for. Don't pad with unnecessary steps once you have a clear, well-supported answer to your hypothesis (positive or negative). Don't declare success on a single lucky run where variance matters -- make sure your evidence would survive scrutiny (sensible baselines, no obvious bugs, replication where it matters). Prefer to fully nail a small, clean core result over spreading yourself thin across an over-ambitious protocol you can't finish.

When you're done -- either because you have a solid answer, or because you've hit a genuine dead end worth reporting honestly -- write a clear final summary as your last message. You do NOT need to format it as JSON or fill any fields: a separate step reads your summary plus the actual result files and turns it into the structured report. In that summary cover: what you did; the concrete results with actual numbers; which files in the working directory hold the real results and which already work; what you tried that failed and why; and what a next round should do. A clean, well-supported negative result is a real result, not a failure; a confused or buggy experiment is not, even if something looked promising at one point. Ground every claim in what actually ran -- never state a result you did not produce."""


VALID_STATUSES = ("completed", "abandoned", "incomplete")

REPORT_SYNTHESIS_INSTRUCTIONS = """You write the final ExperimentReport for one research round, from clean context -- so it is accurate and well-formed even though the researcher's own working session was long. You are given: the idea, the researcher's own final notes (which may be empty if they ran out of their step budget before summarizing), and a listing plus the contents of the actual result/log files in the working directory. Produce an honest report grounded ONLY in this evidence -- do NOT invent results.

- status: exactly one of "completed" / "incomplete" / "abandoned". "completed" if there is a solid, well-supported answer to the hypothesis (a clean positive OR a clean negative result both count). "incomplete" if the round did not reach an evaluable result (e.g. it was cut off, or produced no evaluable evidence yet). "abandoned" only if the evidence shows the idea's premise is genuinely broken.
- summary and key_results: report only what the files and notes substantiate, with concrete numbers wherever available.
- files_of_interest: result/plot files worth citing or plotting in the paper.
- working_assets / completed_steps / dead_ends / next_steps: a handoff so whoever continues in this same directory can build on the work rather than start over."""


def synthesize_report(
    workdir: str,
    idea: Dict[str, Any],
    model: str,
    agent_narrative: str = "",
    max_files: int = 10,
    per_file_chars: int = 2000,
) -> ExperimentReport:
    """Produce the ExperimentReport in a SEPARATE, short, clean call rather than as
    the research agent's final in-conversation turn. The long research conversation
    (dozens of turns of Codex output near the context limit) is exactly where fugu's
    structured output degrades into garbage; a fresh call over just the workdir files
    + the agent's free-form notes produces a clean report (verified: the same model
    garbles the in-conversation final turn but returns clean JSON from clean context).
    agent_narrative may be empty (e.g. the round hit its turn cap before summarizing);
    the report is then synthesized from the on-disk evidence alone."""
    listing = list_workdir_files(workdir)
    contents = []
    for line in listing.splitlines():
        rel = line.rsplit(" (", 1)[0].strip()
        if rel.lower().endswith((".json", ".txt", ".log", ".md", ".csv")):
            contents.append(f"### {rel}\n{read_text_file(osp.join(workdir, rel), max_chars=per_file_chars)}")
        if len(contents) >= max_files:
            break
    blob = "\n\n".join(contents) if contents else "(no readable result/log files found)"

    agent = Agent(
        name="ReportSynthesizer",
        instructions=REPORT_SYNTHESIS_INSTRUCTIONS,
        model=model,
        output_type=ExperimentReport,
    )
    prompt = (
        f"IDEA:\n```json\n{json.dumps(idea, indent=2)}\n```\n\n"
        f"RESEARCHER'S FINAL NOTES:\n{agent_narrative.strip() or '(none -- the round ended before a summary was written)'}\n\n"
        f"WORKING DIRECTORY LISTING:\n{listing}\n\n"
        f"FILE CONTENTS:\n{blob}"
    )
    try:
        report = run_agent_with_retry(agent, prompt, max_turns=2).final_output
    except Exception:
        return ExperimentReport(
            status="incomplete",
            summary="A report could not be synthesized from the working directory "
            "(the synthesis call failed).",
            key_results=[],
            files_of_interest=[],
            next_steps=["Inspect the working directory manually and continue from there."],
        )
    if report.status not in VALID_STATUSES:
        # Even the clean synthesis emitted an out-of-vocabulary status -- coerce it
        # rather than let a garbage status reach the evaluator.
        report.status = "incomplete"
    return report


def run_research_agent(
    idea: Dict[str, Any],
    workdir: str,
    max_turns: int = 60,
    model: str = "fugu",
    knowledge_bank_dir: str = None,
    prior_context: str = "",
    loop_dir: str = None,
    codex_timeout: int = 3600,
    worker: str = "codex",
    codex_profile: str = "fugu",
) -> ExperimentReport:
    """Run one development round. The agent works free-form (no output_type): its
    final message is a plain-text summary, NOT the structured report. The
    ExperimentReport is then produced by synthesize_report in a SEPARATE clean call
    over the workdir files + that summary -- because the report garbles specifically
    when emitted as the final turn of a long, heavy conversation, while a fresh clean
    call produces a well-formed one. If the agent hits its turn cap, we synthesize
    from the on-disk evidence alone (no narrative). prior_context is a short handoff
    headline; the full prior history is reachable via recall_prior_rounds."""
    agent_model = configure_model_provider(model)
    agent = Agent(
        name="ResearchAgent",
        instructions=RESEARCH_AGENT_INSTRUCTIONS.format(idea_json=json.dumps(idea, indent=2)),
        model=agent_model,
        tools=[
            run_experiment_task,
            list_experiment_files,
            read_experiment_file,
            inspect_plot,
            recall_prior_rounds,
            search_literature,
            read_paper_in_depth,
        ],
        # No output_type on purpose -- the agent ends with a free-form summary and
        # the structured report is synthesized separately (see synthesize_report).
    )
    ctx = ResearchContext(
        workdir=workdir, model=model, knowledge_bank_dir=knowledge_bank_dir,
        loop_dir=loop_dir, codex_timeout=codex_timeout,
        worker=worker, codex_profile=codex_profile,
    )
    kickoff = "Begin implementing and running this research idea."
    if prior_context:
        kickoff = (
            prior_context
            + "\n\nContinue this work in the same working directory: reuse what already "
            "works, don't redo completed steps or repeat dead ends, and pick up from the "
            "next steps above. Confirm the directory contents first, then proceed."
        )
    # A retry re-runs the whole conversation (Codex calls included) from scratch --
    # expensive late, but cheaper than losing the run to one bad generation. On
    # MaxTurnsExceeded we don't re-run (it would just hit the cap again) -- we
    # synthesize the report from what's already on disk, with no agent narrative.
    try:
        result = run_agent_with_retry(agent, kickoff, context=ctx, max_turns=max_turns)
        narrative = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
    except MaxTurnsExceeded:
        print(
            f"Research agent hit the {max_turns}-turn cap; synthesizing the report from "
            "the working directory (no final summary was written)."
        )
        narrative = ""
    return synthesize_report(workdir, idea, agent_model, agent_narrative=narrative)
