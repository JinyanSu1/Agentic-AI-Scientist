"""Replaces BFTS's hand-rolled tree-search (fixed stages, fixed max_iters,
Node/Journal machinery) with a single agent that owns the whole experiment
lifecycle: implement, run, look at real results, and decide what to do next --
fix a bug, try a baseline, scale up, try an ablation, or stop -- entirely on its
own judgment, with no fixed stage structure or iteration cap baked in by us.

The actual code-writing/running/debugging is delegated to Codex (a proven
coding agent) via run_experiment_task; this agent is the "brain" deciding what
to ask Codex to do and when enough evidence has been gathered.
"""

import json
import os.path as osp
from typing import Any, Dict, List

from agents import Agent
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel

from ai_scientist.agents_common import (
    ResearchContext,
    configure_fugu_as_default,
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

When you're done -- either because you have a solid answer, or because you've hit a genuine dead end worth reporting honestly -- produce your final ExperimentReport. A clean, well-supported negative result is a valid "completed" status; a confused or buggy experiment is not, even if something looked promising at one point. Always fill in the handoff fields (working_assets, completed_steps, dead_ends, next_steps) honestly, as if briefing a colleague who will continue in this same directory -- these are what let the next round build on your work instead of starting over."""


REPORT_SALVAGE_INSTRUCTIONS = """You are salvaging a research run that ran out of its step budget before the researcher could write a final report. You are given a listing of the working directory and the contents of its result/log files. Produce an honest ExperimentReport from ONLY what these files actually show -- do NOT invent results. Set status to "incomplete". In summary and key_results, report only what the files substantiate, or state plainly that no evaluable results were produced yet. Fill working_assets / completed_steps / dead_ends / next_steps as a handoff so whoever continues in this same directory can pick up where this left off rather than starting over."""


def _report_has_no_evidence(report: ExperimentReport) -> bool:
    """A 'completed' report is unusable if it carries no concrete evidence at all
    -- no non-empty key results AND no files of interest. This catches the
    'well-formed but garbage/placeholder' report (WakeTrace's round-4 case) that
    returns normally without raising MaxTurnsExceeded, so such a report never
    silently reaches the evaluator as if it were a real result. A clean negative
    result still lists its finding in key_results, so this stays high-precision."""
    key_results = [k for k in (report.key_results or []) if str(k).strip()]
    files = [f for f in (report.files_of_interest or []) if str(f).strip()]
    return not key_results and not files


def _distill_partial_report(
    workdir: str, idea: Dict[str, Any], model: str, max_files: int = 8, per_file_chars: int = 2000
) -> ExperimentReport:
    """Salvage a partial ExperimentReport from what's physically on disk when the
    agent hit its turn cap before producing one -- so a budget-exhausted round
    yields an honest 'incomplete' report (and a handoff) grounded in the real
    result files, instead of a forced, garbled final answer."""
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
        name="ReportSalvager",
        instructions=REPORT_SALVAGE_INSTRUCTIONS,
        model=model,
        output_type=ExperimentReport,
    )
    prompt = (
        f"IDEA:\n```json\n{json.dumps(idea, indent=2)}\n```\n\n"
        f"WORKING DIRECTORY LISTING:\n{listing}\n\n"
        f"FILE CONTENTS:\n{blob}"
    )
    try:
        return run_agent_with_retry(agent, prompt, max_turns=2).final_output
    except Exception:
        # Even salvage failed; return a minimal honest placeholder rather than raising.
        return ExperimentReport(
            status="incomplete",
            summary="Ran out of the step budget before producing a report, and automatic "
            "salvage from the working directory also failed.",
            key_results=[],
            files_of_interest=[],
            next_steps=["Inspect the working directory manually and continue from there."],
        )


def run_research_agent(
    idea: Dict[str, Any],
    workdir: str,
    max_turns: int = 60,
    model: str = "fugu",
    knowledge_bank_dir: str = None,
    prior_context: str = "",
    loop_dir: str = None,
    codex_timeout: int = 3600,
) -> ExperimentReport:
    """Run one development round. If prior_context is given (a short handoff headline
    from the previous round, which ran in this same workdir), it's prepended to the
    kickoff so the agent continues from -- rather than restarts -- the earlier work;
    the full prior history stays out of context and is reachable via recall_prior_rounds
    (loop_dir points at the round_XX_outcome.json files). If the agent hits its turn
    cap, a partial 'incomplete' report is salvaged from the workdir instead of a
    forced, garbled final answer."""
    configure_fugu_as_default(model)
    agent = Agent(
        name="ResearchAgent",
        instructions=RESEARCH_AGENT_INSTRUCTIONS.format(idea_json=json.dumps(idea, indent=2)),
        model=model,
        tools=[
            run_experiment_task,
            list_experiment_files,
            read_experiment_file,
            inspect_plot,
            recall_prior_rounds,
            search_literature,
            read_paper_in_depth,
        ],
        output_type=ExperimentReport,
    )
    ctx = ResearchContext(
        workdir=workdir, model=model, knowledge_bank_dir=knowledge_bank_dir,
        loop_dir=loop_dir, codex_timeout=codex_timeout,
    )
    kickoff = "Begin implementing and running this research idea."
    if prior_context:
        kickoff = (
            prior_context
            + "\n\nContinue this work in the same working directory: reuse what already "
            "works, don't redo completed steps or repeat dead ends, and pick up from the "
            "next steps above. Confirm the directory contents first, then proceed."
        )
    # A non-MaxTurns retry re-runs the whole conversation (Codex calls included)
    # from scratch -- expensive if it happens late, but far cheaper than losing the
    # entire multi-hour pipeline run to one garbled final-answer generation. A
    # MaxTurnsExceeded is handled separately: rather than re-running (which would
    # just hit the cap again), salvage a partial report from what's on disk.
    try:
        report = run_agent_with_retry(agent, kickoff, context=ctx, max_turns=max_turns).final_output
    except MaxTurnsExceeded:
        print(
            f"Research agent hit the {max_turns}-turn cap before producing a report; "
            "salvaging a partial 'incomplete' report from the working directory."
        )
        return _distill_partial_report(workdir, idea, model)

    if _report_has_no_evidence(report):
        # Well-formed but empty/garbage final report -- don't let it reach the
        # evaluator as if it were a real result; salvage whatever is actually on
        # disk instead (falls back to an honest 'incomplete' if nothing is there).
        print(
            "Final report carries no concrete results or files of interest; salvaging "
            "from the working directory instead of trusting it."
        )
        return _distill_partial_report(workdir, idea, model)
    return report
