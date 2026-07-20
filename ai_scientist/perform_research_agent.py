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
from typing import Any, Dict, List

from agents import Agent
from pydantic import BaseModel

from ai_scientist.agents_common import (
    ResearchContext,
    configure_fugu_as_default,
    inspect_plot,
    list_experiment_files,
    read_experiment_file,
    read_paper_in_depth,
    run_agent_with_retry,
    run_experiment_task,
    search_literature,
)


class ExperimentReport(BaseModel):
    status: str  # "completed" or "abandoned"
    summary: str  # what was done and found -- this feeds the paper writeup
    key_results: List[str]  # specific, concrete findings/metrics, one per item
    files_of_interest: List[str]  # paths (relative to workdir) worth citing/plotting in the paper


RESEARCH_AGENT_INSTRUCTIONS = """You are an experienced empirical ML researcher actually carrying out a research idea from implementation through to a defensible conclusion, entirely on your own judgment. There are no fixed stages, and no iteration count you must follow -- that decision is yours.

Your idea:
```json
{idea_json}
```

You have a working directory where Codex (via run_experiment_task) will actually write and run code for you -- Codex has no memory between calls, so give it a complete, self-contained task each time, including what to fix if the last attempt failed. Inspect what it produces with list_experiment_files / read_experiment_file / inspect_plot before deciding your next move. Use search_literature / read_paper_in_depth if you need to check something in the literature.

Work iteratively: implement, run, look at the actual results, and decide what to do next based on what you see -- fix bugs, add baselines, scale up, try alternative approaches, run ablations, whatever the results actually call for. Don't pad with unnecessary steps once you have a clear, well-supported answer to your hypothesis (positive or negative). Don't declare success on a single lucky run where variance matters -- make sure your evidence would survive scrutiny (sensible baselines, no obvious bugs, replication where it matters).

When you're done -- either because you have a solid answer, or because you've hit a genuine dead end worth reporting honestly -- produce your final ExperimentReport. A clean, well-supported negative result is a valid "completed" status; a confused or buggy experiment is not, even if something looked promising at one point."""


def run_research_agent(
    idea: Dict[str, Any], workdir: str, max_turns: int = 60, model: str = "fugu"
) -> ExperimentReport:
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
            search_literature,
            read_paper_in_depth,
        ],
        output_type=ExperimentReport,
    )
    ctx = ResearchContext(workdir=workdir, model=model)
    # A retry here re-runs the whole conversation (Codex calls included) from
    # scratch -- expensive if it happens late, but far cheaper than losing the
    # entire multi-hour pipeline run to one garbled final-answer generation.
    result = run_agent_with_retry(
        agent,
        "Begin implementing and running this research idea.",
        context=ctx,
        max_turns=max_turns,
    )
    return result.final_output
