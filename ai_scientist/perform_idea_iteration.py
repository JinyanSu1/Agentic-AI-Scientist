"""After the Research Agent produces an ExperimentReport for an idea, an
Evaluator decides whether the idea + results are ready to write up, need
revision, or should be abandoned -- using the same openai-agents structured-
output pattern as the ideation debate (Verdict/IdeaDraft), not a hand-rolled
text parser. If revision is needed, a Proposer-style agent revises the idea
directly against what the experiment actually showed.
"""

import json
from typing import Any, Dict, List

from agents import Agent
from pydantic import BaseModel

from ai_scientist.agents_common import (
    ResearchContext,
    configure_fugu_as_default,
    read_paper_in_depth,
    run_agent_with_retry,
    search_literature,
)
from ai_scientist.perform_ideation_temp_free import (
    IdeaDraft,
    Verdict,
    idea_draft_to_dict,
)
from ai_scientist.perform_research_agent import ExperimentReport

EXPERIMENT_EVALUATOR_INSTRUCTIONS = """You are an objective judge deciding whether an AI research idea, after actually being implemented and experimented on, is ready to be written up as a FULL-LENGTH paper for a top venue (ACL Rolling Review) -- not a short workshop note.

Judge based on what the experiment report actually shows -- not on how ambitious or interesting the idea sounded originally. A clean, well-supported result (positive OR negative) is only lockable if it is thoroughly established, not the product of a single quick exploratory run.

Hold a full-paper bar for "lock": the study must have enough substantiated empirical content to fill a full-length paper with a rich appendix -- typically sensible baselines, ablations of the key components, more than one setting/model/dataset where feasible, adequate replication (multiple seeds) for the central claims, and robustness/sanity checks. A thin, single-setting, or pilot-level result -- even if clean -- is NOT ready; send it back to be deepened.

Decide "lock" (thorough enough to write up as a full paper), "revise" (the core idea is worth keeping but the empirical study must be deepened or corrected based on what the experiment revealed -- say concretely what more is needed), or "abandon" (the results show the idea's premise is broken or uninteresting; a fresh idea should be tried instead).

Always end by producing your verdict."""

PROPOSER_REVISE_INSTRUCTIONS_TEMPLATE = """You are revising a research idea based on what actually happened when it was implemented and run.

ORIGINAL IDEA:
```json
{idea_json}
```

EXPERIMENT REPORT:
Status: {status}
Summary: {summary}
Key results: {key_results}

ALREADY COMPLETED (keep these; the next round continues in the same workdir and will reuse them):
{completed_steps}

TRIED AND FAILED (do not re-introduce these as if they were new ideas):
{dead_ends}

EVALUATOR'S REASONING FOR REVISION:
{reasoning}

UNRESOLVED ISSUES TO ADDRESS:
{unresolved_issues}

Revise the idea to directly address what the experiment revealed. Stick to the spirit of the original idea unless the results show a genuine flaw that requires a different approach. Keep the scope executable: prefer tightening or fixing the existing plan over expanding it with more models/baselines/stages the agent may not be able to finish in its budget. If the current results already answer the core hypothesis cleanly -- even negatively -- lean toward a minimal revision rather than a larger protocol. Produce your revised idea."""


class NoveltyVerdict(BaseModel):
    is_novel: bool
    closest_prior_work: str  # the single closest existing work found, or "(none found)"
    differentiation: str  # how the idea differs from that work; say so honestly if trivial
    concurrent_work_found: List[str]  # very recent (last 3-6 months) work that might have scooped it
    reasoning: str


NOVELTY_CHECK_INSTRUCTIONS = """You are doing a thorough, skeptical novelty verification for a research idea that has already shown a positive pilot signal -- before real development effort is committed to it, verify it isn't already published.

Search extensively -- at least 3 distinct queries covering different phrasings/angles of the idea, using search_literature, and read_paper_in_depth on anything that looks like a close match -- before concluding. Do not rely on your own background knowledge alone.

Then, explicitly:
1. Identify the SINGLE closest existing work to this idea, if any -- name it specifically (title/authors).
2. Explain precisely how this idea differs from that closest work. If the differentiation is trivial or cosmetic, say so honestly rather than talking yourself into novelty.
3. Check specifically for very recent (last 3-6 months) concurrent work that might have already scooped this idea.
4. Conclude is_novel: true only if the idea is genuinely distinct from what you found, not already published, and not obviously scooped by concurrent work."""


def verify_novelty(
    idea: Dict[str, Any], workdir: str, model: str = "fugu", knowledge_bank_dir: str = None
) -> NoveltyVerdict:
    configure_fugu_as_default(model)
    agent = Agent(
        name="NoveltyVerifier",
        instructions=NOVELTY_CHECK_INSTRUCTIONS,
        model=model,
        tools=[search_literature, read_paper_in_depth],
        output_type=NoveltyVerdict,
    )
    ctx = ResearchContext(workdir=workdir, model=model, knowledge_bank_dir=knowledge_bank_dir)
    prompt = (
        f"IDEA:\n```json\n{json.dumps(idea, indent=2)}\n```\n\n"
        "Verify novelty thoroughly before we commit real development effort to this."
    )
    result = run_agent_with_retry(agent, prompt, context=ctx, max_turns=12)
    return result.final_output


def evaluate_experiment(
    idea: Dict[str, Any], report: ExperimentReport, model: str = "fugu"
) -> Verdict:
    configure_fugu_as_default(model)
    agent = Agent(
        name="ExperimentEvaluator",
        instructions=EXPERIMENT_EVALUATOR_INSTRUCTIONS,
        model=model,
        output_type=Verdict,
    )
    prompt = (
        f"IDEA:\n```json\n{json.dumps(idea, indent=2)}\n```\n\n"
        f"EXPERIMENT REPORT:\nStatus: {report.status}\nSummary: {report.summary}\n"
        f"Key results: {report.key_results}\nFiles of interest: {report.files_of_interest}"
    )
    result = run_agent_with_retry(agent, prompt, max_turns=3)
    return result.final_output


def revise_idea_from_experiment(
    idea: Dict[str, Any], report: ExperimentReport, verdict: Verdict, model: str = "fugu"
) -> Dict[str, Any]:
    configure_fugu_as_default(model)
    agent = Agent(
        name="Proposer",
        instructions=PROPOSER_REVISE_INSTRUCTIONS_TEMPLATE.format(
            idea_json=json.dumps(idea, indent=2),
            status=report.status,
            summary=report.summary,
            key_results=report.key_results,
            completed_steps=getattr(report, "completed_steps", None) or "(none reported)",
            dead_ends=getattr(report, "dead_ends", None) or "(none reported)",
            reasoning=verdict.reasoning,
            unresolved_issues=verdict.unresolved_issues,
        ),
        model=model,
        output_type=IdeaDraft,
    )
    result = run_agent_with_retry(agent, "Revise the idea now.", max_turns=3)
    return idea_draft_to_dict(result.final_output)
