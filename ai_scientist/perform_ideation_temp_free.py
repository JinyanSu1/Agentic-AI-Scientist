"""Generates research ideas via a genuine multi-agent debate: a Proposer drafts
(reading seed papers and searching literature as real tool calls, not a Python
pre-fetch step), a Challenger critiques it, and an Evaluator decides whether to
lock the idea in or send it back for another revision round. Built on the
openai-agents SDK (Agent/Runner/handoff/structured outputs) rather than a
hand-rolled ACTION/ARGUMENTS text parser, so tool-calling, multi-turn handling,
and structured output are all the framework's job, not ours.
"""

import argparse
import json
import os
import os.path as osp
import time
import traceback
from typing import Any, Dict, List, Optional

import tiktoken
from agents import Agent, handoff
from pydantic import BaseModel

from ai_scientist.agents_common import (
    ResearchContext,
    configure_model_provider,
    read_paper_in_depth,
    run_agent_with_retry,
    search_literature,
)

_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKEN_ENCODING.encode(text))


# --- Structured idea/critique/verdict shapes -------------------------------


class ExperimentPlan(BaseModel):
    datasets: List[str]
    baselines: List[str]
    metrics: List[str]
    compute_estimate: str
    steps: List[str]


class IdeaDraft(BaseModel):
    name: str
    title: str
    short_hypothesis: str
    related_work: str
    abstract: str
    experiments: ExperimentPlan
    risk_factors_and_limitations: List[str]


class Assessment(BaseModel):
    assessment: str
    severity: str  # "low", "medium", or "high"


class Critique(BaseModel):
    novelty: Assessment
    feasibility: Assessment
    resource_requirements: Assessment
    failure_modes: Assessment


class Verdict(BaseModel):
    decision: str  # "lock" or "continue"
    unresolved_issues: List[str]
    reasoning: str


def idea_draft_to_dict(draft: IdeaDraft) -> Dict[str, Any]:
    """Convert to the plain-dict shape the rest of the codebase (idea_to_markdown,
    agent_manager.py, perform_icbinb_writeup.py) expects."""
    return {
        "Name": draft.name,
        "Title": draft.title,
        "Short Hypothesis": draft.short_hypothesis,
        "Related Work": draft.related_work,
        "Abstract": draft.abstract,
        "Experiments": {
            "datasets": draft.experiments.datasets,
            "baselines": draft.experiments.baselines,
            "metrics": draft.experiments.metrics,
            "compute_estimate": draft.experiments.compute_estimate,
            "steps": draft.experiments.steps,
        },
        "Risk Factors and Limitations": draft.risk_factors_and_limitations,
    }


# --- Agent instructions ------------------------------------------------------

NO_WORKSHOP_TOPIC_FRAMING = (
    "There is no fixed workshop topic to fit. Propose a novel, high-impact "
    "research idea in the general area of the seed papers below -- let the "
    "papers themselves define the space to work in, rather than fitting any "
    "external theme."
)

PROPOSER_INSTRUCTIONS_TEMPLATE = """You are an experienced AI researcher who proposes high-impact research ideas resembling exciting grant proposals. Be creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis. Clearly clarify how the proposal distinguishes from existing literature.

Ensure the proposal does not require resources beyond what an academic lab could afford. It should lead to a paper publishable at a top ML conference.

{topic_framing}
{seed_papers_section}
Before proposing, use the read_paper_in_depth tool to read the seed papers above (pass the reason you're reading each one as `why`), and use search_literature for anything else you need to check. Do not propose cold without reading them first.

Your idea's "Experiments" section must be a genuinely detailed, executable plan -- specific dataset names, specific baselines, specific metrics, a compute estimate, and an ordered list of concrete steps. Not prose, and not vague ("implement the method and evaluate it" is not acceptable).

Once you have a solid idea (or, on a revision round, once you've addressed the feedback below), hand it off to the Challenger with the full idea draft.
{revision_context}"""

CHALLENGER_INSTRUCTIONS = """You are a skeptical, rigorous reviewer whose job is to find real problems with a proposed AI research idea before it wastes compute and time. You are not trying to be agreeable -- you are trying to prevent a weak idea from being executed. Be specific and concrete; vague objections are not useful.

You will receive an idea draft via handoff. Critique it on exactly these four dimensions, each with a severity (low/medium/high):
- novelty: Is this meaningfully different from existing work, or a trivial extension? Use search_literature if you need to check.
- feasibility: Can this actually be executed as described, with the methods/data available?
- resource_requirements: What compute/data/time would this realistically take, and is that affordable for an academic lab?
- failure_modes: What is most likely to go wrong when this is actually run?

Once your critique is ready, hand off to the Evaluator with the full critique."""

EVALUATOR_INSTRUCTIONS = """You are an objective judge overseeing a debate between a Proposer and a Challenger about an AI research idea. You do not propose or critique the idea yourself -- you evaluate whether the proposer's idea (visible above, from the handoff history) has, in light of the Challenger's critique (also visible above), substantively addressed the real concerns, or whether serious unresolved issues remain.

Decide "lock" only when the challenger's most serious concerns are genuinely addressed (not just acknowledged) by the idea as drafted. Decide "continue" if concrete, unresolved issues remain -- list them specifically in unresolved_issues so the proposer knows exactly what to fix next round. Be decisive -- do not send it back over issues that are already adequately handled.

Always end by producing your verdict."""


def _build_agents(
    topic_framing: str, seed_papers_section: str, revision_context: str, model: str = "fugu"
):
    # Handoffs pass the full conversation history forward by default, so a
    # later agent in the chain can see earlier tool calls (e.g. the Proposer
    # reading a seed paper) and try to call the same tool itself. Every agent
    # in the chain gets the same tool set so that never raises a
    # ModelBehaviorError for an undefined tool.
    common_tools = [read_paper_in_depth, search_literature]

    evaluator = Agent(
        name="Evaluator",
        instructions=EVALUATOR_INSTRUCTIONS,
        model=model,
        tools=common_tools,
        output_type=Verdict,
    )
    challenger = Agent(
        name="Challenger",
        instructions=CHALLENGER_INSTRUCTIONS,
        model=model,
        tools=common_tools,
    )
    proposer = Agent(
        name="Proposer",
        instructions=PROPOSER_INSTRUCTIONS_TEMPLATE.format(
            topic_framing=topic_framing,
            seed_papers_section=seed_papers_section,
            revision_context=revision_context,
        ),
        model=model,
        tools=common_tools,
    )

    draft_box: Dict[str, IdeaDraft] = {}
    critique_box: Dict[str, Critique] = {}

    def capture_draft(_ctx, input_data: IdeaDraft) -> None:
        draft_box["value"] = input_data

    def capture_critique(_ctx, input_data: Critique) -> None:
        critique_box["value"] = input_data

    challenger.handoffs = [
        handoff(evaluator, input_type=Critique, on_handoff=capture_critique)
    ]
    proposer.handoffs = [
        handoff(challenger, input_type=IdeaDraft, on_handoff=capture_draft)
    ]

    return proposer, draft_box, critique_box


def run_debate_round(
    topic_framing: str,
    seed_papers_section: str,
    revision_context: str,
    research_ctx: ResearchContext,
    max_turns: int = 20,
    model: str = "fugu",
) -> tuple[Optional[IdeaDraft], Optional[Critique], Verdict]:
    proposer, draft_box, critique_box = _build_agents(
        topic_framing, seed_papers_section, revision_context, model=model
    )
    result = run_agent_with_retry(
        proposer,
        "Begin." if not revision_context else "Revise your idea per the feedback above.",
        context=research_ctx,
        max_turns=max_turns,
    )
    verdict = result.final_output
    if not isinstance(verdict, Verdict):
        raise RuntimeError(
            f"Debate round did not end with the Evaluator's Verdict (got {type(verdict)}); "
            "the chain likely didn't reach a handoff/final answer within max_turns."
        )
    return draft_box.get("value"), critique_box.get("value"), verdict


def _persist_round(
    transcript_dir: str,
    round_idx: int,
    draft: Optional[IdeaDraft],
    critique: Optional[Critique],
    verdict: Verdict,
) -> None:
    payload = {
        "round": round_idx,
        "draft": draft.model_dump() if draft else None,
        "critique": critique.model_dump() if critique else None,
        "verdict": verdict.model_dump(),
    }
    with open(osp.join(transcript_dir, f"round_{round_idx:02d}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def run_debate_for_idea(
    workshop_description: str,
    seed_papers: Optional[List[str]],
    prev_ideas_string: str,
    max_debate_rounds: int,
    transcript_dir: str,
    model: str = "fugu",
    agent_model: Any = None,
    knowledge_bank_dir: Optional[str] = None,
) -> Dict[str, Any]:
    # agent_model is what configure_model_provider(model) returned -- may differ
    # from the plain `model` string (e.g. a LitellmModel instance for Anthropic).
    # ResearchContext needs the string (other tools do their own provider dispatch
    # from it); the debate Agents need agent_model. Defaults to `model` so this
    # function is still usable standalone without a separate resolution step.
    agent_model = agent_model if agent_model is not None else model
    os.makedirs(transcript_dir, exist_ok=True)
    research_ctx = ResearchContext(
        workdir=transcript_dir, model=model, knowledge_bank_dir=knowledge_bank_dir
    )

    topic_framing = workshop_description.strip() or NO_WORKSHOP_TOPIC_FRAMING
    if prev_ideas_string.strip():
        topic_framing += (
            "\n\nProposals you have already generated in previous debates (propose "
            f"something that differs from these):\n'''\n{prev_ideas_string}\n'''"
        )

    seed_papers_section = ""
    if seed_papers:
        seed_papers_section = (
            "\nSeed papers to read before proposing (local paths or search "
            f"queries): {seed_papers}\n"
        )

    revision_context = ""
    last_draft: Optional[IdeaDraft] = None
    last_verdict: Optional[Verdict] = None
    locked = False
    round_idx = 0
    for round_idx in range(max_debate_rounds):
        draft, critique, verdict = run_debate_round(
            topic_framing, seed_papers_section, revision_context, research_ctx, model=agent_model
        )
        _persist_round(transcript_dir, round_idx, draft, critique, verdict)
        if draft is not None:
            last_draft = draft
        last_verdict = verdict

        print(
            f"Round {round_idx}: decision={verdict.decision} reasoning={verdict.reasoning[:200]}"
        )

        if verdict.decision == "lock":
            locked = True
            break

        revision_context = (
            "\nFEEDBACK FROM PREVIOUS ROUND -- address this directly:\n"
            f"Your previous draft: {draft.model_dump() if draft else '(none captured)'}\n"
            f"Challenger's critique: {critique.model_dump() if critique else '(none captured)'}\n"
            f"Evaluator's reasoning for sending it back: {verdict.reasoning}\n"
            f"Unresolved issues to fix: {verdict.unresolved_issues}\n"
        )

    if last_draft is None:
        raise RuntimeError(
            f"Debate in {transcript_dir} never produced a captured idea draft "
            "(the Proposer->Challenger handoff may not have fired)."
        )

    final_idea = idea_draft_to_dict(last_draft)
    final_idea["_debate"] = {
        "rounds": round_idx + 1,
        "locked": locked,
        "flagged": not locked,
        "transcript_dir": transcript_dir,
    }
    if not locked and last_verdict is not None:
        final_idea["_debate"]["last_unresolved_issues"] = last_verdict.unresolved_issues
    return final_idea


def derive_seed_queries_from_workshop(
    model: str, workshop_description: str, max_queries: int = 2
) -> List[str]:
    """When no explicit --seed-papers are given, derive a couple of literature
    search queries from the workshop topic itself, so ideation still starts from
    having surveyed the sub-field instead of proposing cold."""

    class SurveyQueries(BaseModel):
        queries: List[str]

    try:
        agent = Agent(
            name="SurveyPlanner",
            instructions=(
                "You are an AI researcher about to propose ideas for a workshop "
                "topic. Before proposing anything, propose up to "
                f"{max_queries} literature search queries that would surface the "
                "most important/representative recent papers for this topic, so "
                "they can be read in full before proposing an idea. Prefer "
                "queries likely to surface well-known, influential papers over "
                "obscure ones."
            ),
            model=model,
            output_type=SurveyQueries,
        )
        result = run_agent_with_retry(
            agent, f"Workshop topic:\n{workshop_description}", max_turns=3
        )
        return result.final_output.queries[:max_queries]
    except Exception as e:
        print(f"Failed to derive survey queries from workshop description: {e}")
        return []


def generate_temp_free_idea(
    idea_fname: str,
    workshop_description: str,
    max_num_generations: int = 1,
    max_debate_rounds: int = 12,
    reload_ideas: bool = True,
    seed_papers: Optional[List[str]] = None,
    model: str = "fugu",
    knowledge_bank_dir: Optional[str] = None,
) -> List[Dict]:
    agent_model = configure_model_provider(model)

    ideas: List[Dict[str, Any]] = []
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            ideas = json.load(f)
        print(f"Loaded {len(ideas)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    debates_root = osp.join(
        "experiments", "idea_debates", osp.splitext(osp.basename(idea_fname))[0]
    )

    seed_queries = seed_papers
    if not seed_queries and workshop_description.strip():
        seed_queries = derive_seed_queries_from_workshop(agent_model, workshop_description)
        if seed_queries:
            print(f"No --seed-papers given; auto-derived survey queries: {seed_queries}")

    for gen_idx in range(max_num_generations):
        print(f"\nRunning debate for idea {gen_idx + 1}/{max_num_generations}")
        prev_ideas_string = "\n\n".join(
            json.dumps({k: v for k, v in idea.items() if k != "_debate"})
            for idea in ideas
        )
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        transcript_dir = osp.join(debates_root, f"idea_{gen_idx:02d}_{timestamp}")
        try:
            idea = run_debate_for_idea(
                workshop_description=workshop_description,
                seed_papers=seed_queries,
                prev_ideas_string=prev_ideas_string,
                max_debate_rounds=max_debate_rounds,
                transcript_dir=transcript_dir,
                model=model,
                agent_model=agent_model,
                knowledge_bank_dir=knowledge_bank_dir,
            )
            ideas.append(idea)
            status = "locked" if idea["_debate"]["locked"] else "FLAGGED (unresolved at round cap)"
            print(f"Idea {gen_idx + 1} finished: {status}. Transcript: {transcript_dir}")
        except Exception:
            print(f"Failed to generate idea {gen_idx + 1}:")
            traceback.print_exc()
            continue

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals via a proposer/challenger/evaluator debate"
    )
    parser.add_argument("--model", type=str, default="fugu")
    parser.add_argument("--max-num-generations", type=int, default=1)
    parser.add_argument(
        "--workshop-file",
        type=str,
        default=None,
        help="Optional workshop description file. If omitted, --seed-papers alone "
        "defines the space to propose in.",
    )
    parser.add_argument("--max-debate-rounds", type=int, default=12)
    parser.add_argument(
        "--seed-papers",
        type=str,
        nargs="+",
        default=None,
        help="Local PDF paths and/or search queries for seed papers to read "
        "before debating, so the proposer starts grounded instead of cold.",
    )
    args = parser.parse_args()

    if args.workshop_file:
        with open(args.workshop_file, "r") as f:
            workshop_description = f.read()
        print(f"Using workshop description from {args.workshop_file} for idea generation.")
        idea_fname = args.workshop_file.replace(".md", ".json")
    else:
        if not args.seed_papers:
            raise ValueError("Provide --workshop-file and/or --seed-papers.")
        workshop_description = ""
        print("No --workshop-file given; proposing freely in the area of --seed-papers.")
        idea_fname = "ideas/seed_paper_ideas.json"

    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        max_debate_rounds=args.max_debate_rounds,
        seed_papers=args.seed_papers,
        model=args.model,
    )
    print(f"{idea_fname} generated {len(ideas)} ideas.")
