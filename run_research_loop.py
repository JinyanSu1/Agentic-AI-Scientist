"""Orchestrates the full research loop end to end, with no fixed BFTS-style
stages anywhere in the pipeline:

1. Generate a handful of candidate ideas (ideation debate, grounded in seed
   papers and/or a workshop topic, informed by the persistent research wiki so
   it doesn't repeat previously-failed directions).
2. If there's more than one candidate, run a cheap pilot on each (the Research
   Agent, tightly turn-capped) and pick the most promising by empirical signal
   -- not by how appealing the idea sounds on paper.
2.5. Deep novelty verification on the pilot winner (multiple targeted
   literature searches, explicit closest-prior-work identification, a check
   for very recent concurrent work) -- only spend this effort on the idea
   that already showed a real pilot signal, not on every candidate up front.
3. Develop the winning idea with the Research Agent (no fixed iteration count
   -- it decides what to do next and when it's done), with an Evaluator
   deciding after each attempt whether to lock it in, revise it, or abandon it
   for a fresh idea. Every outcome is logged to the research wiki.
4. On lock, hand off to the existing citation/knowledge-bank/writeup/tectonic
   pipeline for the final paper.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import time
from typing import Any, Dict, List, Optional

from agents import Agent
from pydantic import BaseModel

from ai_scientist import research_wiki
from ai_scientist.agents_common import configure_fugu_as_default, run_agent_with_retry
from ai_scientist.llm import create_client
from ai_scientist.perform_idea_iteration import (
    evaluate_experiment,
    revise_idea_from_experiment,
    verify_novelty,
)
from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea
from ai_scientist.perform_research_agent import ExperimentReport, run_research_agent
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review


def idea_to_markdown(idea: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in idea.items():
            if key.startswith("_"):
                continue
            f.write(f"## {key.replace('_', ' ').title()}\n\n")
            if isinstance(value, (list, tuple)):
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n{sub_value}\n\n")
            else:
                f.write(f"{value}\n\n")


class PilotRanking(BaseModel):
    winner_index: int
    reasoning: str


def rank_pilots(
    ideas: List[Dict[str, Any]], reports: List[ExperimentReport], model: str = "fugu"
) -> PilotRanking:
    configure_fugu_as_default(model)
    agent = Agent(
        name="PilotRanker",
        instructions=(
            "You are deciding which of several piloted research ideas is most "
            "worth developing further, based on the empirical pilot evidence -- "
            "not on how appealing the idea sounds. Prefer ideas with a clear, "
            "well-supported signal (positive or a clean negative result) over "
            "ones that are inconclusive, buggy, or barely implemented."
        ),
        model=model,
        output_type=PilotRanking,
    )
    prompt = "\n\n".join(
        f"### Idea {i}: {idea.get('Title')}\n"
        f"Hypothesis: {idea.get('Short Hypothesis')}\n"
        f"Pilot status: {report.status}\n"
        f"Pilot summary: {report.summary}\n"
        f"Key results: {report.key_results}"
        for i, (idea, report) in enumerate(zip(ideas, reports))
    )
    result = run_agent_with_retry(agent, prompt, max_turns=3)
    return result.final_output


def load_latest_round_report(loop_dir: str) -> tuple[Optional[ExperimentReport], Optional[str]]:
    """Reconstruct the most recent round's ExperimentReport (and its workdir) from
    the round_XX_outcome.json files on disk. Used when resuming a loop that already
    hit the safety cap: no new round runs, so there's no in-memory report to write
    up -- without this the writeup handoff would crash on a None report."""
    outcomes = sorted(
        f for f in os.listdir(loop_dir)
        if f.startswith("round_") and f.endswith("_outcome.json")
    )
    if not outcomes:
        return None, None
    with open(osp.join(loop_dir, outcomes[-1])) as f:
        payload = json.load(f)
    try:
        report = ExperimentReport(**payload["report"])
    except Exception:
        return None, None
    idea_name = payload.get("idea", {}).get("Name", "idea")
    round_idx = int(outcomes[-1].split("_")[1])
    workdir = osp.join(loop_dir, f"round_{round_idx:02d}_{idea_name}")
    return report, (workdir if osp.isdir(workdir) else loop_dir)


def run_development_loop(
    idea: Dict[str, Any],
    loop_dir: str,
    args,
    start_round: int = 0,
    have_report_for_round0: bool = False,
    current_workdir: Optional[str] = None,
    latest_report: Optional[ExperimentReport] = None,
) -> None:
    """The idea<->experiment development loop (no fixed iteration count -- an
    Evaluator decides lock/revise/abandon after each attempt), followed by the
    writeup handoff on lock. Shared between a fresh run and --resume-loop-dir
    so the two paths can't drift out of sync."""
    kb_dir = osp.join(loop_dir, "knowledge_bank")
    decision = None
    workdir, report = current_workdir, latest_report
    for round_idx in range(start_round, args.max_safety_rounds):
        with open(osp.join(loop_dir, "current_idea.json"), "w") as f:
            json.dump(idea, f, indent=2)

        if round_idx == start_round and have_report_for_round0:
            report = latest_report
            workdir = current_workdir
        else:
            workdir = osp.join(loop_dir, f"round_{round_idx:02d}_{idea.get('Name', 'idea')}")
            print(f"\n=== Round {round_idx}: developing '{idea.get('Name')}' in {workdir} ===")
            report = run_research_agent(
                idea, workdir, max_turns=args.final_max_turns, model=args.model,
                knowledge_bank_dir=kb_dir,
            )

        verdict = evaluate_experiment(idea, report, model=args.model)
        print(f"Evaluator decision: {verdict.decision}\nReasoning: {verdict.reasoning}")

        with open(osp.join(loop_dir, f"round_{round_idx:02d}_outcome.json"), "w") as f:
            json.dump(
                {"idea": idea, "report": report.model_dump(), "verdict": verdict.model_dump()},
                f,
                indent=2,
            )

        if verdict.decision == "lock":
            research_wiki.add_entry(idea, "locked", verdict.reasoning, args.wiki_path)
            decision = "lock"
            break
        elif verdict.decision == "abandon":
            research_wiki.add_entry(idea, "abandoned", verdict.reasoning, args.wiki_path)
            decision = "abandon"
            break
        else:
            idea = revise_idea_from_experiment(idea, report, verdict, model=args.model)
    else:
        print(f"Hit safety cap of {args.max_safety_rounds} rounds without a 'lock' decision.")
        decision = "cap"
        # Record it in the cross-run wiki -- otherwise the runs that struggled
        # most (never locked) are exactly the ones the wiki has no memory of, and
        # ideation could keep re-proposing them. 'capped' is distinct from a real
        # lock so future ideation treats it as weak/unresolved.
        research_wiki.add_entry(
            idea,
            "capped",
            f"Hit the safety cap of {args.max_safety_rounds} development rounds "
            "without the evaluator ever locking (last verdict was still 'revise'). "
            "Written up, but not a clean lock.",
            args.wiki_path,
        )

    if decision == "abandon":
        print("Idea was abandoned; not proceeding to writeup.")
        return

    if report is None:
        # Resumed a loop that already sat at the safety cap: no round ran this
        # invocation, so there's no in-memory report. Recover the last recorded
        # one from disk rather than crashing in run_writeup on a None report.
        report, workdir = load_latest_round_report(loop_dir)
        if report is None:
            print(
                "No experiment report available (loop is already at the safety cap "
                "with no recoverable round outcome); cannot write up. Nothing to do."
            )
            return

    print(f"\n=== Writing up '{idea.get('Name')}' (development outcome: {decision}) ===")
    idea_dir = run_writeup(idea, report, workdir, loop_dir, args, decision=decision)
    print(f"Done. Final results in {idea_dir}")


def find_pdf_path_for_review(idea_dir: str):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    return osp.join(idea_dir, pdf_files[0])


def run_writeup(
    idea: Dict[str, Any],
    report: ExperimentReport,
    workdir: str,
    loop_dir: str,
    args,
    decision: str = "lock",
) -> str:
    """Bridge our ExperimentReport into the existing citation/writeup/tectonic
    pipeline: build an idea_dir with idea.json/idea.md/experiment_report.json,
    pull over any plots the Research Agent's Codex calls produced, then run the
    same gather_citations -> perform_writeup -> review sequence the old BFTS
    pipeline used at the end."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = osp.join(loop_dir, f"final_{timestamp}_{idea.get('Name', 'idea')}")
    os.makedirs(idea_dir, exist_ok=True)
    os.makedirs(osp.join(idea_dir, "figures"), exist_ok=True)

    with open(osp.join(idea_dir, "idea.json"), "w") as f:
        json.dump(idea, f, indent=2)
    idea_to_markdown(idea, osp.join(idea_dir, "idea.md"))
    with open(osp.join(idea_dir, "experiment_report.json"), "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    # Distinguish a real evaluator "lock" from a "cap" writeup (safety cap hit
    # while the last verdict was still "revise"), so a capped paper is never
    # silently indistinguishable from a locked one in the output.
    with open(osp.join(idea_dir, "development_decision.json"), "w") as f:
        json.dump({"decision": decision, "locked": decision == "lock"}, f, indent=2)

    for rel_path in report.files_of_interest:
        src = osp.join(workdir, rel_path)
        if osp.isfile(src) and src.lower().endswith((".png", ".jpg", ".jpeg")):
            shutil.copy(src, osp.join(idea_dir, "figures", osp.basename(src)))

    citations_text = gather_citations(
        idea_dir,
        num_cite_rounds=args.num_cite_rounds,
        small_model=args.model,
        knowledge_bank_dir=osp.join(loop_dir, "knowledge_bank"),
    )
    writeup_success = False
    for attempt in range(args.writeup_retries):
        print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
        writeup_success = perform_icbinb_writeup(
            base_folder=idea_dir,
            small_model=args.model,
            big_model=args.model,
            page_limit=4,
            citations_text=citations_text,
        )
        if writeup_success:
            break
    if not writeup_success:
        print("Writeup process did not complete successfully after all retries.")

    pdf_path = find_pdf_path_for_review(idea_dir)
    if pdf_path:
        print("Paper found at:", pdf_path)
        client, client_model = create_client(args.model)
        paper_content = load_paper(pdf_path)
        review_text = perform_review(paper_content, client_model, client)
        review_img_cap_ref = perform_imgs_cap_ref_review(client, client_model, pdf_path)
        with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
            f.write(json.dumps(review_text, indent=4))
        with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
            json.dump(review_img_cap_ref, f, indent=4)
        print("Paper review completed.")
    else:
        print("No PDF found for review, skipping review step.")

    return idea_dir


def main():
    parser = argparse.ArgumentParser(
        description="Idea <-> experiment research loop: no fixed workshop topic or "
        "BFTS-style stages required -- an agent decides what to do at each step."
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default=None,
        help="Optional workshop description file. If omitted, --seed-papers alone "
        "defines the space to propose in.",
    )
    parser.add_argument("--model", type=str, default="fugu")
    parser.add_argument("--seed-papers", type=str, nargs="+", default=None)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--candidate-debate-rounds", type=int, default=3)
    parser.add_argument("--pilot-max-turns", type=int, default=12)
    parser.add_argument("--final-max-turns", type=int, default=60)
    parser.add_argument(
        "--max-safety-rounds",
        type=int,
        default=8,
        help="Hard cap on idea<->experiment development rounds, in case the "
        "evaluator never locks.",
    )
    parser.add_argument(
        "--max-novelty-retries",
        type=int,
        default=3,
        help="Max attempts (deep novelty check -> fresh idea if it fails) before "
        "giving up without developing anything.",
    )
    parser.add_argument("--wiki-path", type=str, default=research_wiki.DEFAULT_WIKI_PATH)
    parser.add_argument(
        "--start-idea-file",
        type=str,
        default=None,
        help="Path to an existing ideas JSON to start from, skipping candidate "
        "generation and pilots entirely (used with --start-idea-idx).",
    )
    parser.add_argument("--start-idea-idx", type=int, default=0)
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument("--writeup-retries", type=int, default=3)
    parser.add_argument(
        "--resume-loop-dir",
        type=str,
        default=None,
        help="Resume an interrupted run (e.g. after a crash) from an existing "
        "experiments/idea_loops/loop_* directory: reuses that directory, loads "
        "current_idea.json, and continues the development loop from the next "
        "round -- skips candidate generation, pilots, and novelty check entirely.",
    )
    args = parser.parse_args()

    configure_fugu_as_default(args.model)

    if args.resume_loop_dir:
        loop_dir = args.resume_loop_dir
        with open(osp.join(loop_dir, "current_idea.json")) as f:
            idea = json.load(f)
        start_round = len(
            [f for f in os.listdir(loop_dir) if f.startswith("round_") and f.endswith("_outcome.json")]
        )
        print(f"Resuming {loop_dir} from round {start_round} with idea '{idea.get('Name')}'")
        run_development_loop(idea, loop_dir, args, start_round=start_round)
        return

    if not args.workshop_file and not args.seed_papers and not args.start_idea_file:
        raise ValueError("Provide --workshop-file, --seed-papers, and/or --start-idea-file.")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    loop_dir = osp.join("experiments", "idea_loops", f"loop_{timestamp}")
    os.makedirs(loop_dir, exist_ok=True)
    print(f"Research loop working directory: {loop_dir}")

    # One paper cache shared across every stage of this loop (candidate debates,
    # pilots, novelty check, development rounds, writeup), so a paper is
    # downloaded + summarized once instead of re-fetched per stage.
    kb_dir = osp.join(loop_dir, "knowledge_bank")

    workshop_description = ""
    if args.workshop_file:
        with open(args.workshop_file) as f:
            workshop_description = f.read()

    wiki_section = research_wiki.format_wiki_section(args.wiki_path)
    combined_context = (workshop_description + "\n\n" + wiki_section).strip()

    if args.start_idea_file:
        print(f"Starting from existing idea {args.start_idea_idx} in {args.start_idea_file}")
        with open(args.start_idea_file) as f:
            start_ideas = json.load(f)
        candidates = [start_ideas[args.start_idea_idx]]
    else:
        print(f"Generating {args.num_candidates} candidate ideas...")
        candidates_fname = osp.join(loop_dir, "candidates.json")
        candidates = []
        for i in range(args.num_candidates):
            ideas = generate_temp_free_idea(
                idea_fname=candidates_fname,
                workshop_description=combined_context,
                max_num_generations=1,
                max_debate_rounds=args.candidate_debate_rounds,
                reload_ideas=(i > 0),  # accumulate so prev_ideas_string keeps later candidates distinct
                seed_papers=args.seed_papers,
                model=args.model,
                knowledge_bank_dir=kb_dir,
            )
            candidates.append(ideas[-1])

    if len(candidates) > 1:
        print("\nRunning cheap pilots on all candidates...")
        reports = []
        for i, idea in enumerate(candidates):
            pilot_dir = osp.join(loop_dir, f"pilot_{i:02d}_{idea.get('Name', 'idea')}")
            print(f"\n--- Pilot {i}: {idea.get('Name')} ---")
            report = run_research_agent(
                idea, pilot_dir, max_turns=args.pilot_max_turns, model=args.model,
                knowledge_bank_dir=kb_dir,
            )
            reports.append(report)
            print(f"Pilot {i} ({idea.get('Name')}): {report.status} -- {report.summary[:300]}")

        ranking = rank_pilots(candidates, reports, model=args.model)
        print(f"\nPilot winner: idea {ranking.winner_index} ({candidates[ranking.winner_index].get('Name')}).")
        print(f"Reasoning: {ranking.reasoning}")

        for i, idea in enumerate(candidates):
            if i != ranking.winner_index:
                research_wiki.add_entry(
                    idea, "eliminated", f"Lost pilot comparison: {ranking.reasoning}", args.wiki_path
                )
        idea = candidates[ranking.winner_index]
        current_workdir = osp.join(loop_dir, f"pilot_{ranking.winner_index:02d}_{idea.get('Name', 'idea')}")
        latest_report = reports[ranking.winner_index]
        have_report_for_round0 = True
    else:
        idea = candidates[0]
        current_workdir = None
        latest_report = None
        have_report_for_round0 = False

    print(f"\n=== Deep novelty verification on '{idea.get('Name')}' ===")
    novelty_workdir = osp.join(loop_dir, "novelty_check")
    for novelty_attempt in range(args.max_novelty_retries):
        novelty = verify_novelty(idea, novelty_workdir, model=args.model, knowledge_bank_dir=kb_dir)
        print(f"Novel: {novelty.is_novel}. Closest prior work: {novelty.closest_prior_work}")
        print(f"Differentiation: {novelty.differentiation}\nReasoning: {novelty.reasoning}")
        if novelty.is_novel:
            break
        research_wiki.add_entry(
            idea,
            "eliminated",
            f"Failed novelty check (closest prior work: {novelty.closest_prior_work}): {novelty.reasoning}",
            args.wiki_path,
        )
        if novelty_attempt == args.max_novelty_retries - 1:
            print(
                "Idea failed novelty verification and no retries remain; stopping "
                "without proceeding to development."
            )
            return
        print("Generating a fresh idea to replace the one that failed novelty check...")
        wiki_section = research_wiki.format_wiki_section(args.wiki_path)
        fresh_ideas = generate_temp_free_idea(
            idea_fname=osp.join(loop_dir, f"novelty_retry_{novelty_attempt}_ideas.json"),
            workshop_description=(workshop_description + "\n\n" + wiki_section).strip(),
            max_num_generations=1,
            max_debate_rounds=args.candidate_debate_rounds,
            reload_ideas=False,
            seed_papers=args.seed_papers,
            model=args.model,
            knowledge_bank_dir=kb_dir,
        )
        idea = fresh_ideas[-1]
        have_report_for_round0 = False  # the fresh idea has no pilot report yet

    run_development_loop(
        idea,
        loop_dir,
        args,
        start_round=0,
        have_report_for_round0=have_report_for_round0,
        current_workdir=current_workdir,
        latest_report=latest_report,
    )


if __name__ == "__main__":
    main()
