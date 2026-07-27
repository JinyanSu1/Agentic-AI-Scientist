<div align="center">
  <h1><b>Agentic AI Scientist</b></h1>
  <p><i>An open-ended, agent-driven automated research loop.</i></p>
</div>

Agentic AI Scientist is a heavily rearchitected fork of [SakanaAI's AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2). It replaces the original best-first tree-search (BFTS) pipeline with a single open-ended **research loop**: an agent proposes ideas through a structured debate, pilots them, verifies novelty, then develops the winner while an evaluator decides after each attempt whether to lock, revise, or abandon it — with no fixed stages or iteration counts baked in. The actual coding and running of experiments is delegated to a pluggable **coding worker** — either [Codex CLI](https://github.com/openai/codex) or the [Claude Code CLI](https://github.com/anthropics/claude-code), selected with `--worker` (see [Coding worker](#coding-worker)). The orchestration layer (ideation, evaluation, the Research Agent's reasoning) is built on the [openai-agents](https://github.com/openai/openai-agents-python) SDK and, as shipped, is wired to the internal Sakana **fugu** gateway (`SAKANA_API_KEY`) — swapping that layer to a different provider requires a small code change in [`ai_scientist/agents_common.py`](ai_scientist/agents_common.py), not just a flag.

> **⚠️ Caution — this executes LLM-written code.**
> The research agent delegates to its coding worker running fully unattended — Codex with `--dangerously-bypass-approvals-and-sandbox`, or Claude Code with `--dangerously-skip-permissions` — which will write and run arbitrary code and shell commands in its working directory. Run it only inside an isolated environment (e.g. a dedicated SLURM allocation or container) that you are willing to treat as the security boundary. Use at your own discretion.

## How it works

The whole loop is orchestrated by [`run_research_loop.py`](run_research_loop.py):

1. **Ideation debate** ([`ai_scientist/perform_ideation_temp_free.py`](ai_scientist/perform_ideation_temp_free.py)) — a Proposer / Challenger / Evaluator debate (openai-agents handoffs, structured outputs) produces candidate ideas, grounded in seed papers read in full and informed by the cross-run research wiki so previously-failed directions aren't repeated.
2. **Pilots** — if there is more than one candidate, each gets a cheap, tightly turn-capped pilot run, and the winner is picked by *empirical signal*, not by how appealing the idea sounds.
3. **Deep novelty verification** ([`verify_novelty`](ai_scientist/perform_idea_iteration.py)) — multiple targeted literature searches, explicit closest-prior-work identification, and a concurrent-work check on the pilot winner before real effort is committed.
4. **Development loop** — the Research Agent ([`ai_scientist/perform_research_agent.py`](ai_scientist/perform_research_agent.py)) implements/runs/inspects experiments via its coding worker ([Codex or Claude Code](#coding-worker)) and decides what to do next; after each attempt an Evaluator decides **lock / revise / abandon**, up to a safety cap. Every outcome is logged to the persistent research wiki ([`ai_scientist/research_wiki.py`](ai_scientist/research_wiki.py)).
5. **Writeup & review** ([`ai_scientist/perform_icbinb_writeup.py`](ai_scientist/perform_icbinb_writeup.py)) — on lock, the report is handed to multi-source citation search (Semantic Scholar + OpenAlex + arXiv, merged and deduplicated), LaTeX writeup (compiled with `tectonic`), and an LLM + VLM review.

Papers read anywhere in a run are cached once per loop in a shared knowledge bank, so the same paper is never re-downloaded or re-summarized across stages.

## Requirements

Designed to run on Linux, with NVIDIA GPUs available to the experiment worker.

### Installation

```bash
# Create a new conda environment
conda create -n ai-scientist-v2 python=3.11
conda activate ai-scientist-v2

# Python package requirements
pip install -r requirements.txt

# The agent framework used throughout the loop
pip install openai-agents
```

You also need:

- A **coding worker** on `PATH` — either [Codex CLI](https://github.com/openai/codex) (default, `--worker codex`) or the [Claude Code CLI](https://github.com/anthropics/claude-code) (`--worker claude-code`). See [Coding worker](#coding-worker) below for setup of each.
- **[tectonic](https://tectonic-typesetting.github.io/)** for LaTeX/PDF compilation of the final paper.
- **poppler** (`pdftotext`, `pdftoppm`) and **chktex**, used by the writeup's review/reflection step to read the compiled PDF back and lint the LaTeX — `conda install -c conda-forge poppler chktex`. Without them the paper still compiles, but the review step is degraded.
- `pymupdf4llm` (in `requirements.txt`) for reading paper full text.

All of the `PATH` tools above (coding worker CLI, tectonic, poppler, chktex) must be resolvable by the process that runs `run_research_loop.py` — e.g. on `~/.local/bin` or the active conda env's `bin`.

### Coding worker

`--worker` picks which CLI agent actually writes, runs, and debugs experiment code (`ai_scientist/tools/coding_worker.py`). The Research Agent's reasoning ("what should I try next") stays on the `--model` orchestrator regardless of which worker you pick.

- **`--worker codex` (default)** — shells out to `codex exec --profile <name> --dangerously-bypass-approvals-and-sandbox ...`. `--codex-profile` (default `fugu`) selects the Codex CLI profile; pass `--codex-profile ""` to use Codex's own default profile/login (e.g. a regular OpenAI account) instead of the internal Sakana-routed `fugu` profile.
- **`--worker claude-code`** — shells out to `claude -p "<task>" --dangerously-skip-permissions --output-format json`. Requires the `claude` CLI on `PATH`, authenticated independently (e.g. `claude auth` or `ANTHROPIC_API_KEY`) of whatever `--model`/`SAKANA_API_KEY` is doing for the orchestrator.

Both run fully unattended with approvals/sandboxing bypassed — see the caution above.

### Environment variables

```bash
export SAKANA_API_KEY="YOUR_FUGU_GATEWAY_KEY"   # required — all models route through the fugu gateway

# Optional, for literature search:
export S2_API_KEY="YOUR_S2_KEY"                 # higher-throughput Semantic Scholar (falls back to keyless + rate limits)
export OPENALEX_MAILTO="you@example.com"         # polite pool for OpenAlex
```

## Usage

### Run the full research loop

Point the loop at one or more seed papers (local PDF paths and/or search queries) and/or a workshop-topic file, and it will handle ideation → pilots → novelty check → development → writeup end to end:

```bash
python run_research_loop.py \
  --seed-papers "path/to/paper.pdf" "some topic to search for" \
  --model fugu
```

Key flags (defaults in parentheses):

| Flag | Meaning |
| --- | --- |
| `--seed-papers` | Local PDF paths and/or search queries that define the space to propose in. |
| `--workshop-file` | Optional Markdown topic description (alternative or complement to seed papers). |
| `--start-idea-file` / `--start-idea-idx` | Skip ideation/pilots and develop an existing idea from a JSON file. |
| `--model` (`fugu`) | Model used for the orchestrator (ideation/evaluation/Research-Agent reasoning); as shipped this routes through the Sakana fugu gateway regardless of value (see caveat above). |
| `--worker` (`codex`) | Which CLI coding agent executes experiment code: `codex` or `claude-code`. See [Coding worker](#coding-worker). |
| `--codex-profile` (`fugu`) | Codex CLI profile used when `--worker codex`; `""` for Codex's own default profile/login. |
| `--num-candidates` (`3`) | How many candidate ideas to generate and pilot. |
| `--candidate-debate-rounds` (`3`) | Max Proposer/Challenger/Evaluator rounds per candidate. |
| `--pilot-max-turns` (`12`) | Turn cap for each cheap pilot run. |
| `--final-max-turns` (`60`) | Turn cap for each development round. |
| `--max-safety-rounds` (`8`) | Hard cap on develop↔revise rounds if the evaluator never locks. |
| `--max-novelty-retries` (`3`) | Attempts to re-ideate if the novelty check fails. |
| `--num-cite-rounds` (`20`) | Citation-gathering rounds during writeup. |
| `--writeup-retries` (`3`) | Retries for the LaTeX writeup. |
| `--resume-loop-dir` | Resume an interrupted run from an existing `experiments/idea_loops/loop_*` directory. |

### Generate ideas only

The ideation debate can be run standalone to produce a JSON idea file without developing anything:

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --seed-papers "path/to/paper.pdf" "some topic" \
  --model fugu \
  --max-num-generations 5 \
  --max-debate-rounds 12
```

### Outputs

Each run writes to a timestamped directory under `experiments/idea_loops/loop_<timestamp>/`:

- `candidates.json` — generated candidate ideas
- `pilot_*/`, `novelty_check/` — pilot runs and the novelty verification workdir
- `round_XX_<name>/` and `round_XX_outcome.json` — each development round's workdir and evaluator verdict
- `current_idea.json` — the live idea (used for `--resume-loop-dir`)
- `final_<timestamp>_<name>/` — the writeup: `idea.json`, `experiment_report.json`, `development_decision.json` (records whether the paper came from a clean `lock` or a `cap`), figures, and the compiled `.pdf` plus reviews

Cross-run memory of every idea tried (locked / abandoned / eliminated / capped) accumulates in `research_wiki.jsonl` and is fed back into future ideation.

## Acknowledgement

This project is a fork of [SakanaAI's AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2); the writeup/review and literature-search components descend from that codebase, whose experiment engine was in turn built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the original authors for making their work publicly available.

## License & Responsible Use

This project inherits **The AI Scientist Source Code License** (a derivative of the Responsible AI License) from the upstream repository; see [`LICENSE`](LICENSE).

**Mandatory disclosure:** by using this code you are bound to clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers.
