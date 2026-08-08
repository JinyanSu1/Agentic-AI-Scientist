# Example: what jailbreak-wrapped benign supervision teaches

Two papers from a **single** run of the loop, four days apart, on the same idea. We include both because the pair shows something one paper cannot: what the loop does on its own, and what changes when a human relays the loop's own reviewer back into the idea.

**The question.** Hard-example safety tuning makes a model safer by making it more reluctant, so practitioners mix in jailbreak-shaped *benign* examples to undo the resulting overrefusal. Does that benign data teach transferable harmless behaviour, or only how to handle jailbreak-shaped wrappers? The loop built a same-core, same-target trial: every harmless request appears as a plain version (D) and as the identical request inside a jailbreak wrapper (W), sharing target tokens, hard harmful examples, optimizer slots, initialisation and update count, so the only thing that differs is the benign user context.

## 1. [`1-autonomous-capped/`](1-autonomous-capped) — no human input

Eight development rounds, no intervention in the science. The evaluator never issued a `lock`, the run hit its safety cap, and it wrote the paper up anyway and reviewed itself: **`Decision: Reject`, `Overall: 3`**.

The conclusion it reached was **"Surface Form, Not Intent"** — that the benign data buys wrapper-specific adaptation rather than transferable intent.

Why it never locked is itself informative: the idea's own preregistration had designated a **human annotation audit** "non-cuttable", and no LLM can satisfy that. The agent behaved correctly — it built and integrity-checked a 1,024-response blinded audit package and then wrote, plainly, that no human labels were collected and that it would not fabricate them. So the evaluator withheld `lock` for five consecutive rounds over a requirement that could never be met.

## 2. [`2-human-assisted-locked/`](2-human-assisted-locked) — human in the loop

**This one was not produced autonomously.** Between the two papers a human edited the idea and the run parameters. Exactly what changed:

- **Broke the deadlock.** Rewrote the human-audit requirement so that two independently qualified automatic judges *from different model families* are the accepted terminal standard, with every conclusion explicitly labelled judge-dependent. The very next round locked.
- **Relayed the pipeline's own review back into the idea.** The reviewer of paper 1 had identified one critique as most damaging: a treated-vs-treated `W − D` contrast cannot establish the *absence* of an effect that both arms share, because the D arm also receives benign supervision. That critique was written into the idea as a mandatory experiment — add a supervision-free, design-matched **harmful-only control arm**.
- Raised the round cap from 8 to 14 and resumed; two smaller reviewer points (report the missing statistics; calibrate the claim) were added the same way.

**What that control did is the interesting part.** With a supervision-free anchor, both benign arms turn out to transfer substantially on direct requests (`D − H_norm` +2.40, `W − H_norm` +2.21 log-odds). That **refutes paper 1's own headline**. The surviving claim is narrower and survives cross-family re-judging by a Llama-3.1-70B judge: benign supervision *does* transfer, and it is only W's *incremental* advantage over D that is presentation-dependent. The title changed accordingly, from "Surface Form, **Not Intent**" to "Benign Supervision **Transfers**, but Its Jailbreak-Wrapper Advantage Is Presentation-Dependent".

## The scores did not improve

| | autonomous (capped) | human-assisted (locked) |
|---|---|---|
| Decision | Reject | Reject |
| Overall | 3 | 3 |
| Originality | 3 | **2** |
| Soundness | 2 | 2 |

We are showing this rather than hiding it. Two things are worth knowing about *why*:

- **The reviewer flagged a real citation as fabricated.** It called `arXiv:2605.03226` "a future/nonexistent arXiv identifier … serious credibility red flag". That paper exists — it is one of the seed papers this run was given, and it resolves on arxiv.org. The review model's knowledge cutoff simply predates it. Expect this whenever a run is seeded with genuinely recent work.
- **Honest calibration reads as timidity.** The reviewer's other main objection was that the final claim is "extremely narrow and heavily hedged". It is narrow — because the study refuted its own stronger version. A reviewer rewarding boldness and a pipeline rewarding self-correction will disagree, and that tension does not have a clean resolution.

The reviewer did endorse the substance of the fix, calling the identification argument correct and the design careful, and noted the paper is "unusually honest about limitations, negative results, evaluator sensitivity, and the boundaries of what it can and cannot claim."

## Files

Each directory holds `paper.pdf` (compiled, ARR two-column), `review.json` (the pipeline's own review of it), and `idea.md` (the idea as it stood when that paper was written). The PDFs and reviews are exactly as produced; only the idea text differs between the two, in the ways listed above.
