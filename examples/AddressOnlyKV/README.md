# Example: AddressOnlyKV

One complete, unedited run of the research loop, included so you can see what it actually produces before you set anything up.

**In plain terms:** LLM inference caches store a "key" (where to look) and a "value" (what to retrieve) for every past token. The idea tested here was whether some of those cached entries are only useful for their key — i.e. whether you could zero out the value and save memory without hurting the model. [`idea.md`](idea.md) has the full hypothesis in the system's own (dense, jargon-heavy) words; the short version is: the loop designed a rigorous audit for this, ran it across 3 models and multiple context lengths, and the answer came back **no, not in a way that's worth exploiting** — a clean, preregistered negative result, with a couple of small reproducible exceptions honestly reported rather than glossed over.

- [`paper.pdf`](paper.pdf) — the compiled paper (`--model fugu`, ARR two-column format).
- [`review.json`](review.json) — the pipeline's own LLM review of that paper. We're including it as-is: it scored the paper `Overall: 4` / `Decision: Reject`, and its stated weaknesses (dense presentation, a small model/document count) are fair. We'd rather show the system's honest self-assessment than a cherry-picked "win" — the whole point of this pipeline is that it isn't tuned to only report positive results.
- [`idea.md`](idea.md) — the idea as the loop's own ideation debate wrote it up (title, hypothesis, related work, experiment plan), before development started.

This is one idea from one full run, kept exactly as the loop produced it — nothing here was hand-edited afterward.
