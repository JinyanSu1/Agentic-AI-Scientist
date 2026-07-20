"""A persistent, per-idea cache of papers we've already looked up and read, so the
same paper never gets re-downloaded or re-summarized twice across seed-reading and
citation-search rounds. Each entry is stored as its own JSON file, keyed by a slug of
the paper's title, under <base_folder>/knowledge_bank/.
"""

import json
import os
import os.path as osp
import re
import traceback
from typing import Dict, Optional

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.tools.paper_fulltext import fetch_fulltext

KNOWLEDGE_BANK_DIRNAME = "knowledge_bank"


def _slugify_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:100] or "untitled"


def _bank_dir(base_folder: str) -> str:
    path = osp.join(base_folder, KNOWLEDGE_BANK_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def lookup(base_folder: str, title: str) -> Optional[Dict]:
    """Return the cached entry for this paper title, if we've already looked it up."""
    entry_path = osp.join(_bank_dir(base_folder), _slugify_title(title) + ".json")
    if not osp.exists(entry_path):
        return None
    try:
        with open(entry_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save(base_folder: str, title: str, entry: Dict) -> None:
    entry_path = osp.join(_bank_dir(base_folder), _slugify_title(title) + ".json")
    with open(entry_path, "w") as f:
        json.dump(entry, f, indent=2)


SUMMARIZE_SYSTEM_MESSAGE = (
    "You are a research assistant reading a paper on behalf of someone planning their "
    "own research project. Your summary will be used to inform their idea, experiment "
    "plan, and related-work writing -- not just the last of those."
)
SUMMARIZE_PROMPT_TEMPLATE = """Here is the research project this paper is being read for:
```markdown
{idea_context}
```

Paper title: {title}

Content (full text if available, otherwise just the abstract):
```
{content}
```

Summarize this paper for someone planning the research project above. Cover, concretely
wherever the content supports it (skip anything not covered rather than guessing):
1. What the paper does and its key method/result.
2. Concrete methodological details relevant to planning our own experiments: which
   datasets/benchmarks it uses, what models/baselines it compares against, what metrics
   it reports, and (if stated) compute budget or runtime.
3. How it specifically relates to our project above -- e.g. a method/dataset we could
   reuse or must differentiate from, a baseline we should compare to, a result that
   supports or undercuts our hypothesis.
4. What it would be cited for (e.g. baseline, dataset/benchmark, related method,
   motivating negative result).

Be concrete and specific, not generic."""


def get_paper_knowledge(base_folder: str, paper: Dict, model: str, idea_context: str = "") -> str:
    """Look up (or fetch + summarize + cache) what we know about this paper.

    Checks the knowledge bank by title first, so a paper turned up again by a later
    search (or by a different seed/citation round) is never re-downloaded or
    re-summarized. The summarization call is a single, dedicated LLM call with no
    other context, not tied into whatever conversation/round is asking for it -- except
    for idea_context, which tells it what to read the paper *for*, so the summary pulls
    out details relevant to planning our own experiments, not just a generic abstract.
    """
    title = paper.get("title") or "Unknown Title"
    cached = lookup(base_folder, title)
    if cached is not None:
        return cached.get("summary", "")

    fulltext = fetch_fulltext(paper.get("pdf_url"))
    content = fulltext if fulltext else paper.get("abstract", "No abstract available.")
    read_full = fulltext is not None

    summary = content
    try:
        client, client_model = create_client(model)
        text, _ = get_response_from_llm(
            prompt=SUMMARIZE_PROMPT_TEMPLATE.format(
                title=title,
                content=content,
                idea_context=idea_context or "(no specific project context provided)",
            ),
            client=client,
            model=client_model,
            system_message=SUMMARIZE_SYSTEM_MESSAGE,
            print_debug=False,
        )
        summary = text.strip()
    except Exception:
        print(f"EXCEPTION summarizing paper {title!r}:")
        print(traceback.format_exc())
        # fall back to raw abstract/full text rather than losing the paper entirely

    save(
        base_folder,
        title,
        {
            "title": title,
            "read_full_text": read_full,
            "pdf_url": paper.get("pdf_url"),
            "summary": summary,
        },
    )
    print(f"Knowledge bank: added {'(full text)' if read_full else '(abstract only)'}: {title}")
    return summary
