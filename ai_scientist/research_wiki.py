"""A persistent, cross-run memory of every idea tried: what it was, what
happened, and why it was locked/abandoned/eliminated. Unlike iteration_memory.md
(scoped to one research-loop run), this survives across separate invocations of
run_research_loop.py, so ideation never proposes something already known to have
failed, and repeated failures in the same direction get flagged.
"""

import json
import os
import os.path as osp
import time
from typing import Any, Dict, List

DEFAULT_WIKI_PATH = "research_wiki.jsonl"


def add_entry(idea: Dict[str, Any], outcome: str, reason: str, wiki_path: str = DEFAULT_WIKI_PATH) -> None:
    """outcome: 'locked' | 'abandoned' | 'eliminated' (didn't win a pilot comparison)
    | 'capped' (written up but the evaluator never cleanly locked it -- it hit the
    development safety cap after repeated revisions)."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": idea.get("Name", "unknown"),
        "title": idea.get("Title", ""),
        "hypothesis": idea.get("Short Hypothesis", ""),
        "outcome": outcome,
        "reason": reason,
    }
    # If a previous append was torn mid-line (crash/interleaving), start on a
    # fresh line so only the torn line is lost, not this entry glued onto it.
    prefix = ""
    if osp.exists(wiki_path) and osp.getsize(wiki_path) > 0:
        with open(wiki_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(wiki_path, "a") as f:
        f.write(prefix + json.dumps(entry) + "\n")


def load_entries(wiki_path: str = DEFAULT_WIKI_PATH, max_entries: int = 30) -> List[Dict[str, Any]]:
    if not osp.exists(wiki_path):
        return []
    entries = []
    with open(wiki_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn/corrupted line (a crash mid-append, or two concurrent
                # runs' appends interleaving) must not brick ideation for every
                # future run; skip it and keep the rest of the wiki usable.
                continue
    return entries[-max_entries:]


def format_wiki_section(wiki_path: str = DEFAULT_WIKI_PATH, max_entries: int = 30) -> str:
    """A digest for ideation prompts: what's already been tried and why it didn't
    pan out, so a new proposal doesn't repeat it. Empty string if the wiki is empty."""
    entries = load_entries(wiki_path, max_entries)
    if not entries:
        return ""

    failed = [e for e in entries if e["outcome"] in ("abandoned", "eliminated")]
    locked = [e for e in entries if e["outcome"] == "locked"]
    capped = [e for e in entries if e["outcome"] == "capped"]

    lines = [
        "RESEARCH WIKI -- ideas already tried across all previous runs. Do not "
        "propose something too close to a failed one below without a genuinely "
        "different angle; you may build on/extend a locked one if relevant."
    ]
    if locked:
        lines.append("\nPreviously locked (successfully pursued) ideas:")
        for e in locked:
            lines.append(f"- {e['title']} ({e['name']}): {e['hypothesis']}")
    if failed:
        lines.append("\nPreviously failed/eliminated ideas (avoid repeating):")
        for e in failed:
            lines.append(f"- {e['title']} ({e['name']}): {e['hypothesis']}\n  Why it failed: {e['reason']}")
    if capped:
        lines.append(
            "\nIdeas written up but never cleanly locked (hit the development "
            "safety cap after repeated revisions -- treat as weak/unresolved; "
            "only revisit with a materially stronger angle):"
        )
        for e in capped:
            lines.append(f"- {e['title']} ({e['name']}): {e['hypothesis']}\n  Note: {e['reason']}")

    # Same-direction repeated-failure flag, matching ARIS's "3+ failed ideas
    # trigger re-ideation suggestion" -- surface it, don't decide for the caller.
    # Capped ideas count as struggling directions too.
    if len(failed) + len(capped) >= 3:
        lines.append(
            f"\nNote: {len(failed)} ideas have failed/been eliminated recently. If "
            "they cluster around the same underlying approach, consider a "
            "structurally different direction rather than another small variation."
        )

    return "\n".join(lines)
