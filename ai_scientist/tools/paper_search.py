"""Citation search that queries multiple free literature APIs concurrently and
merges/deduplicates the results, so a rate-limited or blind-spot source doesn't
starve the candidate list the LLM picks citations from."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from ai_scientist.tools import arxiv_search, openalex, semantic_scholar

SOURCES = (
    ("Semantic Scholar", semantic_scholar.search_for_papers),
    ("OpenAlex", openalex.search_for_papers),
    ("arXiv", arxiv_search.search_for_papers),
)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _search_one(source_name: str, search_fn, query: str, result_limit: int) -> List[Dict]:
    try:
        papers = search_fn(query, result_limit=result_limit)
    except Exception as e:
        print(f"{source_name} search failed for query {query!r}: {e}")
        return []
    return papers or []


def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None

    with ThreadPoolExecutor(max_workers=len(SOURCES)) as executor:
        futures = [
            executor.submit(_search_one, source_name, search_fn, query, result_limit)
            for source_name, search_fn in SOURCES
        ]
        results_by_source = [f.result() for f in futures]

    seen_titles = set()
    merged: List[Dict] = []
    # SOURCES order sets priority when the same paper turns up more than once
    for papers in results_by_source:
        for paper in papers:
            key = _normalize_title(paper.get("title", ""))
            if key and key in seen_titles:
                continue
            if key:
                seen_titles.add(key)
            merged.append(paper)

    return merged or None
