import os
import re
import requests
import time
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.semantic_scholar import on_backoff

OPENALEX_BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex stores abstracts as an inverted index ({word: [positions]}); rebuild plain text."""
    if not abstract_inverted_index:
        return "No abstract available."
    positions: Dict[int, str] = {}
    for word, idxs in abstract_inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def _make_cite_key(author_name: str, year: Union[int, str]) -> str:
    last_name = author_name.split()[-1] if author_name else "Unknown"
    key = re.sub(r"[^A-Za-z0-9]", "", last_name) + str(year)
    return key or f"Unknown{year}"


def _work_to_paper_dict(work: Dict) -> Dict:
    authorships = work.get("authorships", [])
    author_names = [
        a.get("author", {}).get("display_name", "Unknown") for a in authorships
    ]
    authors_str = ", ".join(author_names) if author_names else "Unknown"
    year = work.get("publication_year", "Unknown Year")
    title = work.get("display_name", "Unknown Title")

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name", "Unknown Venue")

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    cite_key = _make_cite_key(author_names[0] if author_names else "Unknown", year)
    doi = work.get("doi")

    open_access = work.get("open_access") or {}
    pdf_url = open_access.get("oa_url") or primary_location.get("pdf_url")
    bibtex_fields = [
        f"  title = {{{title}}}",
        f"  author = {{{' and '.join(author_names) if author_names else 'Unknown'}}}",
        f"  year = {{{year}}}",
    ]
    if venue and venue != "Unknown Venue":
        bibtex_fields.append(f"  journal = {{{venue}}}")
    if doi:
        bibtex_fields.append(f"  doi = {{{doi.replace('https://doi.org/', '')}}}")
    bibtex = "@article{" + cite_key + ",\n" + ",\n".join(bibtex_fields) + "\n}"

    return {
        "title": title,
        "authors": authors_str,
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": work.get("cited_by_count", 0),
        "citationStyles": {"bibtex": bibtex},
        "pdf_url": pdf_url,
    }


class OpenAlexSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex. "
            "Free, no API key required. Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.mailto = os.getenv("OPENALEX_MAILTO")

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
        max_time=120,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        return search_for_papers(query, result_limit=self.max_results, mailto=self.mailto)

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {paper.get("authors", "Unknown")}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    on_backoff=on_backoff,
    max_time=120,
)
def search_for_papers(
    query, result_limit=10, mailto: Optional[str] = None
) -> Union[None, List[Dict]]:
    if not query:
        return None

    params = {
        "search": query,
        "per-page": result_limit,
    }
    mailto = mailto or os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto

    rsp = requests.get(OPENALEX_BASE_URL, params=params)
    print(f"Response Status Code: {rsp.status_code}")
    print(f"Response Content: {rsp.text[:500]}")
    rsp.raise_for_status()
    results = rsp.json()
    works = results.get("results", [])
    time.sleep(0.2)
    if not works:
        return None

    return [_work_to_paper_dict(w) for w in works]
