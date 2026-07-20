import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

import backoff
import requests

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.semantic_scholar import on_backoff

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _make_cite_key(arxiv_id: str) -> str:
    return "arxiv" + re.sub(r"[^A-Za-z0-9]", "", arxiv_id)


def _entry_to_paper_dict(entry: ET.Element) -> Dict:
    title = entry.findtext("atom:title", default="Unknown Title", namespaces=ATOM_NS)
    title = " ".join(title.split())
    abstract = entry.findtext("atom:summary", default="No abstract available.", namespaces=ATOM_NS)
    abstract = " ".join(abstract.split())
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
    year = published[:4] if published else "Unknown Year"

    author_names = [
        a.findtext("atom:name", default="Unknown", namespaces=ATOM_NS)
        for a in entry.findall("atom:author", ATOM_NS)
    ]
    authors_str = ", ".join(author_names) if author_names else "Unknown"

    entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    # entry_id looks like http://arxiv.org/abs/2401.12345v1
    arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else "unknown"
    arxiv_id_no_version = re.sub(r"v\d+$", "", arxiv_id)

    cite_key = _make_cite_key(arxiv_id_no_version)
    bibtex = (
        "@misc{" + cite_key + ",\n"
        f"  title = {{{title}}},\n"
        f"  author = {{{' and '.join(author_names) if author_names else 'Unknown'}}},\n"
        f"  year = {{{year}}},\n"
        f"  eprint = {{{arxiv_id_no_version}}},\n"
        "  archivePrefix = {arXiv}\n"
        "}"
    )

    return {
        "title": title,
        "authors": authors_str,
        "venue": "arXiv preprint",
        "year": year,
        "abstract": abstract,
        "citationCount": "N/A",
        "citationStyles": {"bibtex": bibtex},
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id_no_version}",
    }


class ArxivSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchArxiv",
        description: str = (
            "Search for relevant preprints using the arXiv API. "
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
        return search_for_papers(query, result_limit=self.max_results)

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {paper.get("authors", "Unknown")}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    on_backoff=on_backoff,
    max_time=120,
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": result_limit,
    }
    rsp = requests.get(ARXIV_BASE_URL, params=params)
    print(f"Response Status Code: {rsp.status_code}")
    print(f"Response Content: {rsp.text[:500]}")
    rsp.raise_for_status()
    root = ET.fromstring(rsp.text)
    entries = root.findall("atom:entry", ATOM_NS)
    time.sleep(0.5)
    if not entries:
        return None

    return [_entry_to_paper_dict(e) for e in entries]
