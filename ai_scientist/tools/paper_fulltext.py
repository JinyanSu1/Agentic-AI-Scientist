"""Get a paper's full text, either from a remote pdf_url (from semantic_scholar/
openalex/arxiv_search) or a local PDF file path, so citation/seed-paper reading can
be grounded in the actual paper content instead of just title/abstract."""

import os
import tempfile
from typing import Optional

import requests

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, source: str) -> Optional[str]:
    if pymupdf4llm is None:
        print("pymupdf4llm not installed; cannot extract PDF text.")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        return pymupdf4llm.to_markdown(tmp_path)
    except Exception as e:
        print(f"Failed to extract text from PDF at {source}: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def fetch_fulltext(pdf_url_or_path: str, max_chars: int = 12000, timeout: int = 30) -> Optional[str]:
    """Get a paper's text from a local PDF path or a remote pdf_url, truncated to
    max_chars. Returns None if it can't be read/downloaded/parsed (paywalled,
    scanned, bad link, missing file, etc.)."""
    if not pdf_url_or_path:
        return None

    if os.path.isfile(pdf_url_or_path):
        try:
            with open(pdf_url_or_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception as e:
            print(f"Failed to read local PDF at {pdf_url_or_path}: {e}")
            return None
        text = _extract_text_from_pdf_bytes(pdf_bytes, pdf_url_or_path)
    else:
        try:
            rsp = requests.get(
                pdf_url_or_path,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (AI-Scientist-v2 citation research bot)"},
            )
            rsp.raise_for_status()
            content_type = rsp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not pdf_url_or_path.lower().endswith(".pdf"):
                return None
        except Exception as e:
            print(f"Failed to download PDF from {pdf_url_or_path}: {e}")
            return None
        text = _extract_text_from_pdf_bytes(rsp.content, pdf_url_or_path)

    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text
