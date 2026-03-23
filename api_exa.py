"""api_exa.py — All Exa API interactions: search and content retrieval only."""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()


class ExaResult(TypedDict):
    url: str
    title: str
    text_excerpt: str
    published_date: str


def get_exa_client():
    """Initialize and return Exa client using EXA_API_KEY from environment."""
    import exa_py  # local import so missing package gives a clear error

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise EnvironmentError("EXA_API_KEY not set in environment.")
    return exa_py.Exa(api_key=api_key)


def search_book_passages(
    book_title: str,
    author: str,
    topic: str,
    num_results: int = 5,
) -> list[ExaResult]:
    """Search Exa for passages, quotes, and page references from a specific book."""
    try:
        client = get_exa_client()
        query = f'"{book_title}" {author} quote passage excerpt "{topic}"'
        response = client.search_and_contents(
            query,
            num_results=num_results,
            text=True,
            highlights=True,
        )
        results: list[ExaResult] = []
        for r in response.results:
            text_excerpt = ""
            if hasattr(r, "highlights") and r.highlights:
                text_excerpt = " ... ".join(r.highlights[:3])
            elif hasattr(r, "text") and r.text:
                text_excerpt = r.text[:600]
            results.append(
                ExaResult(
                    url=r.url or "",
                    title=r.title or "",
                    text_excerpt=text_excerpt,
                    published_date=getattr(r, "published_date", "") or "",
                )
            )
        return results
    except Exception as exc:
        print(f"[api_exa] search_book_passages error: {exc}", file=sys.stderr)
        return []


def search_work_context(
    title: str,
    author: str,
    context_type: str = "themes",
) -> list[ExaResult]:
    """Search for critical analysis, chapter summaries, themes, and scholarly commentary."""
    try:
        client = get_exa_client()
        query = f'{author} "{title}" {context_type} literary analysis'
        response = client.search_and_contents(
            query,
            num_results=5,
            text=True,
            highlights=True,
        )
        results: list[ExaResult] = []
        for r in response.results:
            text_excerpt = ""
            if hasattr(r, "highlights") and r.highlights:
                text_excerpt = " ... ".join(r.highlights[:3])
            elif hasattr(r, "text") and r.text:
                text_excerpt = r.text[:600]
            results.append(
                ExaResult(
                    url=r.url or "",
                    title=r.title or "",
                    text_excerpt=text_excerpt,
                    published_date=getattr(r, "published_date", "") or "",
                )
            )
        return results
    except Exception as exc:
        print(f"[api_exa] search_work_context error: {exc}", file=sys.stderr)
        return []


def extract_usable_quotes(
    results: list[ExaResult],
    max_quotes: int = 8,
) -> list[str]:
    """Extract clean, formatted quote strings from raw Exa results."""
    quotes: list[str] = []
    for r in results:
        text = r.get("text_excerpt", "").strip()
        if not text:
            continue
        # Truncate quote text to 300 characters
        if len(text) > 300:
            text = text[:297] + "..."
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        # Extract domain from URL
        domain = ""
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
            except Exception:
                domain = url
        formatted = f'"{text}" — {title} (source: {domain})'
        quotes.append(formatted)
        if len(quotes) >= max_quotes:
            break
    return quotes
