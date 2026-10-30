"""Web API tool used by the LLM (tool calling) when the PDF does not cover a question.

Uses the public Wikipedia API, which needs no API key.
"""
import requests
from langchain_core.tools import tool

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "rag-pdf-qa/1.0 (PDF RAG assignment)"}


def wikipedia_search(query: str, limit: int = 2) -> list[dict]:
    """Return [{title, url, extract}] for the top Wikipedia matches. Empty list on any failure."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 900,
        "inprop": "url",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"[warn] Wikipedia lookup failed: {e}")
        return []
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))
    return [
        {"title": p["title"], "url": p.get("fullurl", ""), "extract": p.get("extract", "").strip()}
        for p in ordered
        if p.get("extract")
    ]


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for general facts that are NOT covered by the PDF document.
    Input: a short, focused search query."""
    results = wikipedia_search(query)
    return "\n\n".join(f"{r['title']}: {r['extract']}" for r in results) or "No results."
