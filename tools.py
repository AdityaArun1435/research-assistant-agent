"""
Tool implementations for the research assistant agent.

Each tool is a plain Python function that takes simple arguments and returns
a JSON-serializable dict. Nothing here knows about Groq, the agent loop, or
Streamlit, this module only talks to arXiv and Wikipedia.

Every tool also returns a "sources" list of {"title": ..., "url": ...} pairs.
The agent loop uses that list to build the set of "actually retrieved" URLs
for citation verification, so keep it accurate: only include a source here if
its URL was genuinely returned by the API call.
"""

import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

REQUEST_TIMEOUT_SECONDS = 15

# arXiv's API silently hangs/throttles requests that don't send a descriptive
# User-Agent (Python's default "python-requests/x.y" gets deprioritized).
# Wikipedia's etiquette guidelines ask for the same. Set on both.
_REQUEST_HEADERS = {"User-Agent": "research-assistant-agent/1.0 (portfolio project; contact via github)"}

# Atom/arXiv XML namespaces
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# arXiv publishes a soft rate limit of roughly one request per 3 seconds and
# returns HTTP 429 above that. A single user asking one question rarely hits
# this, but the agent can legitimately fire a few arxiv calls in a row while
# refining a query, so a short retry-with-backoff makes the tool resilient
# to a transient 429 instead of surfacing it as a hard failure immediately.
_ARXIV_MAX_RETRIES = 2
_ARXIV_RETRY_BACKOFF_SECONDS = 3


def search_arxiv(query: str, max_results: int = 5) -> dict:
    """
    Search arXiv for papers matching `query`.

    Returns a dict with a "results" list, each entry holding title, authors,
    summary, published date, and URL, plus a "sources" list built from those
    same URLs for citation verification.
    """
    max_results = max(1, min(int(max_results), 10))  # keep requests reasonable

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    response = None
    for attempt in range(_ARXIV_MAX_RETRIES + 1):
        is_last_attempt = attempt == _ARXIV_MAX_RETRIES
        try:
            response = requests.get(
                ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS, headers=_REQUEST_HEADERS
            )
            if response.status_code == 429 and not is_last_attempt:
                time.sleep(_ARXIV_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            response.raise_for_status()
            break  # success
        except requests.RequestException as exc:
            if is_last_attempt:
                if response is not None and response.status_code == 429:
                    return {
                        "error": "arXiv is rate-limiting this client (HTTP 429) after retries. Try again shortly.",
                        "results": [],
                        "sources": [],
                    }
                return {"error": f"arXiv request failed: {exc}", "results": [], "sources": []}
            time.sleep(_ARXIV_RETRY_BACKOFF_SECONDS * (attempt + 1))

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        return {"error": f"arXiv response could not be parsed: {exc}", "results": [], "sources": []}

    results = []
    sources = []

    for entry in root.findall(f"{_ATOM_NS}entry"):
        title_el = entry.find(f"{_ATOM_NS}title")
        summary_el = entry.find(f"{_ATOM_NS}summary")
        published_el = entry.find(f"{_ATOM_NS}published")
        id_el = entry.find(f"{_ATOM_NS}id")

        title = (title_el.text or "").strip() if title_el is not None else "Untitled"
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""
        url = (id_el.text or "").strip() if id_el is not None else ""

        authors = [
            (name_el.text or "").strip()
            for author_el in entry.findall(f"{_ATOM_NS}author")
            for name_el in [author_el.find(f"{_ATOM_NS}name")]
            if name_el is not None and name_el.text
        ]

        if not url:
            continue  # a result we can't cite is not useful to this agent

        results.append(
            {
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "url": url,
            }
        )
        sources.append({"title": title, "url": url})

    if not results:
        return {"error": None, "results": [], "sources": [], "note": "No arXiv results found for this query."}

    return {"error": None, "results": results, "sources": sources}


def search_wikipedia(query: str) -> dict:
    """
    Look up `query` on Wikipedia: find the best-matching page, then fetch a
    plain-text summary (via the same MediaWiki API's extracts prop) and its
    canonical URL.

    Returns a dict with "summary", "url", "title", and a matching "sources"
    list for citation verification.
    """
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }

    try:
        search_response = requests.get(
            WIKIPEDIA_API_URL, params=search_params, timeout=REQUEST_TIMEOUT_SECONDS,
            headers=_REQUEST_HEADERS,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"Wikipedia search failed: {exc}", "summary": "", "url": "", "sources": []}

    hits = search_data.get("query", {}).get("search", [])
    if not hits:
        return {
            "error": None,
            "summary": "",
            "url": "",
            "title": "",
            "sources": [],
            "note": f"No Wikipedia page found for '{query}'.",
        }

    page_title = hits[0]["title"]

    extract_params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
        "titles": page_title,
        "format": "json",
    }

    try:
        extract_response = requests.get(
            WIKIPEDIA_API_URL, params=extract_params, timeout=REQUEST_TIMEOUT_SECONDS,
            headers=_REQUEST_HEADERS,
        )
        extract_response.raise_for_status()
        extract_data = extract_response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"Wikipedia page fetch failed: {exc}", "summary": "", "url": "", "sources": []}

    pages = extract_data.get("query", {}).get("pages", {})
    if not pages:
        return {"error": None, "summary": "", "url": "", "title": page_title, "sources": [],
                "note": f"No Wikipedia page found for '{query}'."}

    page = next(iter(pages.values()))
    summary = (page.get("extract") or "").strip()
    url = page.get("fullurl", "")
    title = page.get("title", page_title)

    if not url:
        return {"error": None, "summary": summary, "url": "", "title": title, "sources": [],
                "note": "Wikipedia page found but no canonical URL was returned."}

    return {
        "error": None,
        "summary": summary,
        "url": url,
        "title": title,
        "sources": [{"title": title, "url": url}],
    }


# --- Tool registry -----------------------------------------------------
# This is the single source of truth mapping a tool name to its Python
# callable and its OpenAI-style JSON schema. agent.py imports TOOL_SCHEMAS
# to send to Groq, and TOOL_FUNCTIONS to dispatch a requested call.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv for academic papers on a topic. Use this for research "
                "literature, technical methods, experimental results, or anything "
                "that would show up as a preprint or academic paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'transformer attention mechanism'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of papers to return (1-10). Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Fetch a Wikipedia summary for background or contextual information "
                "that is not the subject of academic papers, definitions, history, "
                "general concepts, biographies, or events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to look up, e.g. 'CRISPR gene editing'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_arxiv": search_arxiv,
    "search_wikipedia": search_wikipedia,
}
