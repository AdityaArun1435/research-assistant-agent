"""
Unit tests focused on the parts of this project that don't require a live
Groq API key: citation verification, and the tool functions' XML/JSON
parsing (using mocked HTTP responses, no real network calls).

Run with: pytest
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import verify_citations
from tools import search_arxiv, search_wikipedia


# --- verify_citations -----------------------------------------------------

def test_verify_citations_all_verified():
    sources = [{"title": "Paper A", "url": "https://arxiv.org/abs/1234.5678"}]
    answer = "According to [Paper A](https://arxiv.org/abs/1234.5678), this works."
    assert verify_citations(answer, sources) == []


def test_verify_citations_flags_unmatched_url():
    sources = [{"title": "Paper A", "url": "https://arxiv.org/abs/1234.5678"}]
    answer = "According to [Paper B](https://arxiv.org/abs/9999.0000), this works."
    assert verify_citations(answer, sources) == ["https://arxiv.org/abs/9999.0000"]


def test_verify_citations_no_citations():
    assert verify_citations("Plain text answer with no links.", []) == []


def test_verify_citations_mixed():
    sources = [{"title": "Real", "url": "https://example.com/real"}]
    answer = (
        "See [Real](https://example.com/real) and also "
        "[Fake](https://example.com/fake)."
    )
    assert verify_citations(answer, sources) == ["https://example.com/fake"]


# --- search_arxiv -----------------------------------------------------

ARXIV_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""


def test_search_arxiv_parses_entries():
    mock_response = Mock()
    mock_response.text = ARXIV_SAMPLE_XML
    mock_response.raise_for_status = Mock()

    with patch("tools.requests.get", return_value=mock_response):
        result = search_arxiv("attention", max_results=1)

    assert result["error"] is None
    assert len(result["results"]) == 1
    paper = result["results"][0]
    assert paper["title"] == "Attention Is All You Need"
    assert paper["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper["url"] == "http://arxiv.org/abs/1706.03762v5"
    assert result["sources"] == [
        {"title": "Attention Is All You Need", "url": "http://arxiv.org/abs/1706.03762v5"}
    ]


def test_search_arxiv_handles_request_failure():
    import requests

    with patch("tools.requests.get", side_effect=requests.RequestException("boom")), \
         patch("tools.time.sleep"):  # skip real backoff delays in tests
        result = search_arxiv("anything")

    assert result["error"] is not None
    assert result["results"] == []
    assert result["sources"] == []


# --- search_wikipedia -----------------------------------------------------

def test_search_wikipedia_happy_path():
    search_response = Mock()
    search_response.raise_for_status = Mock()
    search_response.json.return_value = {"query": {"search": [{"title": "CRISPR"}]}}

    extract_response = Mock()
    extract_response.raise_for_status = Mock()
    extract_response.json.return_value = {
        "query": {
            "pages": {
                "12345": {
                    "title": "CRISPR",
                    "extract": "CRISPR is a family of DNA sequences...",
                    "fullurl": "https://en.wikipedia.org/wiki/CRISPR",
                }
            }
        }
    }

    with patch("tools.requests.get", side_effect=[search_response, extract_response]):
        result = search_wikipedia("CRISPR gene editing")

    assert result["error"] is None
    assert result["title"] == "CRISPR"
    assert result["url"] == "https://en.wikipedia.org/wiki/CRISPR"
    assert result["sources"] == [{"title": "CRISPR", "url": "https://en.wikipedia.org/wiki/CRISPR"}]


def test_search_wikipedia_no_results():
    search_response = Mock()
    search_response.raise_for_status = Mock()
    search_response.json.return_value = {"query": {"search": []}}

    with patch("tools.requests.get", return_value=search_response):
        result = search_wikipedia("asdkjhaslkdjhaslkdjh")

    assert result["error"] is None
    assert result["url"] == ""
    assert result["sources"] == []
    assert "note" in result
