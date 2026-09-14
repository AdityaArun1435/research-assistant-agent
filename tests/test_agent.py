"""
Unit tests focused on the parts of this project that don't require a live
Groq API key: citation verification, and the tool functions' XML/JSON
parsing (using mocked HTTP responses, no real network calls).

Run with: pytest
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import AgentError, MAX_TOOL_FAILURES, run_agent, verify_citations
from tools import search_arxiv, search_semantic_scholar, search_wikipedia


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


# --- search_semantic_scholar -----------------------------------------------

def test_search_semantic_scholar_parses_results():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "data": [
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authors": [{"name": "Kaiming He"}, {"name": "Xiangyu Zhang"}],
                "abstract": "Deeper neural networks are more difficult to train...",
                "year": 2016,
                "venue": "CVPR",
                "url": "https://www.semanticscholar.org/paper/abc123",
            }
        ]
    }

    with patch("tools.requests.get", return_value=mock_response):
        result = search_semantic_scholar("resnet", max_results=1)

    assert result["error"] is None
    assert len(result["results"]) == 1
    paper = result["results"][0]
    assert paper["title"] == "Deep Residual Learning for Image Recognition"
    assert paper["authors"] == ["Kaiming He", "Xiangyu Zhang"]
    assert paper["year"] == 2016
    assert result["sources"] == [
        {"title": "Deep Residual Learning for Image Recognition", "url": "https://www.semanticscholar.org/paper/abc123"}
    ]


def test_search_semantic_scholar_handles_429():
    mock_response = Mock()
    mock_response.status_code = 429

    with patch("tools.requests.get", return_value=mock_response):
        result = search_semantic_scholar("anything")

    assert result["error"] is not None
    assert "rate-limiting" in result["error"]
    assert result["results"] == []
    assert result["sources"] == []


def test_search_semantic_scholar_skips_papers_without_url():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "data": [
            {"title": "No URL Paper", "authors": [], "url": ""},
            {"title": "Has URL Paper", "authors": [], "url": "https://example.com/paper"},
        ]
    }

    with patch("tools.requests.get", return_value=mock_response):
        result = search_semantic_scholar("anything")

    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Has URL Paper"


# --- run_agent's tool-failure backstop -------------------------------------
# In manual testing, the model sometimes ignored the system prompt's "stop
# retrying a failing tool" instruction and kept calling it anyway. These
# tests confirm the code-level backstop (MAX_TOOL_FAILURES) holds regardless
# of what the model does, since a prompt instruction alone isn't a guarantee.

def _tool_call_response(call_id, name="search_arxiv", arguments='{"query": "test"}'):
    """Build a fake Groq response whose message requests one tool call."""
    tool_call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _AlwaysRequestsToolClient:
    """Fake Groq client whose model never stops calling the tool, simulating
    a model that doesn't follow the 'stop retrying' prompt instruction."""

    def __init__(self):
        self.call_count = 0

    def _create(self, **_kwargs):
        self.call_count += 1
        return _tool_call_response(call_id=str(self.call_count))

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def test_tool_failure_backstop_caps_real_invocations():
    real_call_count = SimpleNamespace(n=0)

    def always_failing_tool(**_kwargs):
        real_call_count.n += 1
        return {"error": "boom", "results": [], "sources": []}

    fake_client = _AlwaysRequestsToolClient()

    with patch.object(agent, "_get_client", return_value=fake_client), \
         patch.dict(agent.TOOL_FUNCTIONS, {"search_arxiv": always_failing_tool}):
        try:
            run_agent("irrelevant question")
            assert False, "expected AgentError since the fake model never stops calling tools"
        except AgentError:
            pass

    # The real tool function is only ever invoked up to MAX_TOOL_FAILURES
    # times, every call after that is short-circuited locally in agent.py.
    assert real_call_count.n == MAX_TOOL_FAILURES
    # But the model kept "requesting" the tool for the full iteration budget.
    assert fake_client.call_count == agent.MAX_ITERATIONS
