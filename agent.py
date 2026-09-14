"""
Agent loop: the tool-calling conversation between the user, Groq's model, and
our local tools (tools.py). This module knows nothing about Streamlit, it
exposes a single entry point, `run_agent`, that a UI (app.py) or a CLI can
call with a question and a callback for reporting progress.

Architecture:
  tools.py  -> what the tools DO (arXiv/Wikipedia HTTP calls)
  agent.py  -> the LOOP: send messages + tool schemas to Groq, execute any
               requested tool calls, feed results back, repeat until the
               model returns a final answer (or we hit MAX_ITERATIONS)
  app.py    -> the UI: renders the loop's progress and final answer

Citation verification lives here too (verify_citations), since it needs the
same "sources actually retrieved this conversation" list the loop builds.
"""

import json
import os
import re

from groq import Groq

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "openai/gpt-oss-120b"
# With three tools available, a thorough answer can reasonably involve one
# Wikipedia call plus arXiv and Semantic Scholar (each possibly needing one
# retry on failure), so this gives enough headroom for that while still
# bounding runaway loops. Hitting this cap always raises AgentError with a
# clear message, it never fails silently or loops forever.
MAX_ITERATIONS = 10

# The system prompt asks the model to stop retrying a tool after one extra
# attempt, but prompt instructions are a request, not a guarantee, models
# (this one included, in testing) sometimes keep hammering a failing tool
# anyway. This cap is enforced in code as the actual backstop: once a tool
# has errored this many times in a conversation, further calls to it are
# short-circuited locally (no network request) with a message telling the
# model to stop, rather than trusting it to self-regulate.
MAX_TOOL_FAILURES = 2

SYSTEM_PROMPT = """You are a careful research assistant with access to three tools:
- search_arxiv: searches preprints on arXiv, strong for recent/cutting-edge research
- search_semantic_scholar: searches published/peer-reviewed papers across all fields,
  broader than arXiv, use it to cross-check a claim or for fields arXiv covers thinly
- search_wikipedia: fetches background/context from Wikipedia

Rules you must follow:
1. Only cite a source if you actually retrieved it via a tool call in this
   conversation. Never cite a paper, article, or fact from your own training
   knowledge as if you looked it up, if you did not call a tool for it, it
   is not a citation, it is background knowledge, and you should say so
   plainly rather than presenting it as sourced.
2. When you cite a source, use *exactly* this Markdown link syntax:
   [Title](url) - square brackets around the title, immediately followed
   by the url in parentheses, using the exact title and url returned by
   the tool. For example: according to [Few-shot learning](https://en.wikipedia.org/wiki/Few-shot_learning),
   ... Do not use any other citation style (no bracketed bare urls, no
   footnote markers, no numbered references), only this Markdown link
   form, so the UI can render it as a clickable link.
3. If the tool results do not adequately answer the question, say so
   explicitly, for example "The retrieved sources do not cover X" rather
   than filling the gap from memory.
4. Use search_arxiv and/or search_semantic_scholar for academic/technical/
   research questions (both if you want preprint plus peer-reviewed
   coverage, or to cross-check a claim), and search_wikipedia for
   background, definitions, history, or general context. Use whichever
   combination the question actually needs. You may call a tool more than
   once, for example to refine a query that returned nothing useful, but
   never call the same tool again with a query that means essentially the
   same thing as one you already successfully got a result for, reuse that
   result instead. Your tool-call budget is limited, spend it on genuinely
   new information, not repeats.
5. If a tool call returns an error (not just "no results", an actual
   error), retry that tool at most once more, ideally with a reworded
   query. If it fails again, stop calling it, do not keep retrying the
   same failing tool with new phrasings. Move on and answer using whatever
   you did successfully retrieve, and say plainly in your final answer
   which source was unavailable, for example "arXiv search was unavailable
   during this query, so the answer below draws only on Wikipedia." A
   partial answer with an honest caveat is always better than exhausting
   your tool-call budget on a tool that is not responding.
6. When you have enough information, write a final synthesized answer with
   inline citations. Do not call any more tools once you are ready to answer.
"""

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


class AgentError(Exception):
    """Raised when the agent loop cannot produce a final answer."""


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise AgentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


def run_agent(question: str, on_event=None) -> dict:
    """
    Run the tool-calling agent loop for a single user question.

    `on_event` is an optional callback invoked with a dict describing each
    step as it happens, e.g. {"type": "tool_call", "name": ..., "args": ...}
    so a UI can show live progress. If omitted, the loop runs silently.

    Returns a dict:
        {
            "answer": str,               # final text answer from the model
            "sources": list[dict],       # every {"title", "url"} retrieved
            "unverified_citations": list[str],  # cited urls not in sources
            "tool_calls": list[dict],    # log of every tool call made
        }

    Raises AgentError if the loop hits MAX_ITERATIONS without a final answer,
    or if the API key is missing.
    """
    if on_event is None:
        on_event = lambda event: None

    client = _get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    retrieved_sources = []  # every {"title", "url"} dict actually returned by a tool
    tool_call_log = []
    tool_failure_counts = {}  # name -> consecutive-ish error count this conversation

    for iteration in range(1, MAX_ITERATIONS + 1):
        on_event({"type": "thinking", "iteration": iteration})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            # Final answer reached.
            answer = message.content or ""
            unverified = verify_citations(answer, retrieved_sources)
            on_event({"type": "final_answer", "answer": answer})
            return {
                "answer": answer,
                "sources": retrieved_sources,
                "unverified_citations": unverified,
                "tool_calls": tool_call_log,
            }

        # The model wants to call one or more tools. Append its request to
        # the conversation, then execute each call and append the results.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            on_event({"type": "tool_call", "name": name, "args": args})

            if tool_failure_counts.get(name, 0) >= MAX_TOOL_FAILURES:
                # Backstop: this tool has already failed too many times.
                # Don't hit the network again, tell the model to give up on it.
                result = {
                    "error": (
                        f"{name} has failed {tool_failure_counts[name]} time(s) already "
                        "this conversation and will not be retried. Stop calling this "
                        "tool and answer with whatever information is already available, "
                        "noting that this source was unavailable."
                    )
                }
            else:
                func = TOOL_FUNCTIONS.get(name)
                if func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    result = func(**args)

            if result.get("error"):
                tool_failure_counts[name] = tool_failure_counts.get(name, 0) + 1

            tool_call_log.append({"name": name, "args": args, "result": result})

            for source in result.get("sources", []) or []:
                if source not in retrieved_sources:
                    retrieved_sources.append(source)

            on_event({"type": "tool_result", "name": name, "result": result})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    raise AgentError(
        f"Agent did not produce a final answer within {MAX_ITERATIONS} iterations. "
        "The question may need to be narrowed, or the model kept requesting tools "
        "without converging."
    )


def verify_citations(answer: str, retrieved_sources: list) -> list:
    """
    Parse [Title](url) citations out of `answer` and return the list of
    cited urls that do NOT appear among `retrieved_sources`. An empty
    return means every citation is backed by an actual tool result.
    """
    retrieved_urls = {source["url"] for source in retrieved_sources}
    cited_urls = {url for _title, url in CITATION_PATTERN.findall(answer)}
    return sorted(cited_urls - retrieved_urls)
