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
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are a careful research assistant with access to two tools:
- search_arxiv: searches academic papers on arXiv
- search_wikipedia: fetches background/context from Wikipedia

Rules you must follow:
1. Only cite a source if you actually retrieved it via a tool call in this
   conversation. Never cite a paper, article, or fact from your own training
   knowledge as if you looked it up, if you did not call a tool for it, it
   is not a citation, it is background knowledge, and you should say so
   plainly rather than presenting it as sourced.
2. When you cite a source, use the exact format [Title](url), using the
   exact title and url returned by the tool, not a paraphrased title or a
   URL you constructed yourself.
3. If the tool results do not adequately answer the question, say so
   explicitly, for example "The retrieved sources do not cover X" rather
   than filling the gap from memory.
4. Use search_arxiv for academic/technical/research questions and
   search_wikipedia for background, definitions, history, or general
   context. Use both when a question needs both. You may call tools more
   than once, for example to refine a query that returned nothing useful.
5. When you have enough information, write a final synthesized answer with
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

            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                result = {"error": f"Unknown tool '{name}'"}
            else:
                result = func(**args)

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
