"""
Streamlit UI for the research assistant agent.

This file only renders things: chat history, live tool-call progress, the
final answer, and any unverified-citation flags. All the actual logic (the
Groq conversation loop, tool execution, citation verification) lives in
agent.py / tools.py.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from agent import AgentError, run_agent

load_dotenv()

st.set_page_config(page_title="Research Assistant Agent", page_icon="🔎", layout="centered")

st.title("🔎 Research Assistant Agent")
st.caption(
    "Ask a research question. The agent searches arXiv and Wikipedia as needed, "
    "and only cites sources it actually retrieved."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key, "
        "and restart the app."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": ..., "result": {...} or None, "error": ... or None}


def render_tool_call(name: str, args: dict):
    query = args.get("query", "")
    extra = f", max_results={args['max_results']}" if "max_results" in args else ""
    icon = "📄" if name == "search_arxiv" else "📚"
    st.markdown(f"{icon} **{name}**(`query=\"{query}\"{extra}`)")


def render_tool_result(name: str, result: dict):
    if result.get("error"):
        st.markdown(f"&nbsp;&nbsp;↳ :red[error: {result['error']}]")
        return
    if result.get("note"):
        st.markdown(f"&nbsp;&nbsp;↳ {result['note']}")
        return
    if name == "search_arxiv":
        n = len(result.get("results", []))
        st.markdown(f"&nbsp;&nbsp;↳ found {n} paper(s)")
    elif name == "search_wikipedia":
        title = result.get("title", "")
        st.markdown(f"&nbsp;&nbsp;↳ found page: {title}")


def render_result(result: dict):
    st.markdown(result["answer"])

    if result["unverified_citations"]:
        st.warning(
            "⚠️ **Unverified citation(s) detected** — these URLs appear in the answer "
            "but were not returned by any tool call this conversation:\n\n"
            + "\n".join(f"- {url}" for url in result["unverified_citations"])
        )

    if result["sources"]:
        with st.expander(f"Sources retrieved this turn ({len(result['sources'])})"):
            for source in result["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")


# --- Replay history -----------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
        else:
            for event in turn["events"]:
                if event["type"] == "tool_call":
                    render_tool_call(event["name"], event["args"])
                elif event["type"] == "tool_result":
                    render_tool_result(event["name"], event["result"])
            render_result(turn["result"])


# --- New question ---------------------------------------------------------
question = st.chat_input("Ask a research question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    turn = {"question": question, "events": [], "result": None, "error": None}

    with st.chat_message("assistant"):
        progress_area = st.container()

        def on_event(event):
            turn["events"].append(event)
            with progress_area:
                if event["type"] == "tool_call":
                    render_tool_call(event["name"], event["args"])
                elif event["type"] == "tool_result":
                    render_tool_result(event["name"], event["result"])
                # "thinking" and "final_answer" events are not rendered directly,
                # the final answer is rendered once after the loop returns.

        try:
            with st.spinner("Researching..."):
                result = run_agent(question, on_event=on_event)
            turn["result"] = result
            render_result(result)
        except AgentError as exc:
            turn["error"] = str(exc)
            st.error(str(exc))

    st.session_state.history.append(turn)
