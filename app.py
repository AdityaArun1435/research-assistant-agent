"""
Streamlit UI for the research assistant agent.

This file only renders things: chat history, live tool-call progress, the
final answer, and any unverified-citation flags. All the actual logic (the
Groq conversation loop, tool execution, citation verification) lives in
agent.py / tools.py.
"""

import base64
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import AgentError, run_agent

load_dotenv()

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(page_title="Research Assistant Agent", page_icon=str(LOGO_PATH), layout="centered")

# Header: logo + title side by side. Built as one inline-HTML block (rather
# than st.image()/st.title() in separate st.columns) so the two align on
# the same baseline instead of stacking or drifting apart at odd widths.
_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
        <img src="data:image/png;base64,{_logo_b64}" width="42" height="42" />
        <h1 style="margin:0; font-size:2rem;">Research Assistant Agent</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Ask a research question. The agent searches arXiv, Semantic Scholar, and Wikipedia "
    "as needed, and only cites sources it actually retrieved."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key, "
        "and restart the app."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": ..., "result": {...} or None, "error": ... or None}


# Short display label + accent color per tool, used for the badge shown
# next to each live tool call. No emoji, color is the only differentiator.
_TOOL_BADGES = {
    "search_arxiv": ("arXiv", "#2563eb", "#eff6ff"),
    "search_semantic_scholar": ("Semantic Scholar", "#7c3aed", "#f5f3ff"),
    "search_wikipedia": ("Wikipedia", "#16a34a", "#f0fdf4"),
}


def _badge(name: str) -> str:
    label, color, bg = _TOOL_BADGES.get(name, (name, "#475569", "#f1f5f9"))
    return (
        f'<span style="background:{bg}; color:{color}; padding:1px 8px; '
        f'border-radius:4px; font-size:0.85em; font-weight:600;">{label}</span>'
    )


def render_tool_call(name: str, args: dict):
    query = args.get("query", "")
    extra = f", max_results={args['max_results']}" if "max_results" in args else ""
    st.markdown(f"{_badge(name)} `query=\"{query}\"{extra}`", unsafe_allow_html=True)


def render_tool_result(name: str, result: dict):
    if result.get("error"):
        st.markdown(f"&nbsp;&nbsp;↳ :red[error: {result['error']}]")
        return
    if result.get("note"):
        st.markdown(f"&nbsp;&nbsp;↳ {result['note']}")
        return
    if name in ("search_arxiv", "search_semantic_scholar"):
        n = len(result.get("results", []))
        st.markdown(f"&nbsp;&nbsp;↳ found {n} paper(s)")
    elif name == "search_wikipedia":
        title = result.get("title", "")
        st.markdown(f"&nbsp;&nbsp;↳ found page: {title}")


def render_result(result: dict):
    st.markdown(result["answer"])

    if result["unverified_citations"]:
        st.warning(
            "**Unverified citation(s) detected** — these URLs appear in the answer "
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
    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
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

    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
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
