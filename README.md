# Research Assistant Agent

A tool-calling research assistant that answers questions by actually going
and looking things up, rather than relying on the model's training data. It
searches arXiv and Semantic Scholar for academic papers and Wikipedia for
background context, and verifies after the fact that every citation in its
answer actually points to something it retrieved during the conversation.

Built to be a clean example of the "LLM decides which tools to call, in a
loop, across multiple turns" pattern, not a single-tool wrapper.

## Why this exists

Most LLM demos either (a) don't use tools at all, or (b) wire up one tool and
call it a day. This project is deliberately a *multi*-tool agent that has to
choose between sources, chain calls together, and be honest when its sources
come up short, plus it does not just trust the model's citations, it checks
them. It also doesn't just trust the model's *behavior*: a few things below
(the tool-failure backstop especially) exist because live testing showed the
model doesn't always follow prompt instructions, so the loop enforces the
important guarantees in code instead of hoping the model complies.

## Architecture

```
tools.py   -> Tool implementations. Pure functions that call the arXiv,
              Semantic Scholar, and Wikipedia APIs and return
              JSON-serializable dicts, plus the OpenAI-style tool schemas
              Groq needs to see. Knows nothing about the agent loop or the UI.

agent.py   -> The agent loop. Sends the conversation + tool schemas to Groq,
              executes any requested tool calls, feeds results back, repeats
              until the model returns a final answer or MAX_ITERATIONS (10)
              is hit. Enforces a per-tool failure backstop (MAX_TOOL_FAILURES,
              2): once a tool has errored twice in a conversation, further
              calls to it are short-circuited locally (no network request)
              rather than trusting the model to stop retrying on its own, see
              "Lessons from live testing" below. Also owns citation
              verification: after the model answers, every [Title](url) in
              the answer is checked against the URLs actually returned by
              tool calls this conversation. Unmatched URLs are returned as
              "unverified_citations", not silently trusted. Knows nothing
              about Streamlit.

app.py     -> Streamlit UI. Renders the chat, live tool-call progress (which
              tool, what query, as it happens), the final answer with
              clickable citations, and unverified-citation warnings. Contains
              no business logic, it only calls run_agent() and renders what
              comes back.

cli.py     -> Minimal terminal interface to the same agent loop, useful for
              quick testing without starting Streamlit.

tests/     -> Unit tests for citation verification, tool parsing (mocked
              HTTP, no live calls or Groq key needed), and the tool-failure
              backstop (a fake Groq client that never stops requesting a
              failing tool, asserting the real tool function is still only
              ever invoked MAX_TOOL_FAILURES times).

assets/    -> generate_logo.py draws logo.png (the app icon/header mark)
              programmatically with Pillow, so the design is versioned as
              readable code, not an opaque binary. Re-run it after editing
              colors/proportions: python assets/generate_logo.py
```

The separation matters here specifically: `tools.py` gained a third tool
(Semantic Scholar) without touching `agent.py`'s loop logic at all, and the
UI could be swapped for a CLI-only or FastAPI interface without touching
either.

## The model

Uses Groq's `openai/gpt-oss-120b`, OpenAI's open-weight flagship model
hosted on Groq's infrastructure. Chosen and verified by calling Groq's live
`/models` endpoint and test-firing a tool call against it directly with this
project's own API key as of September 2026, not by trusting documentation
alone: an initial choice of `llama-3.3-70b-versatile` (Groq's previously
recommended tool-calling model per their docs) turned out to have been
retired from the platform entirely, returning a 404. Of the models actually
live on the account, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and
`qwen/qwen3.6-27b` all confirmed working for tool calls; `gpt-oss-120b` was
picked as the largest and most capable. If you're reading this much later,
re-run the same check, Groq's lineup changes over time:

```bash
python -c "from groq import Groq; import os; from dotenv import load_dotenv; load_dotenv(); print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])"
```

## Setup

1. Install dependencies (creates no system-wide changes, everything below
   assumes a virtual environment):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Groq API key from [console.groq.com/keys](https://console.groq.com/keys).
   (arXiv, Semantic Scholar, and Wikipedia need no API key.)

3. Copy the example env file and add your key:

   ```bash
   copy .env.example .env
   ```

   Then edit `.env` and set `GROQ_API_KEY`. This file is gitignored and
   never committed.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

   Or use the CLI instead:

   ```bash
   python cli.py "What are the main approaches to few-shot learning?"
   ```

5. Run tests (no API key required, all HTTP calls are mocked):

   ```bash
   pytest
   ```

## Deploying (Streamlit Community Cloud)

The repo is public at
[github.com/AdityaArun1435/research-assistant-agent](https://github.com/AdityaArun1435/research-assistant-agent),
so it can be deployed for free on Streamlit Community Cloud without a
separate build step:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **New app**, pick this repo, branch `master`, main file `app.py`.
3. Before (or after) deploying, open **Settings -> Secrets** on the app and
   paste in the contents of `.streamlit/secrets.toml.example` with your real
   key:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

   `app.py` reads `GROQ_API_KEY` from the environment first (how local
   `.env` works) and falls back to `st.secrets` (how Streamlit Cloud's
   dashboard secrets work), so the same code runs in both places unchanged.
4. Deploy. First boot installs `requirements.txt`, subsequent pushes to
   `master` redeploy automatically.

## Example questions to try

- "What does recent research say about retrieval-augmented generation, and
  what is RAG in the first place?" (should trigger `search_arxiv` and/or
  `search_semantic_scholar` for the technical literature, plus
  `search_wikipedia` for background)
- "What's the current state of research on quantum error correction?"
  (academic-only, should trigger `search_arxiv` and/or `search_semantic_scholar`,
  possibly refining the query if the first one comes back thin)

Try also asking something obscure or made-up, the agent should say the
retrieved sources don't answer it rather than inventing a citation.

## Citation discipline, how it actually works

The system prompt instructs the model to only cite what it retrieved, use
`[Title](url)` format, and admit when sources fall short. But prompts alone
don't guarantee compliance, models occasionally hallucinate a plausible-
looking URL, or use a different bracket/footnote style that a stricter prompt
had to specifically rule out (see "Lessons from live testing"). So after
every final answer, `agent.verify_citations()` extracts every `[Title](url)`
pattern from the text and diffs the URLs against the list of URLs actually
returned by tool calls that conversation. Anything that doesn't match is
surfaced in the UI as an "unverified citation" warning, in the terminal, and
in the returned result dict, never silently dropped or silently trusted.

## Lessons from live testing

A few design decisions here exist specifically because live testing against
the real model surfaced behavior a spec or a first-draft prompt didn't
anticipate:

- **A prompt instruction is a request, not a guarantee.** The system prompt
  originally just asked the model to stop retrying a tool after it failed
  once. In testing, the model sometimes ignored that and called arXiv 6
  times in one conversation while it was rate-limited, burning the entire
  iteration budget on a dead tool instead of answering with what it had.
  The fix that actually holds is `MAX_TOOL_FAILURES` in `agent.py`: a
  code-level counter that short-circuits further calls to a tool after it
  has failed twice, regardless of what the model asks for next.
- **arXiv silently penalizes an unidentified client.** `requests`' default
  `python-requests/x.y` User-Agent caused `export.arxiv.org` to hang or
  throttle requests that curl (which sends its own descriptive
  User-Agent) handled instantly. Both arXiv and Wikipedia calls now send a
  descriptive `User-Agent` header.
- **Windows' console isn't UTF-8 by default.** `cli.py` crashed with a
  `UnicodeEncodeError` the first time the model's answer used a curly quote
  or non-breaking hyphen, because Windows' legacy `cp1252` codepage can't
  print them. `cli.py` now forces UTF-8 on stdout/stderr.
- **An exact citation format needs a concrete example, not just a
  description.** "Use `[Title](url)` format" alone was not enough, the
  model sometimes used bracket-enclosed bare URLs instead, which the UI
  can't render as a clickable link. Adding one worked example to the prompt
  fixed it in the next test.

## Notes

- The arXiv and Semantic Scholar APIs need no key but are rate-limited by
  IP; `search_arxiv` retries once with backoff on a 429 before giving up.
  Avoid hammering either in a tight loop.
- The Wikipedia and arXiv calls send a descriptive `User-Agent` header per
  their etiquette guidelines (also fixes a real arXiv reliability issue, see
  above).
- `MAX_ITERATIONS` (10) in `agent.py` caps the tool-calling loop. If it's
  hit, `run_agent` raises `AgentError` with a clear message rather than
  looping forever.
- `MAX_TOOL_FAILURES` (2) caps how many times any single tool can fail
  before further calls to it are short-circuited locally for the rest of
  the conversation, see "Lessons from live testing" above.
