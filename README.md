# Research Assistant Agent

A tool-calling research assistant that answers questions by actually going
and looking things up, rather than relying on the model's training data. It
searches arXiv for academic papers and Wikipedia for background context, and
verifies after the fact that every citation in its answer actually points to
something it retrieved during the conversation.

Built to be a clean example of the "LLM decides which tools to call, in a
loop, across multiple turns" pattern, not a single-tool wrapper.

## Why this exists

Most LLM demos either (a) don't use tools at all, or (b) wire up one tool and
call it a day. This project is deliberately a *multi*-tool agent that has to
choose between sources, chain calls together, and be honest when its sources
come up short, plus it does not just trust the model's citations, it checks
them.

## Architecture

```
tools.py   -> Tool implementations. Pure functions that call the arXiv API
              and the Wikipedia API and return JSON-serializable dicts, plus
              the OpenAI-style tool schemas Groq needs to see. Knows nothing
              about the agent loop or the UI.

agent.py   -> The agent loop. Sends the conversation + tool schemas to Groq,
              executes any requested tool calls, feeds results back, repeats
              until the model returns a final answer or MAX_ITERATIONS (6)
              is hit. Also owns citation verification: after the model
              answers, every [Title](url) in the answer is checked against
              the URLs actually returned by tool calls this conversation.
              Unmatched URLs are returned as "unverified_citations", not
              silently trusted. Knows nothing about Streamlit.

app.py     -> Streamlit UI. Renders the chat, live tool-call progress (which
              tool, what query, as it happens), the final answer with
              clickable citations, and unverified-citation warnings. Contains
              no business logic, it only calls run_agent() and renders what
              comes back.

cli.py     -> Minimal terminal interface to the same agent loop, useful for
              quick testing without starting Streamlit.

tests/     -> Unit tests for citation verification and tool parsing, using
              mocked HTTP responses (no live API calls, no Groq key needed).
```

The separation matters here specifically: `tools.py` could gain a third tool
(say, Semantic Scholar) without touching `agent.py`'s loop logic, and the UI
could be swapped for a CLI-only or FastAPI interface without touching either.

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

## Example questions to try

- "What does recent research say about retrieval-augmented generation, and
  what is RAG in the first place?" (should trigger both `search_arxiv` and
  `search_wikipedia`, one for the technical literature, one for background)
- "What's the current state of research on quantum error correction?"
  (should trigger multiple `search_arxiv` calls, possibly refining the query)

Try also asking something obscure or made-up, the agent should say the
retrieved sources don't answer it rather than inventing a citation.

## Citation discipline, how it actually works

The system prompt instructs the model to only cite what it retrieved, use
`[Title](url)` format, and admit when sources fall short. But prompts alone
don't guarantee compliance, models occasionally hallucinate a plausible-
looking URL. So after every final answer, `agent.verify_citations()` extracts
every `[Title](url)` pattern from the text and diffs the URLs against the
list of URLs actually returned by tool calls that conversation. Anything that
doesn't match is surfaced in the UI as an "unverified citation" warning, in
the terminal, and in the returned result dict, never silently dropped or
silently trusted.

## Notes

- The arXiv API has no key requirement but is rate-limited by IP, avoid
  hammering it in a tight loop.
- The Wikipedia calls use MediaWiki's public API and send a descriptive
  `User-Agent` header per their etiquette guidelines.
- `MAX_ITERATIONS` (6) in `agent.py` caps the tool-calling loop. If it's hit,
  `run_agent` raises `AgentError` with a clear message rather than looping
  forever.
