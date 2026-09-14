"""
Minimal command-line interface for the agent, useful for quick testing
without starting Streamlit. Run: python cli.py "your question here"
"""

import sys

from dotenv import load_dotenv

from agent import AgentError, run_agent

load_dotenv()


def on_event(event):
    if event["type"] == "tool_call":
        args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
        print(f"  -> calling {event['name']}({args_str})")
    elif event["type"] == "tool_result":
        result = event["result"]
        if result.get("error"):
            print(f"     error: {result['error']}")
        elif result.get("note"):
            print(f"     {result['note']}")
        else:
            n = len(result.get("sources", []))
            print(f"     retrieved {n} source(s)")


def main():
    if len(sys.argv) < 2:
        print('Usage: python cli.py "your research question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    try:
        result = run_agent(question, on_event=on_event)
    except AgentError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print("\n--- Answer ---")
    print(result["answer"])

    if result["unverified_citations"]:
        print("\n--- WARNING: unverified citations ---")
        for url in result["unverified_citations"]:
            print(f"  - {url}")

    if result["sources"]:
        print(f"\n--- Sources retrieved ({len(result['sources'])}) ---")
        for source in result["sources"]:
            print(f"  - {source['title']}: {source['url']}")


if __name__ == "__main__":
    main()
