#!/usr/bin/env python3
"""
Integration test for the LLM client.

Sends a simple GPU-related question and prints the response.
Run from the project root:
    python3 test_llm.py

Requires the API_KEY environment variable (or .env file) to be set.
"""

import logging
import sys

# Show retry warnings and LLM debug traces
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from src.llm_client import LLMClient


def main():
    print("=" * 60)
    print(" LLM Client Integration Test")
    print("=" * 60)

    client = LLMClient()

    question = (
        "Please explain in one sentence what a GPU Shared Memory Bank Conflict is."
    )
    print(f"\nQuestion: {question}\n")

    answer = client.generate_reasoning(
        system_prompt="You are a concise GPU architecture expert.",
        user_prompt=question,
    )

    print("-" * 60)
    print(f"Answer:\n{answer}")
    print("-" * 60)

    if answer.strip():
        print("\n[PASS] LLM client returned a non-empty response.")
    else:
        print("\n[FAIL] LLM client returned an empty response.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
