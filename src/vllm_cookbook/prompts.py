"""Prompt helper snippets for reusable demo prompts."""

from __future__ import annotations


def basic_prompt(topic: str) -> str:
    """Return a compact explain-with-example prompt."""
    return f"Explain {topic} briefly and include one practical example."
