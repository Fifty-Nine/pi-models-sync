from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiscoveredModel:
    """A model discovered from a provider."""

    id: str
    name: str
    provider: str
    context_window: int | None = None
    max_tokens: int | None = None
    input_types: list[str] | None = None
    reasoning: bool | None = None
