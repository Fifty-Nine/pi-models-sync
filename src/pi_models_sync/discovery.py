from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiscoveredModel:
    """A model discovered from a provider."""

    id: str
    name: str
    provider: str
