from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for a specific model provider."""

    base_url: str
    provider_type: str
    api_key_path: str | None = None
