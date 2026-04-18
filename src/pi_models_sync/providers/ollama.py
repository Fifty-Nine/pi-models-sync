from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import requests

from pi_models_sync.discovery import DiscoveredModel
from pi_models_sync.providers.base import ModelProvider

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pi_models_sync.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    """Base provider for Ollama instances."""

    provider_name = "ollama"

    def get_models(self) -> Iterator[DiscoveredModel]:
        """Fetch models from the Ollama instance."""
        headers = self._get_headers()
        url = f"{self.config.base_url.rstrip('/')}/api/tags"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Failed to fetch models from %s: %s", self.provider_name, e
            )
            return

        for model_data in data.get("models", []):
            yield self._parse_model(model_data)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for the request."""
        return {}

    def _parse_model(self, data: dict[str, Any]) -> DiscoveredModel:
        """Parse raw model data into a DiscoveredModel."""
        name = data.get("name", "")
        # Ollama /api/tags does not return context window, max tokens, etc
        # directly by default. We populate what we can.
        return DiscoveredModel(
            id=name,
            name=name,
            provider=self.provider_name,
        )


class LocalOllamaProvider(OllamaProvider):
    """Provider for a local Ollama instance."""

    provider_name = "local-ollama"


class CloudOllamaProvider(OllamaProvider):
    """Provider for a cloud Ollama instance."""

    provider_name = "cloud-ollama"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._disabled = False
        self._api_key: str | None = None

        if not config.api_key_path:
            logger.warning(
                "Cloud Ollama provider configured without an API key path. "
                "Provider will be disabled."
            )
            self._disabled = True
            return

        path = pathlib.Path(config.api_key_path)
        self._load_api_key(path)

    def _load_api_key(self, path: pathlib.Path) -> None:
        try:
            with path.open() as f:
                self._api_key = f.read().strip()
            if not self._api_key:
                # Treat empty file as error handled in the same way
                self._disabled = True
                logger.warning(
                    "Cloud Ollama API key file is empty at %s. "
                    "Provider will be disabled.",
                    path,
                )
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(
                "Cloud Ollama API key file not found or invalid at %s. "
                "Provider will be disabled. Error: %s",
                path,
                e,
            )
            self._disabled = True

    def get_models(self) -> Iterator[DiscoveredModel]:
        """Fetch models, bypassing if disabled."""
        if self._disabled:
            return iter([])
        return super().get_models()

    def _get_headers(self) -> dict[str, str]:
        """Add authorization header."""
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}
