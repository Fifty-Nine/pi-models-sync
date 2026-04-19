from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from pi_models_sync.discovery import DiscoveredModel
    from pi_models_sync.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


class LiteLLMSyncError(Exception):
    """Base exception for LiteLLM client errors."""


class ProviderKeyReadError(LiteLLMSyncError):
    """Exception raised when an API key cannot be read."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to read API key at {path}: {reason}")


class InvalidResponseFormatError(LiteLLMSyncError):
    """Exception raised when a response does not match the expected
    structure.
    """

    def __init__(self, url: str, data: Any) -> None:
        super().__init__(f"Unexpected response format from {url}: {data}")


class InvalidResponseDataError(LiteLLMSyncError):
    """Exception raised when the 'data' field is not of the expected type."""

    def __init__(self, got_type: str) -> None:
        super().__init__(f"Expected 'data' field to be a list, got {got_type}")


class LiteLLMClient:
    """Client for interacting with the LiteLLM Gateway."""

    def __init__(
        self,
        base_url: str,
        master_key_path: str | None = None,
        inference_key_path: str | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize the LiteLLM client.

        Args:
            base_url: The base URL of the LiteLLM instance.
            master_key_path: Path to the LiteLLM master key file.
            inference_key_path: Path to the LiteLLM inference key file.
            dry_run: Whether to simulate actions without modifying state.
        """
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run

        self.master_key = self._load_key(master_key_path, "master")
        self.inference_key = self._load_key(inference_key_path, "inference")

    def _load_key(self, path_str: str | None, key_name: str) -> str | None:
        """Load an API key from a file path."""
        if not path_str:
            return None

        path = pathlib.Path(path_str)
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ProviderKeyReadError(str(path), str(e)) from e

        if not key:
            logger.warning(
                "LiteLLM %s key file is empty at %s.", key_name, path
            )
            return ""
        return key

    def get_inference_models(self) -> list[str]:
        """Fetch models available for inference.

        Returns:
            A list of model IDs.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        url = f"{self.base_url}/v1/models"
        headers = {}

        # Use inference key, fallback to master key
        if auth_key := self.inference_key or self.master_key:
            headers["Authorization"] = f"Bearer {auth_key}"

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict) or "data" not in data:
            raise InvalidResponseFormatError(url, data)

        models = data["data"]
        if not isinstance(models, list):
            raise InvalidResponseDataError(type(models).__name__)

        return [
            str(model["id"])
            for model in models
            if isinstance(model, dict) and "id" in model
        ]

    def get_configured_models(self) -> list[str]:
        """Fetch models currently configured in the LiteLLM instance.

        Returns:
            A list of model names.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        url = f"{self.base_url}/model/info"
        headers = {}
        if self.master_key:
            headers["Authorization"] = f"Bearer {self.master_key}"

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict) or "data" not in data:
            raise InvalidResponseFormatError(url, data)

        models = data["data"]
        if not isinstance(models, list):
            raise InvalidResponseDataError(type(models).__name__)

        return [
            str(model["model_name"])
            for model in models
            if isinstance(model, dict) and "model_name" in model
        ]

    def add_model(
        self, model: DiscoveredModel, provider_config: ProviderConfig
    ) -> None:
        """Add a newly discovered model to the LiteLLM Gateway.

        Args:
            model: The discovered model to add.
            provider_config: The provider configuration for this model.

        Raises:
            ProviderKeyReadError: If the provider API key file cannot be read.
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        model_name = f"{model.provider}/{model.id}"

        if self.dry_run:
            logger.info("Dry run: Would add model %s", model_name)
            return

        litellm_params = {
            "model": f"{provider_config.provider_type}/{model.id}",
            "api_base": provider_config.base_url,
        }

        if key := self._load_key(provider_config.api_key_path, "provider"):
            litellm_params["api_key"] = key

        url = f"{self.base_url}/model/new"
        headers = {}
        if self.master_key:
            headers["Authorization"] = f"Bearer {self.master_key}"
        payload = {"model_name": model_name, "litellm_params": litellm_params}

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
