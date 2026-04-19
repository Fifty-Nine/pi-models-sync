from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from pi_models_sync.discovery import DiscoveredModel


class PiConfigError(Exception):
    """Exception raised for errors during Pi config generation or backup."""

    def __init__(self, file_path: Path, error_detail: str) -> None:
        message = (
            f"Failed to backup or write config {file_path}: {error_detail}"
        )
        super().__init__(message)


def backup_existing_config(file_path: Path) -> None:
    """
    Back up the existing config file sequentially.

    If `file_path` exists, it renames it to `file_path.1`, `file_path.2`, etc.,
    finding the first available number. If it does not exist, does nothing.

    Args:
        file_path: The path of the file to back up.

    Raises:
        PiConfigError: If there's an OSError during backup.
    """
    if not file_path.exists():
        return

    i = 1
    while file_path.with_name(f"{file_path.name}.{i}").exists():
        i += 1

    backup_path = file_path.with_name(f"{file_path.name}.{i}")
    try:
        shutil.move(str(file_path), str(backup_path))
    except OSError as e:
        raise PiConfigError(file_path, str(e)) from e


def generate_pi_config(
    models: list[DiscoveredModel],
    litellm_url: str,
    api_key: str,
    output_path: Path,
) -> None:
    """
    Generate the Pi agent configuration and write it to a JSON file.

    Args:
        models: The list of models to include in the configuration.
        litellm_url: The base URL of the LiteLLM gateway.
        api_key: The API key for the LiteLLM gateway.
        output_path: The file path where the configuration should be written.

    Raises:
        PiConfigError: If there's an error during backup or file writing.
    """
    base_url = litellm_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    models_data: list[dict[str, Any]] = []
    for model in models:
        model_dict: dict[str, Any] = {"id": model.id}
        if model.name is not None:
            model_dict["name"] = model.name
        if model.context_window is not None:
            model_dict["contextWindow"] = model.context_window
        if model.max_tokens is not None:
            model_dict["maxTokens"] = model.max_tokens
        models_data.append(model_dict)

    config = {
        "providers": {
            "litellm_gateway": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
                "models": models_data,
            }
        }
    }

    backup_existing_config(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600
        with os.fdopen(
            os.open(output_path, flags, mode), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    except OSError as e:
        raise PiConfigError(output_path, str(e)) from e
