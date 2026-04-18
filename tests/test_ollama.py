from __future__ import annotations

import pathlib

import requests
import responses

from pi_models_sync.providers.config import ProviderConfig
from pi_models_sync.providers.ollama import (
    CloudOllamaProvider,
    LocalOllamaProvider,
)


@responses.activate
def test_local_ollama_provider_success() -> None:
    config = ProviderConfig(
        base_url="http://localhost:11434", provider_type="local"
    )
    provider = LocalOllamaProvider(config)

    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={
            "models": [
                {"name": "llama3:8b"},
                {"name": "qwen:7b"},
            ]
        },
        status=200,
    )

    models = list(provider.get_models())
    assert len(models) == 2
    assert models[0].id == "llama3:8b"
    assert models[0].name == "llama3:8b"
    assert models[0].provider == "local-ollama"
    assert models[1].id == "qwen:7b"
    assert models[1].name == "qwen:7b"
    assert models[1].provider == "local-ollama"


@responses.activate
def test_local_ollama_provider_network_error() -> None:
    config = ProviderConfig(
        base_url="http://localhost:11434", provider_type="local"
    )
    provider = LocalOllamaProvider(config)

    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        body=requests.exceptions.ConnectionError("Connection refused"),
    )

    models = list(provider.get_models())
    assert len(models) == 0


@responses.activate
def test_cloud_ollama_provider_no_key() -> None:
    # Test path where _api_key is somehow None despite not being disabled
    # (e.g. if _load_api_key was overridden or manual modification occurred)
    tmp_path = pathlib.Path("dummy")
    config = ProviderConfig(
        base_url="https://cloud.ollama.ai",
        provider_type="cloud",
        api_key_path=str(tmp_path),
    )
    provider = CloudOllamaProvider(config)
    provider._disabled = False
    provider._api_key = None

    responses.add(
        responses.GET,
        "https://cloud.ollama.ai/api/tags",
        json={"models": [{"name": "test-model"}]},
        status=200,
    )

    models = list(provider.get_models())
    assert len(models) == 1


@responses.activate
def test_cloud_ollama_provider_success(tmp_path: pathlib.Path) -> None:
    key_file = tmp_path / "cloud_ollama.key"
    key_file.write_text("test-api-key")

    config = ProviderConfig(
        base_url="https://cloud.ollama.ai",
        provider_type="cloud",
        api_key_path=str(key_file),
    )
    provider = CloudOllamaProvider(config)

    responses.add(
        responses.GET,
        "https://cloud.ollama.ai/api/tags",
        json={
            "models": [
                {"name": "mixtral:8x7b"},
            ]
        },
        status=200,
        match=[
            responses.matchers.header_matcher(
                {"Authorization": "Bearer test-api-key"}
            )
        ],
    )

    models = list(provider.get_models())
    assert len(models) == 1
    assert models[0].id == "mixtral:8x7b"
    assert models[0].provider == "cloud-ollama"


def test_cloud_ollama_provider_missing_key() -> None:
    config = ProviderConfig(
        base_url="https://cloud.ollama.ai",
        provider_type="cloud",
        api_key_path="nonexistent.key",
    )
    provider = CloudOllamaProvider(config)

    models = list(provider.get_models())
    assert len(models) == 0


def test_cloud_ollama_provider_empty_key(tmp_path: pathlib.Path) -> None:
    key_file = tmp_path / "empty_cloud_ollama.key"
    key_file.write_text("")

    config = ProviderConfig(
        base_url="https://cloud.ollama.ai",
        provider_type="cloud",
        api_key_path=str(key_file),
    )
    provider = CloudOllamaProvider(config)

    models = list(provider.get_models())
    assert len(models) == 0


def test_cloud_ollama_provider_no_key_path_configured() -> None:
    config = ProviderConfig(
        base_url="https://cloud.ollama.ai",
        provider_type="cloud",
    )
    provider = CloudOllamaProvider(config)

    models = list(provider.get_models())
    assert len(models) == 0
