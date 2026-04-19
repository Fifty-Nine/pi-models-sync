from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest
import requests
import responses

from pi_models_sync.discovery import DiscoveredModel
from pi_models_sync.litellm_client import (
    InvalidResponseFormatError,
    LiteLLMClient,
    LiteLLMSyncError,
    ProviderKeyReadError,
)
from pi_models_sync.providers.config import ProviderConfig

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def master_key_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "master.key"
    p.write_text("test-master-key")
    return p


@pytest.fixture
def inference_key_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "inference.key"
    p.write_text("test-inference-key")
    return p


@pytest.fixture
def provider_key_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "provider.key"
    p.write_text("test-provider-key")
    return p


def test_client_init_keys(
    master_key_file: pathlib.Path, inference_key_file: pathlib.Path
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
        inference_key_path=str(inference_key_file),
    )
    assert client.master_key == "test-master-key"
    assert client.inference_key == "test-inference-key"


def test_client_init_missing_keys() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    assert client.master_key is None
    assert client.inference_key is None


@responses.activate
def test_get_inference_models(
    inference_key_file: pathlib.Path, master_key_file: pathlib.Path
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
        inference_key_path=str(inference_key_file),
    )

    responses.add(
        responses.GET,
        "http://test.com/v1/models",
        json={"data": [{"id": "model-1"}, {"id": "model-2"}]},
        status=200,
    )

    models = client.get_inference_models()
    assert models == ["model-1", "model-2"]

    assert len(responses.calls) == 1
    assert (
        responses.calls[0].request.headers["Authorization"]
        == "Bearer test-inference-key"
    )


@responses.activate
def test_get_inference_models_fallback_master(
    master_key_file: pathlib.Path,
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    responses.add(
        responses.GET,
        "http://test.com/v1/models",
        json={"data": [{"id": "model-3"}]},
        status=200,
    )

    models = client.get_inference_models()
    assert models == ["model-3"]

    assert len(responses.calls) == 1
    assert (
        responses.calls[0].request.headers["Authorization"]
        == "Bearer test-master-key"
    )


@responses.activate
def test_get_inference_models_no_keys() -> None:
    client = LiteLLMClient(base_url="http://test.com")

    responses.add(
        responses.GET,
        "http://test.com/v1/models",
        json={"data": [{"id": "model-4"}]},
        status=200,
    )

    models = client.get_inference_models()
    assert models == ["model-4"]

    assert len(responses.calls) == 1
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_get_configured_models(master_key_file: pathlib.Path) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    responses.add(
        responses.GET,
        "http://test.com/model/info",
        json={"data": [{"model_name": "provider/model-1"}]},
        status=200,
    )

    models = client.get_configured_models()
    assert models == ["provider/model-1"]

    assert len(responses.calls) == 1
    assert (
        responses.calls[0].request.headers["Authorization"]
        == "Bearer test-master-key"
    )


@responses.activate
def test_get_configured_models_no_master_key() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    responses.add(
        responses.GET,
        "http://test.com/model/info",
        json={"data": [{"model_name": "provider/model-1"}]},
        status=200,
    )
    models = client.get_configured_models()
    assert models == ["provider/model-1"]
    assert len(responses.calls) == 1
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_add_model_success(
    master_key_file: pathlib.Path, provider_key_file: pathlib.Path
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    responses.add(
        responses.POST,
        "http://test.com/model/new",
        json={"status": "success"},
        status=200,
    )

    model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
    provider_config = ProviderConfig(
        base_url="http://provider.com",
        provider_type="local",
        api_key_path=str(provider_key_file),
    )

    client.add_model(model, provider_config)

    assert len(responses.calls) == 1
    req = responses.calls[0].request
    assert req.headers["Authorization"] == "Bearer test-master-key"
    assert isinstance(req.body, (str, bytes, bytearray))

    body = json.loads(req.body)
    assert body["model_name"] == "local/llama3"
    assert body["litellm_params"]["model"] == "local/llama3"
    assert body["litellm_params"]["api_base"] == "http://provider.com"
    assert body["litellm_params"]["api_key"] == "test-provider-key"


def test_add_model_dry_run(master_key_file: pathlib.Path) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
        dry_run=True,
    )

    model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
    provider_config = ProviderConfig(
        base_url="http://provider.com",
        provider_type="local",
    )

    # Should not raise exception and should not make HTTP request
    client.add_model(model, provider_config)


@responses.activate
def test_add_model_no_master_key() -> None:
    client = LiteLLMClient(base_url="http://test.com")

    model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
    provider_config = ProviderConfig(
        base_url="http://provider.com",
        provider_type="local",
    )

    responses.add(
        responses.POST,
        "http://test.com/model/new",
        json={"status": "success"},
        status=200,
    )

    client.add_model(model, provider_config)
    assert len(responses.calls) == 1
    assert "Authorization" not in responses.calls[0].request.headers


def test_add_model_unreadable_provider_key(
    master_key_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    unreadable_key = tmp_path / "unreadable.key"
    unreadable_key.touch()

    # Mock open to raise OSError
    with mock.patch(
        "pathlib.Path.open", side_effect=OSError("Permission denied")
    ):
        model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
        provider_config = ProviderConfig(
            base_url="http://provider.com",
            provider_type="local",
            api_key_path=str(unreadable_key),
        )

        with pytest.raises(ProviderKeyReadError):
            client.add_model(model, provider_config)


@responses.activate
def test_add_model_http_error(master_key_file: pathlib.Path) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    responses.add(
        responses.POST,
        "http://test.com/model/new",
        status=500,
    )

    model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
    provider_config = ProviderConfig(
        base_url="http://provider.com",
        provider_type="local",
    )

    with pytest.raises(requests.exceptions.HTTPError):
        client.add_model(model, provider_config)


def test_client_init_empty_key(tmp_path: pathlib.Path) -> None:
    empty_key = tmp_path / "empty.key"
    empty_key.touch()

    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(empty_key),
    )
    assert client.master_key == ""


def test_client_init_unreadable_key(tmp_path: pathlib.Path) -> None:
    unreadable_key = tmp_path / "unreadable.key"
    unreadable_key.touch()

    with (
        mock.patch(
            "pathlib.Path.open", side_effect=OSError("Permission denied")
        ),
        pytest.raises(ProviderKeyReadError),
    ):
        LiteLLMClient(
            base_url="http://test.com",
            master_key_path=str(unreadable_key),
        )


@responses.activate
def test_add_model_empty_provider_key(
    master_key_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    client = LiteLLMClient(
        base_url="http://test.com",
        master_key_path=str(master_key_file),
    )

    empty_key = tmp_path / "empty.key"
    empty_key.touch()

    responses.add(
        responses.POST,
        "http://test.com/model/new",
        json={"status": "success"},
        status=200,
    )

    model = DiscoveredModel(id="llama3", name="Llama 3", provider="local")
    provider_config = ProviderConfig(
        base_url="http://provider.com",
        provider_type="local",
        api_key_path=str(empty_key),
    )

    client.add_model(model, provider_config)

    assert len(responses.calls) == 1
    req = responses.calls[0].request
    assert isinstance(req.body, (str, bytes, bytearray))

    body = json.loads(req.body)
    assert "api_key" not in body["litellm_params"]


@responses.activate
def test_get_inference_models_invalid_format_no_dict() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    responses.add(
        responses.GET,
        "http://test.com/v1/models",
        json=["not_a_dict"],
        status=200,
    )
    with pytest.raises(
        InvalidResponseFormatError, match="Unexpected response format"
    ):
        client.get_inference_models()


@responses.activate
def test_get_inference_models_invalid_format_no_list() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    responses.add(
        responses.GET,
        "http://test.com/v1/models",
        json={"data": "not_a_list"},
        status=200,
    )
    with pytest.raises(LiteLLMSyncError, match="Unexpected response format"):
        client.get_inference_models()


@responses.activate
def test_get_configured_models_invalid_format_no_dict() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    responses.add(
        responses.GET,
        "http://test.com/model/info",
        json=["not_a_dict"],
        status=200,
    )
    with pytest.raises(
        InvalidResponseFormatError, match="Unexpected response format"
    ):
        client.get_configured_models()


@responses.activate
def test_get_configured_models_invalid_format_no_list() -> None:
    client = LiteLLMClient(base_url="http://test.com")
    responses.add(
        responses.GET,
        "http://test.com/model/info",
        json={"data": "not_a_list"},
        status=200,
    )
    with pytest.raises(LiteLLMSyncError, match="Unexpected response format"):
        client.get_configured_models()
