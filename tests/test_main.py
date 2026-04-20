from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import responses
from click.testing import CliRunner

from pi_models_sync.__main__ import _resolve_path, cli

if TYPE_CHECKING:
    import pathlib


def test_resolve_path() -> None:
    """Test the _resolve_path helper function."""
    # Test with None
    assert _resolve_path(None) is None

    # Test with a regular path
    path = Path("/tmp/test.txt")
    assert _resolve_path(path) == path

    # Test with ~ (home directory)
    home_path = Path("~")
    resolved = _resolve_path(home_path)
    assert resolved == Path.home()


@pytest.fixture
def mock_keys(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to set up mock keys and chdir."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "litellm_master.key").write_text("sk-master-key")
    (tmp_path / "litellm_inference.key").write_text("sk-inference-key")
    (tmp_path / "cloud_ollama.key").write_text("sk-cloud-ollama-key")


@responses.activate
def test_cli_defaults(mock_keys: None, tmp_path: pathlib.Path) -> None:
    # Mock LiteLLM config fetch
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": [{"model_name": "local-ollama/existing-model"}]},
        status=200,
    )

    # Mock local Ollama fetch
    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": [{"name": "new-local-model"}]},
        status=200,
    )

    # Mock LiteLLM add model
    responses.add(
        responses.POST,
        "http://localhost:4000/model/new",
        json={"status": "success"},
        status=200,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    result = runner.invoke(cli, ["--pi-models-path", str(pi_models_path)])
    assert result.exit_code == 0
    assert "Starting pi-models-sync" in result.output
    assert "LiteLLM URL: http://localhost:4000" in result.output
    assert "models.json" in result.output
    assert "Local Ollama URL: http://localhost:11434" in result.output
    assert "Dry Run: False" in result.output
    assert "Pi Only: False" in result.output

    assert result.exit_code == 0
    assert pi_models_path.exists()

    config = json.loads(pi_models_path.read_text(encoding="utf-8"))
    models = config["providers"]["litellm_gateway"]["models"]

    # We found 1 model from local ollama that wasn't in LiteLLM,
    # so it gets added. The models in the config will be the
    # ones discovered during sync.
    assert len(models) == 1
    assert models[0]["id"] == "local-ollama/new-local-model"


@responses.activate
def test_cli_with_args_and_cloud_ollama(
    mock_keys: None, tmp_path: pathlib.Path
) -> None:
    # Need to mock LiteLLM config fetch since pi_only is false
    responses.add(
        responses.GET,
        "http://example.com/litellm/model/info",
        json={"data": [{"model_name": "local-ollama/existing-model"}]},
        status=200,
    )

    # Need to mock local and cloud ollama fetches
    responses.add(
        responses.GET,
        "http://127.0.0.1:11434/api/tags",
        json={"models": [{"name": "new-local-model"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://cloud.ollama.ai/api/tags",
        json={"models": [{"name": "new-cloud-model"}]},
        status=200,
    )

    # Mock LiteLLM add model endpoint
    responses.add(
        responses.POST,
        "http://example.com/litellm/model/new",
        json={"status": "success"},
        status=200,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "custom_models.json"
    result = runner.invoke(
        cli,
        [
            "--litellm-url",
            "http://example.com/litellm",
            "--pi-models-path",
            str(pi_models_path),
            "--cloud-ollama-url",
            "https://cloud.ollama.ai",
            "--local-ollama-url",
            "http://127.0.0.1:11434",
        ],
    )
    assert result.exit_code == 0, f"{result.output} {result.exception}"
    assert "LiteLLM URL: http://example.com/litellm" in result.output
    assert f"Output Path: {pi_models_path}" in result.output
    assert "Cloud Ollama URL: https://cloud.ollama.ai" in result.output
    assert "Local Ollama URL: http://127.0.0.1:11434" in result.output
    assert "Dry Run: False" in result.output
    assert "Pi Only: False" in result.output


@responses.activate
def test_dry_run_mode(mock_keys: None, tmp_path: pathlib.Path) -> None:
    """Test that --dry-run does not write the file or modify LiteLLM."""
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": [{"model_name": "local-ollama/existing-model"}]},
        status=200,
    )

    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": [{"name": "new-local-model"}]},
        status=200,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    result = runner.invoke(
        cli, ["--pi-models-path", str(pi_models_path), "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Dry run complete." in result.output

    # Add_model shouldn't be making a POST in dry-run, but checking responses:
    for call in responses.calls:
        if call.request.url:
            assert "model/new" not in call.request.url

    # The models.json should not have been created
    assert not pi_models_path.exists()


@responses.activate
def test_cli_exceptions(mock_keys: None, tmp_path: pathlib.Path) -> None:
    """Test exceptions trigger click.Abort."""
    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    # 1. Pi Only fetch failure
    responses.add(
        responses.GET,
        "http://localhost:4000/v1/models",
        status=500,
    )
    result = runner.invoke(
        cli, ["--pi-models-path", str(pi_models_path), "--pi-only"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)

    # 2. Configured models fetch failure
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        status=500,
    )
    result = runner.invoke(cli, ["--pi-models-path", str(pi_models_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)

    # 3. Pi config write failure
    responses.replace(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": []},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": []},
        status=200,
    )

    # Give an invalid path to cause an OSError
    result = runner.invoke(
        cli,
        [
            "--pi-models-path",
            "/invalid/dir/that/does/not/exist/models.json",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)


@responses.activate
def test_cli_add_model_exception(
    mock_keys: None, tmp_path: pathlib.Path
) -> None:
    """Test exceptions when adding models don't crash the script."""
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": [{"model_name": "local-ollama/existing-model"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": [{"name": "failing-model"}]},
        status=200,
    )
    # The POST will fail
    responses.add(
        responses.POST,
        "http://localhost:4000/model/new",
        status=500,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    result = runner.invoke(cli, ["--pi-models-path", str(pi_models_path)])
    assert result.exit_code == 0

    # Despite the exception, config should still be generated
    # but just empty of new models
    config = json.loads(pi_models_path.read_text(encoding="utf-8"))
    models = config["providers"]["litellm_gateway"]["models"]
    assert len(models) == 0


@responses.activate
def test_cli_model_already_exists(
    mock_keys: None, tmp_path: pathlib.Path
) -> None:
    """Test that existing models are correctly skipped during add_model."""
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": [{"model_name": "local-ollama/existing-model"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": [{"name": "existing-model"}]},
        status=200,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    result = runner.invoke(cli, ["--pi-models-path", str(pi_models_path)])
    assert result.exit_code == 0

    # Check responses to ensure no POST /model/new was called
    for call in responses.calls:
        if call.request.url:
            assert "model/new" not in call.request.url

    config = json.loads(pi_models_path.read_text(encoding="utf-8"))
    models = config["providers"]["litellm_gateway"]["models"]
    assert len(models) == 1
    assert models[0]["id"] == "local-ollama/existing-model"


@responses.activate
def test_pi_only_mode(mock_keys: None, tmp_path: pathlib.Path) -> None:
    """Test that --pi-only skips discovery and only uses /v1/models."""
    responses.add(
        responses.GET,
        "http://localhost:4000/v1/models",
        json={"data": [{"id": "litellm-model-a"}, {"id": "litellm-model-b"}]},
        status=200,
    )

    runner = CliRunner()
    pi_models_path = tmp_path / "models.json"

    result = runner.invoke(
        cli, ["--pi-models-path", str(pi_models_path), "--pi-only"]
    )
    assert result.exit_code == 0
    assert (
        "Running in Pi Only mode. Skipping discovery and sync." in result.output
    )

    # Ensure no ollama tags or litellm model/info were called
    for call in responses.calls:
        if call.request.url:
            assert "api/tags" not in call.request.url
            assert "model/info" not in call.request.url
            assert "model/new" not in call.request.url

    config = json.loads(pi_models_path.read_text(encoding="utf-8"))
    models = config["providers"]["litellm_gateway"]["models"]
    assert len(models) == 2
    assert models[0]["id"] == "litellm-model-a"
    assert models[1]["id"] == "litellm-model-b"
