from __future__ import annotations

import json
import pathlib

import responses
from click.testing import CliRunner

from pi_models_sync.__main__ import cli


@responses.activate
def test_cli_defaults(tmp_path: pathlib.Path) -> None:
    # Setup mocks for default behavior (pi_only=False, dry_run=False)
    responses.add(
        responses.GET,
        "http://localhost:4000/model/info",
        json={"data": [{"model_name": "local-ollama/llama3:8b"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://localhost:11434/api/tags",
        json={"models": [{"name": "llama3:8b"}, {"name": "mixtral"}]},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://localhost:4000/model/new",
        json={"data": {}},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://localhost:4000/v1/models",
        json={"data": [{"id": "llama3:8b"}, {"id": "mixtral"}]},
        status=200,
    )

    models_path = tmp_path / "models.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--pi-models-path",
            str(models_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "Starting pi-models-sync" in result.output
    assert "LiteLLM URL: http://localhost:4000" in result.output
    assert "models.json" in result.output
    assert "Local Ollama URL: http://localhost:11434" in result.output
    assert "Dry Run: False" in result.output
    assert "Pi Only: False" in result.output

    # Assert models.json was created correctly
    assert models_path.exists()
    config_data = json.loads(models_path.read_text(encoding="utf-8"))
    assert len(config_data["providers"]["litellm_gateway"]["models"]) == 2

    assert "Model already configured: local-ollama/llama3:8b" in result.output

    # Check POST /model/new was called for mixtral
    post_calls = [c for c in responses.calls if c.request.method == "POST"]
    assert len(post_calls) == 1
    req_body = post_calls[0].request.body
    assert isinstance(req_body, bytes)
    assert "mixtral" in req_body.decode("utf-8")


@responses.activate
def test_cli_with_args_and_dry_run(tmp_path: pathlib.Path) -> None:
    # Set up files
    cloud_key = tmp_path / "cloud_ollama.key"
    cloud_key.write_text("test-cloud-key")

    # Mocks
    responses.add(
        responses.GET,
        "http://example.com/litellm/model/info",
        json={"data": []},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://127.0.0.1:11434/api/tags",
        json={"models": [{"name": "llama3"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://cloud.ollama.ai/api/tags",
        json={"models": [{"name": "cloud-model"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "http://example.com/litellm/v1/models",
        json={"data": []},
        status=200,
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create dummy key files in the isolated filesystem dir
        pathlib.Path("litellm_master.key").write_text("dummy")
        pathlib.Path("litellm_inference.key").write_text("dummy")
        pathlib.Path("cloud_ollama.key").write_text("dummy")

        result = runner.invoke(
            cli,
            [
                "--litellm-url",
                "http://example.com/litellm",
                "--pi-models-path",
                "models.json",
                "--cloud-ollama-url",
                "https://cloud.ollama.ai",
                "--local-ollama-url",
                "http://127.0.0.1:11434",
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, result.exception
    assert "Dry Run: True" in result.output
    assert "Would add model local-ollama/llama3" in result.output
    assert "Would add model cloud-ollama/cloud-model" in result.output

    # Assert models.json was not created
    assert not (tmp_path / "models.json").exists()


@responses.activate
def test_cli_pi_only(tmp_path: pathlib.Path) -> None:
    responses.add(
        responses.GET,
        "http://localhost:4000/v1/models",
        json={"data": [{"id": "model_a"}, {"id": "model_b"}]},
        status=200,
    )

    models_path = tmp_path / "models.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--pi-models-path",
            str(models_path),
            "--pi-only",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert (
        "Running in pi-only mode, bypassing discovery and sync" in result.output
    )
    assert models_path.exists()
    config_data = json.loads(models_path.read_text(encoding="utf-8"))
    assert len(config_data["providers"]["litellm_gateway"]["models"]) == 2
