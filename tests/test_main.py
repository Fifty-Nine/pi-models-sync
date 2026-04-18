from __future__ import annotations

from click.testing import CliRunner

from pi_models_sync.__main__ import cli


def test_cli_defaults() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Starting pi-models-sync" in result.output
    assert "LiteLLM URL: http://localhost:4000" in result.output
    assert "models.json" in result.output
    assert "Local Ollama URL: http://localhost:11434" in result.output
    assert "Dry Run: False" in result.output
    assert "Pi Only: False" in result.output
    assert "Phase 1 complete. Core CLI initialized." in result.output


def test_cli_with_args() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--litellm-url",
            "http://example.com/litellm",
            "--pi-models-path",
            "/tmp/models.json",
            "--cloud-ollama-url",
            "https://cloud.ollama.ai",
            "--local-ollama-url",
            "http://127.0.0.1:11434",
            "--dry-run",
            "--pi-only",
        ],
    )
    assert result.exit_code == 0
    assert "LiteLLM URL: http://example.com/litellm" in result.output
    assert "Output Path: /tmp/models.json" in result.output
    assert "Cloud Ollama URL: https://cloud.ollama.ai" in result.output
    assert "Local Ollama URL: http://127.0.0.1:11434" in result.output
    assert "Dry Run: True" in result.output
    assert "Pi Only: True" in result.output
