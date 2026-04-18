from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from click.testing import CliRunner

from pi_models_sync.__main__ import cli

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def test_cli_defaults(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Starting pi-models-sync" in caplog.text
    assert "LiteLLM URL: http://localhost:4000" in caplog.text
    assert "Output Path: ./models.json" in caplog.text
    assert "Local Ollama URL: http://localhost:11434" in caplog.text
    assert "Dry Run: False" in caplog.text
    assert "Pi Only: False" in caplog.text
    assert "Phase 1 complete. Core CLI initialized." in caplog.text


def test_cli_with_args(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
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
    assert "LiteLLM URL: http://example.com/litellm" in caplog.text
    assert "Output Path: /tmp/models.json" in caplog.text
    assert "Cloud Ollama URL: https://cloud.ollama.ai" in caplog.text
    assert "Local Ollama URL: http://127.0.0.1:11434" in caplog.text
    assert "Dry Run: True" in caplog.text
    assert "Pi Only: True" in caplog.text
