from __future__ import annotations

import json
import shutil
from pathlib import Path as PathLib
from typing import TYPE_CHECKING, Any

import pytest

from pi_models_sync.discovery import DiscoveredModel
from pi_models_sync.pi_config import (
    PiConfigError,
    backup_existing_config,
    generate_pi_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_backup_existing_config_no_file(tmp_path: Path) -> None:
    """Test backup_existing_config when the file does not exist."""
    file_path = tmp_path / "models.json"
    backup_existing_config(file_path)
    assert not file_path.exists()
    assert not (tmp_path / "models.json.1").exists()


def test_backup_existing_config_one_file(tmp_path: Path) -> None:
    """Test backup_existing_config when the file exists."""
    file_path = tmp_path / "models.json"
    file_path.write_text("dummy")

    backup_existing_config(file_path)

    assert not file_path.exists()
    backup_path = tmp_path / "models.json.1"
    assert backup_path.exists()
    assert backup_path.read_text() == "dummy"


def test_backup_existing_config_multiple_files(tmp_path: Path) -> None:
    """Test backup_existing_config with multiple existing backups."""
    file_path = tmp_path / "models.json"
    file_path.write_text("latest")
    (tmp_path / "models.json.1").write_text("old1")
    (tmp_path / "models.json.2").write_text("old2")

    backup_existing_config(file_path)

    assert not file_path.exists()
    assert (tmp_path / "models.json.1").read_text() == "old1"
    assert (tmp_path / "models.json.2").read_text() == "old2"

    backup_path = tmp_path / "models.json.3"
    assert backup_path.exists()
    assert backup_path.read_text() == "latest"


def test_backup_existing_config_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test backup_existing_config when an OSError occurs."""
    file_path = tmp_path / "models.json"
    file_path.write_text("dummy")

    def mock_move(*args: Any, **kwargs: Any) -> None:
        error = OSError("Permission denied")
        raise error

    monkeypatch.setattr(shutil, "move", mock_move)

    with pytest.raises(PiConfigError) as exc_info:
        backup_existing_config(file_path)

    assert "Failed to backup or write config" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)


def test_generate_pi_config(tmp_path: Path) -> None:
    """Test generate_pi_config generates the correct JSON structure."""
    output_path = tmp_path / "models.json"
    models = [
        DiscoveredModel(id="model1", provider="litellm", name="Model 1"),
        DiscoveredModel(
            id="model2",
            provider="litellm",
            name=None,
            context_window=8192,
            max_tokens=4096,
        ),
    ]

    generate_pi_config(
        models=models,
        litellm_url="http://localhost:4000",
        api_key="sk-test",
        output_path=output_path,
    )

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))

    provider_config = data["providers"]["litellm_gateway"]
    assert provider_config["baseUrl"] == "http://localhost:4000/v1"
    assert provider_config["api"] == "openai-completions"
    assert provider_config["apiKey"] == "sk-test"

    models_data = provider_config["models"]
    assert len(models_data) == 2

    assert models_data[0] == {"id": "model1", "name": "Model 1"}
    assert models_data[1] == {
        "id": "model2",
        "contextWindow": 8192,
        "maxTokens": 4096,
    }


def test_generate_pi_config_with_existing_v1(tmp_path: Path) -> None:
    """Test generate_pi_config handling a url that already ends with /v1."""
    output_path = tmp_path / "models.json"
    models = [DiscoveredModel(id="test", provider="litellm")]

    generate_pi_config(
        models=models,
        litellm_url="http://localhost:4000/v1/",
        api_key="sk-test",
        output_path=output_path,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        data["providers"]["litellm_gateway"]["baseUrl"]
        == "http://localhost:4000/v1"
    )


def test_generate_pi_config_writes_backup(tmp_path: Path) -> None:
    """Test that generating config backs up any existing file."""
    output_path = tmp_path / "models.json"
    output_path.write_text("old config")

    generate_pi_config(
        models=[],
        litellm_url="http://localhost:4000",
        api_key="sk-test",
        output_path=output_path,
    )

    assert (tmp_path / "models.json.1").read_text() == "old config"

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["providers"]["litellm_gateway"]["models"] == []


def test_generate_pi_config_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test generate_pi_config when an OSError occurs during writing."""
    output_path = tmp_path / "models.json"

    def mock_open(*args: Any, **kwargs: Any) -> Any:
        error = OSError("Disk full")
        raise error

    monkeypatch.setattr(PathLib, "open", mock_open)

    with pytest.raises(PiConfigError) as exc_info:
        generate_pi_config(
            models=[],
            litellm_url="http://localhost:4000",
            api_key="sk-test",
            output_path=output_path,
        )

    assert "Failed to backup or write config" in str(exc_info.value)
    assert "Disk full" in str(exc_info.value)
