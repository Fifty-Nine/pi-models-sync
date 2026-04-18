from __future__ import annotations

from pi_models_sync.discovery import DiscoveredModel


def test_discovered_model() -> None:
    model = DiscoveredModel(
        id="llama3", name="Llama 3", provider="local-ollama"
    )
    assert model.id == "llama3"
    assert model.name == "Llama 3"
    assert model.provider == "local-ollama"
