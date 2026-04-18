from __future__ import annotations

from pi_models_sync.discovery import DiscoveredModel


def test_discovered_model() -> None:
    model = DiscoveredModel(
        id="llama3", name="Llama 3", provider="local-ollama"
    )
    assert model.id == "llama3"
    assert model.name == "Llama 3"
    assert model.provider == "local-ollama"
    assert model.context_window is None
    assert model.max_tokens is None
    assert model.input_types is None
    assert model.reasoning is None

    model_full = DiscoveredModel(
        id="llama3:8b",
        name="llama3:8b",
        provider="local-ollama",
        context_window=8192,
        max_tokens=4096,
        input_types=["text"],
        reasoning=False,
    )
    assert model_full.id == "llama3:8b"
    assert model_full.context_window == 8192
    assert model_full.max_tokens == 4096
    assert model_full.input_types == ["text"]
    assert model_full.reasoning is False
