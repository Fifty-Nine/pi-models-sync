from __future__ import annotations

from pi_models_sync.providers.config import ProviderConfig


def test_provider_config() -> None:
    config = ProviderConfig(base_url="http://localhost", provider_type="local")
    assert config.base_url == "http://localhost"
    assert config.provider_type == "local"
    assert config.api_key_path is None

    config_with_key = ProviderConfig(
        base_url="https://cloud", provider_type="cloud", api_key_path="key.txt"
    )
    assert config_with_key.api_key_path == "key.txt"
