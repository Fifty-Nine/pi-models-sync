from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from pi_models_sync.litellm_client import LiteLLMClient
    from pi_models_sync.providers.base import ModelProvider
    from pi_models_sync.providers.config import ProviderConfig


# Configure basic logging
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--litellm-url",
    default="http://localhost:4000",
    help="URL for the LiteLLM Management API.",
)
@click.option(
    "--pi-models-path",
    type=click.Path(resolve_path=True, path_type=pathlib.Path),
    default="~/.pi/agent/models.json",
    help="Output path for the generated models.json.",
)
@click.option(
    "--cloud-ollama-url",
    default=None,
    help="URL for the Cloud Ollama instance.",
)
@click.option(
    "--local-ollama-url",
    default="http://localhost:11434",
    help="URL for the Local Ollama instance.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without modifying anything.",
)
@click.option(
    "--pi-only",
    is_flag=True,
    help="Skip discovery and syncing, just generate models.json.",
)
def cli(
    *,
    litellm_url: str,
    pi_models_path: pathlib.Path,
    cloud_ollama_url: str | None,
    local_ollama_url: str,
    dry_run: bool,
    pi_only: bool,
) -> None:
    """Sync models to LiteLLM, then generate Pi models.json."""
    setup_logging()
    logger.info("Starting pi-models-sync")
    logger.info("Configuration:")
    logger.info("  LiteLLM URL: %s", litellm_url)
    logger.info("  Output Path: %s", pi_models_path)
    if cloud_ollama_url:
        logger.info("  Cloud Ollama URL: %s", cloud_ollama_url)
    logger.info("  Local Ollama URL: %s", local_ollama_url)
    logger.info("  Dry Run: %s", dry_run)
    logger.info("  Pi Only: %s", pi_only)

    from pi_models_sync.discovery import DiscoveredModel  # noqa: PLC0415
    from pi_models_sync.litellm_client import LiteLLMClient  # noqa: PLC0415
    from pi_models_sync.pi_config import generate_pi_config  # noqa: PLC0415
    from pi_models_sync.providers.config import ProviderConfig  # noqa: PLC0415
    from pi_models_sync.providers.ollama import (  # noqa: PLC0415
        CloudOllamaProvider,
        LocalOllamaProvider,
    )

    client = LiteLLMClient(
        base_url=litellm_url,
        master_key_path="litellm_master.key",
        inference_key_path="litellm_inference.key",
        dry_run=dry_run,
    )

    if pi_only:
        logger.info("Running in pi-only mode, bypassing discovery and sync")
        model_ids = client.get_inference_models()
        models = [
            DiscoveredModel(id=m_id, provider="litellm") for m_id in model_ids
        ]
        if not dry_run:
            api_key = client.inference_key or client.master_key or ""
            generate_pi_config(models, litellm_url, api_key, pi_models_path)
            logger.info(
                "Successfully generated models.json at %s", pi_models_path
            )
    else:
        logger.info("Discovering and syncing models")
        configured_models = set(client.get_configured_models())

        providers: list[tuple[ModelProvider, ProviderConfig]] = []
        local_config = ProviderConfig(
            base_url=local_ollama_url, provider_type="ollama"
        )
        providers.append((LocalOllamaProvider(local_config), local_config))

        if cloud_ollama_url:
            cloud_config = ProviderConfig(
                base_url=cloud_ollama_url,
                provider_type="openai",
                api_key_path="cloud_ollama.key",
            )
            providers.append((CloudOllamaProvider(cloud_config), cloud_config))

        for provider, config in providers:
            _sync_provider_models(provider, config, client, configured_models)

        model_ids = client.get_inference_models()
        models = [
            DiscoveredModel(id=m_id, provider="litellm") for m_id in model_ids
        ]
        if not dry_run:
            api_key = client.inference_key or client.master_key or ""
            generate_pi_config(models, litellm_url, api_key, pi_models_path)
            logger.info(
                "Successfully generated models.json at %s", pi_models_path
            )


def _sync_provider_models(
    provider: ModelProvider,
    config: ProviderConfig,
    client: LiteLLMClient,
    configured_models: set[str],
) -> None:
    """Helper to sync models from a specific provider."""
    for model in provider.get_models():
        model_key = f"{model.provider}/{model.id}"
        if model_key not in configured_models:
            logger.info("Adding new model: %s", model_key)
            client.add_model(model, config)
        else:
            logger.info("Model already configured: %s", model_key)


if __name__ == "__main__":
    cli()  # pragma: no cover
