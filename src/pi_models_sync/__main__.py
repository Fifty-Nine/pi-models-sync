from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from pi_models_sync.discovery import DiscoveredModel
from pi_models_sync.litellm_client import LiteLLMClient
from pi_models_sync.pi_config import generate_pi_config
from pi_models_sync.providers.config import ProviderConfig
from pi_models_sync.providers.ollama import (
    CloudOllamaProvider,
    LocalOllamaProvider,
)

if TYPE_CHECKING:
    from pi_models_sync.providers.base import ModelProvider


# Configure basic logging
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


logger = logging.getLogger(__name__)


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    return path.expanduser()


@click.command()
@click.option(
    "--litellm-url",
    default="http://localhost:4000",
    help="URL for the LiteLLM Management API.",
)
@click.option(
    "--litellm-master-key",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the LiteLLM master key file (default: litellm_master.key).",
)
@click.option(
    "--litellm-inference-key",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the LiteLLM inference key file "
    "(default: litellm_inference.key).",
)
@click.option(
    "--pi-models-path",
    type=click.Path(resolve_path=True, path_type=Path),
    default="~/.pi/agent/models.json",
    help="Output path for the generated models.json.",
)
@click.option(
    "--cloud-ollama-url",
    default=None,
    help="URL for the Cloud Ollama instance.",
)
@click.option(
    "--cloud-ollama-key",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the Cloud Ollama API key file (default: cloud_ollama.key).",
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
    litellm_master_key: Path | None,
    litellm_inference_key: Path | None,
    pi_models_path: Path,
    cloud_ollama_url: str | None,
    cloud_ollama_key: Path | None,
    local_ollama_url: str,
    dry_run: bool,
    pi_only: bool,
) -> None:
    """Sync models to LiteLLM, then generate Pi models.json."""
    setup_logging()
    logger.info("Starting pi-models-sync")

    # Resolve keys from defaults if not provided
    litellm_master_key = _resolve_path(litellm_master_key)
    litellm_inference_key = _resolve_path(litellm_inference_key)
    cloud_ollama_key = _resolve_path(cloud_ollama_key)
    pi_models_path = pi_models_path.expanduser()

    logger.info("Configuration:")
    logger.info("  LiteLLM URL: %s", litellm_url)
    logger.info("  LiteLLM Master Key: %s", litellm_master_key)
    logger.info("  LiteLLM Inference Key: %s", litellm_inference_key)
    logger.info("  Output Path: %s", pi_models_path)
    if cloud_ollama_url:
        logger.info("  Cloud Ollama URL: %s", cloud_ollama_url)
        logger.info("  Cloud Ollama Key: %s", cloud_ollama_key)
    logger.info("  Local Ollama URL: %s", local_ollama_url)
    logger.info("  Dry Run: %s", dry_run)
    logger.info("  Pi Only: %s", pi_only)

    client = LiteLLMClient(
        base_url=litellm_url,
        master_key_path=str(litellm_master_key) if litellm_master_key else None,
        inference_key_path=(
            str(litellm_inference_key) if litellm_inference_key else None
        ),
        dry_run=dry_run,
    )

    models: list[DiscoveredModel] = []

    if pi_only:
        _run_pi_only(client, models)
    else:
        _run_discovery_and_sync(
            client,
            models,
            local_ollama_url,
            cloud_ollama_url,
            cloud_ollama_key,
        )

    api_key = client.inference_key or client.master_key or ""

    if dry_run:
        logger.info(
            "Dry run complete. Would generate config at %s", pi_models_path
        )
        logger.info("Models discovered: %s", [m.id for m in models])
    else:
        logger.info("Generating Pi configuration at %s", pi_models_path)
        try:
            generate_pi_config(
                models=models,
                litellm_url=litellm_url,
                api_key=api_key,
                output_path=pi_models_path,
            )
        except Exception as e:
            logger.exception("Failed to generate Pi configuration:")
            raise click.Abort from e
        logger.info("Pi configuration successfully generated.")


def _run_pi_only(client: LiteLLMClient, models: list[DiscoveredModel]) -> None:
    logger.info("Running in Pi Only mode. Skipping discovery and sync.")
    try:
        inference_models = client.get_inference_models()
    except Exception as e:
        logger.exception("Failed to fetch inference models from LiteLLM:")
        raise click.Abort from e
    models.extend(
        DiscoveredModel(
            id=model_id,
            name=model_id,
            provider="litellm",
        )
        for model_id in inference_models
    )


def _run_discovery_and_sync(
    client: LiteLLMClient,
    models: list[DiscoveredModel],
    local_ollama_url: str,
    cloud_ollama_url: str | None,
    cloud_ollama_key: Path | None,
) -> None:
    logger.info("Running discovery and sync.")
    try:
        configured_models = client.get_configured_models()
    except Exception as e:
        logger.exception("Failed to fetch configured models from LiteLLM:")
        raise click.Abort from e

    providers: list[ModelProvider] = []
    local_config = ProviderConfig(
        base_url=local_ollama_url,
        provider_type="ollama_chat",
    )
    providers.append(LocalOllamaProvider(local_config))

    if cloud_ollama_url:
        cloud_config = ProviderConfig(
            base_url=cloud_ollama_url,
            provider_type="ollama_chat",
            api_key_path=str(cloud_ollama_key) if cloud_ollama_key else None,
        )
        providers.append(CloudOllamaProvider(cloud_config))

    for provider in providers:
        _process_provider(provider, client, configured_models, models)


def _process_provider(
    provider: ModelProvider,
    client: LiteLLMClient,
    configured_models: list[str],
    models: list[DiscoveredModel],
) -> None:
    logger.info("Discovering models from %s", provider.provider_name)
    for model in provider.get_models():
        _process_model(model, provider, client, configured_models, models)


def _process_model(
    model: DiscoveredModel,
    provider: ModelProvider,
    client: LiteLLMClient,
    configured_models: list[str],
    models: list[DiscoveredModel],
) -> None:
    model_name_in_litellm = f"{model.provider}/{model.id}"
    if model_name_in_litellm not in configured_models:
        logger.info(
            "Model %s not in LiteLLM. Adding...",
            model_name_in_litellm,
        )
        try:
            client.add_model(model, provider.config)
        except Exception:
            logger.exception(
                "Failed to add model %s to LiteLLM:",
                model_name_in_litellm,
            )
            return
        configured_models.append(model_name_in_litellm)
    else:
        logger.info("Model %s already in LiteLLM.", model_name_in_litellm)

    # Use the mapped model name for the config
    models.append(
        DiscoveredModel(
            id=model_name_in_litellm,
            name=model_name_in_litellm,
            provider=model.provider,
            context_window=model.context_window,
            max_tokens=model.max_tokens,
        )
    )


if __name__ == "__main__":
    cli()  # pragma: no cover
