from __future__ import annotations

import logging
import pathlib

import click


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

    # Placeholder for Phase 2-5 implementation
    logger.info("Phase 1 complete. Core CLI initialized.")


if __name__ == "__main__":
    cli()  # pragma: no cover
