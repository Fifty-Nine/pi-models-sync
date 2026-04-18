# Implementation Plan: LiteLLM Model Sync Script

## Objective
Implement a Python CLI tool using `click` that discovers models from Ollama instances, syncs them to a LiteLLM gateway, and generates a `models.json` file for the Pi coding agent. It must ensure older config files are backed up sequentially. Additionally, the tool will support a `--pi-only` mode to skip discovery and syncing, instead just querying the LiteLLM gateway's standard `/v1/models` endpoint (using an inference key) to generate the `models.json` file.

## Key Files & Context
- `sync-models.py`: Main CLI entry point.
- `providers.py`: Module for interacting with Local and Cloud Ollama instances.
- `litellm_client.py`: Module for interacting with the LiteLLM API (Management API for syncing, OpenAI-compatible `/v1/models` API for read-only mode).
- `pi_config.py`: Module handling the generation and sequential backup of `models.json`.
- Secrets: Read from `cloud_ollama.key`, `litellm_master.key`, and optionally `litellm_inference.key` in the local directory.

## Implementation Steps

### Phase 1: Core CLI & Data Models
1. Initialize the Python project structure.
2. Define basic data models using `dataclasses` or `pydantic` for `DiscoveredModel` and `ProviderConfig`.
3. Implement the `click` CLI interface with all required arguments (`--litellm-url`, `--pi-models-path`, `--cloud-ollama-url`, `--local-ollama-url`, `--dry-run`, `--pi-only`).
4. Set up logging to provide clear output during script execution.

### Phase 2: Providers Implementation
1. Create a base `ModelProvider` class in `providers.py`.
2. Implement `LocalOllamaProvider` using `requests` to fetch `/api/tags` from the local instance.
3. Implement `CloudOllamaProvider` using `requests` and reading the API key from `cloud_ollama.key`.
4. Define a standard normalization process so each provider yields instances of `DiscoveredModel`.

### Phase 3: LiteLLM Sync Client
1. Implement a `LiteLLMClient` in `litellm_client.py` using `requests`.
2. Implement an authentication method reading from `litellm_master.key` (for syncing) and `litellm_inference.key` (for read-only querying).
3. Create a method to fetch currently configured models from LiteLLM via the management API.
4. Create a method to register new models via LiteLLM's `/model/new` API (or relevant management endpoint), mapping them uniquely (e.g., `local-ollama/llama3.1:8b`).
5. Create a method to query the standard `/v1/models` endpoint using the inference key, returning models available for inference (used in `--pi-only` mode).
6. Add dry-run support to simulate registration without side effects.

### Phase 4: Pi Config Generator & Backup
1. Implement logic in `pi_config.py` to map the synced LiteLLM models to the Pi schema format (a single `litellm` provider block).
2. Implement the sequential backup function for an existing `models.json` (moving it to `models.json.1`, `models.json.2`, etc., avoiding overwrites).
3. Write the new JSON configuration to the specified output path (default: `~/.pi/agent/models.json` or current directory).

### Phase 5: Integration & Verification
1. Wire the phases together in the main `click` function in `sync-models.py`.
2. Implement conditional logic: if `--pi-only` is passed, skip Phase 2 and Phase 3 (steps 3/4), and instead execute Phase 3 (step 5) to gather models before running Phase 4.
3. Implement end-to-end testing with mock endpoints to verify:
   - Providers correctly yield models.
   - LiteLLM models are synced correctly without duplicates.
   - `--pi-only` correctly bypasses sync and reads `/v1/models`.
   - Pi `models.json` generation handles backups and outputs correct schema.
4. Add a `--dry-run` flag validation across all modules.

### Phase 6: Post-Implementation Enhancements
1. Extend the `DiscoveredModel` data model and LiteLLM sync logic to capture and populate additional metadata, such as:
   - `contextWindow` (maximum context window size)
   - `maxTokens` (maximum output tokens)
   - Other relevant fields from the pi `models-schema.json`.

## Verification & Testing
- Unit tests for the sequential backup logic to ensure `models.json.X` numbers increment correctly.
- Mocks for `requests` to verify payload formatting for LiteLLM.
- Manual execution of the script against a test LiteLLM instance.

## Infrastructure
- Use Python-standard pyproject.toml as the central location for project
  metadata, configuration, etc.
- Use `uv` for dependency management.
- Use `ruff` for linting and ensure aggressive checking.
- Use `pytest` for unit testing.
