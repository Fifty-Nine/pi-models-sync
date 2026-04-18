# LiteLLM Model Sync Script Specification

## 1. Overview
A Python CLI tool to discover models from various LLM providers (initially Local Ollama and Cloud Ollama) and automatically synchronize them with a running LiteLLM Gateway via its Management API. Additionally, the tool generates a `models.json` configuration file for the Pi coding agent, routing all traffic through the LiteLLM gateway using its OpenAI-compatible interface.

## 2. Architecture & Flow
1. **Discovery Phase**: Query configured providers to fetch available models.
2. **LiteLLM Sync Phase**: Use LiteLLM's Management API to register newly discovered models. Existing models are left intact or updated.
3. **Pi Config Generation Phase**: Generate a `models.json` file conforming to the Pi agent schema, creating a single provider block (for the LiteLLM gateway) containing all the synced models.

## 3. Provider Configuration & Extensibility
The script will use a modular architecture to easily support future providers.
- **Base Interface**: A `Provider` class with a `get_models()` method.
- **Initial Implementations**:
  - `LocalOllamaProvider`: Queries a local Ollama instance (default: `http://localhost:11434`).
  - `CloudOllamaProvider`: Queries a remote Ollama instance.
- **Secrets Management**: API keys are read from individual plain text files in the script's directory (e.g., `cloud_ollama.key`, `litellm_master.key`).

## 4. LiteLLM Integration
- **Endpoint**: Connects to the LiteLLM management URL (e.g., `http://localhost:4000`).
- **Authentication**: Uses the master key from `litellm_master.key` as a Bearer token.
- **Model Naming/Routing**:
  - Since multiple providers might host the same model name (e.g., `llama3`), LiteLLM model names will use a prefix to ensure uniqueness (e.g., `local-ollama/llama3:8b`, `cloud-ollama/llama3:8b`).
  - The script will call LiteLLM's `/model/new` (or equivalent management endpoint) to add the model mapping.

## 5. Pi Agent `models.json` Generation
The script generates a valid `models.json` file for the Pi agent.
- **Provider Block**: A single `litellm` provider is created.
- **API Type**: `openai-completions`
- **Models Array**: Populated with the prefixed model names configured in LiteLLM.
- **Output**: Writes to the current directory or directly to `~/.pi/agent/models.json`.
- **Backup Strategy**: Before overwriting an existing `models.json`, the script must back up the current file to `models.json.[number]` (e.g., `models.json.1`, `models.json.2`), ensuring no previous backups are overwritten.

*Example Output Snippet:*
```json
{
  "providers": {
    "litellm_gateway": {
      "baseUrl": "http://localhost:4000/v1",
      "api": "openai-completions",
      "apiKey": "sk-litellm-key",
      "models": [
        { "id": "local-ollama/llama3.1:8b", "name": "Llama 3.1 8B (Local)" },
        { "id": "cloud-ollama/qwen2.5:7b", "name": "Qwen 2.5 7B (Cloud)" }
      ]
    }
  }
}
```

## 6. CLI Interface
Built with `click`.
**Arguments & Options:**
- `--litellm-url`: URL for the LiteLLM Management API (default: `http://localhost:4000`).
- `--pi-models-path`: Output path for the generated `models.json` (default: `./models.json`).
- `--cloud-ollama-url`: URL for the Cloud Ollama instance.
- `--local-ollama-url`: URL for the Local Ollama instance (default: `http://localhost:11434`).
- `--dry-run`: Output what would be done without modifying LiteLLM or writing the file.
