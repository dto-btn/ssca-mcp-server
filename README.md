# SSCA MCP Server

A production-ready MCP orchestrator that classifies user intent and recommends the best downstream MCP server for execution.

This service is standalone and does not depend on categoryService.info.

## Why This Project

- Improves tool routing quality by combining deterministic keyword scoring with optional LLM-first classification.
- Reduces failed tool calls with confidence thresholds, fallbacks, and clarifying-question hints.
- Provides both MCP and HTTP surfaces for chat clients and playground integrations.
- Supports safe runtime operations with registry validation, hot reload, and optional admin-gated updates.

## Key Capabilities

- LLM-first intent classification through LiteLLM proxy with deterministic keyword fallback.
- Multi-category routing support with tie handling and route ranking.
- Built-in fallback behavior when intent confidence is low.
- Registry-backed server metadata, aliases, and routing rules.
- Streamable MCP HTTP app with CORS support.
- Optional HTTP helper endpoint for pre-routing workflows.

## Architecture Overview

- Classification and routing engine: src/server/classifier.py, src/server/router.py
- Registry and schema handling: src/server/registry.py, src/server/schemas.py
- Runtime settings: src/server/config.py
- Server entrypoint (MCP + HTTP): src/server/server.py

High-level flow:

1. Client sends message history.
2. Service classifies intent (LLM-first when enabled, otherwise keyword).
3. Service ranks downstream MCP server candidates.
4. Service returns recommendations or safe fallback guidance.

## Quick Start

Requirements:

- Python 3.12+
- uv

Install dependencies:

```bash
uv sync --all-groups
```

Copy environment template:

```bash
cp .env.example .env
```

Run in MCP dev mode:

```bash
uv run mcp dev src/server/server.py
```

Run as HTTP server:

```bash
uv run uvicorn src.server.server:app --host 0.0.0.0 --port 8000
```

## Endpoints And Transports

- MCP streamable HTTP transport: /mcp
- HTTP routing helper endpoint: POST /orchestrator/suggest-route

Example health-style check:

```bash
curl -i http://localhost:8000/mcp
```

Example routing helper call:

```bash
curl -s http://localhost:8000/orchestrator/suggest-route \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Find SQL schema info for customer table"}],
    "max_recommendations": 3,
    "require_single_best": false
  }'
```

## Docker

Build image:

```bash
docker build -t ssca-orchestrator:local .
```

Run container:

```bash
docker run --rm -p 9000:9000 \
  --env-file .env \
  -e ORCHESTRATOR_HOST=0.0.0.0 \
  -e ORCHESTRATOR_PORT=9000 \
  ssca-orchestrator:local
```

Notes:

- Container command honors PORT when provided by hosting platforms.
- Ensure host binding is 0.0.0.0 and published port mapping is correct.

## Configuration

### Required For LLM Classification

| Variable | Description | Default |
|---|---|---|
| ENABLE_LLM_CLASSIFIER | Enables LLM-first classification | false |
| ORCHESTRATOR_LITELLM_PROXY_URL | LiteLLM/OpenAI-compatible base URL | http://localhost:4000/v1 |
| ORCHESTRATOR_LLM_MODEL | Model id exposed by LiteLLM | none |

Authentication for LiteLLM proxy:

- ORCHESTRATOR_LITELLM_PROXY_API_KEY
- or LITELLM_MASTER_KEY (fallback)
- optional ORCHESTRATOR_LITELLM_PROXY_BEARER_TOKEN

### Core Runtime Settings

| Variable | Description | Default |
|---|---|---|
| ORCHESTRATOR_REGISTRY_PATH | Registry file path | ./mcp_registry.json |
| ORCHESTRATOR_MAX_MESSAGES | Max message turns used for scoring | 10 |
| ORCHESTRATOR_MIN_CONFIDENCE | Confidence threshold for recommendations | 0.4 |
| ORCHESTRATOR_LLM_TIMEOUT_SECONDS | LLM call timeout | 8.0 |
| ORCHESTRATOR_MAX_MESSAGE_CHARS | Per-message character cap | 4000 |
| ORCHESTRATOR_MAX_TOTAL_CHARS | Total context character cap | 20000 |
| ORCHESTRATOR_ENABLE_HOT_RELOAD | Reload registry when file changes | false |
| ORCHESTRATOR_ENABLE_UPDATE_REGISTRY | Enables update_registry tool | false |
| ORCHESTRATOR_ADMIN_SECRET | Shared secret for update_registry tool | unset |
| VERBOSE_LOGGING | Enables verbose logs | false |
| ORCHESTRATOR_REDACT_SENSITIVE | Redacts sensitive tokens in logs | true |

### HTTP And Local TLS Settings

| Variable | Description | Default |
|---|---|---|
| ORCHESTRATOR_ALLOWED_ORIGINS | CORS allow list (comma-separated) | localhost dev origins |
| ORCHESTRATOR_HOST | HTTP bind host | 0.0.0.0 |
| ORCHESTRATOR_PORT | HTTP bind port | 8000 |
| ORCHESTRATOR_GRACEFUL_SHUTDOWN_SECONDS | Uvicorn graceful shutdown timeout | 2 |
| ORCHESTRATOR_TLS_CERT_FILE | Local TLS cert file | unset |
| ORCHESTRATOR_TLS_KEY_FILE | Local TLS key file | unset |
| ORCHESTRATOR_AUTO_TLS | Auto-load certs/localhost.* in script | false |

## MCP Tool Contracts

### classify_context

Input:

```json
{
  "messages": [{ "role": "user", "content": "..." }],
  "locale": "en-CA",
  "metadata": { "request_id": "abc" }
}
```

Output includes categories, confidence, explanation, classification_method, and timestamp.

### suggest_route

Input:

```json
{
  "messages": [{ "role": "user", "content": "..." }],
  "max_recommendations": 3,
  "require_single_best": false
}
```

Output includes ranked recommendations, optional fallback block, optional disambiguation hints, and timestamp.

### route_and_forward

- Phase-1 stub that returns route decision plus a mock forwarding envelope.

### update_registry

- Disabled by default.
- Enabled with ORCHESTRATOR_ENABLE_UPDATE_REGISTRY=true.
- If ORCHESTRATOR_ADMIN_SECRET is set, caller must provide matching admin_secret.

## Registry Model

Registry path resolution:

- ORCHESTRATOR_REGISTRY_PATH
- fallback: ./mcp_registry.json

The registry defines:

- mcp_servers: endpoint, categories, tools, keywords, weight
- category_aliases: alternate terms that map to canonical categories
- routing_rules: max_recommendations, tie_breaker, default_fallback

If the file does not exist, the service creates a valid default scaffold.

## Production Readiness Checklist

- Set explicit CORS origins (avoid permissive defaults).
- Use a strong API key or bearer token between orchestrator and LiteLLM proxy.
- Keep ORCHESTRATOR_ENABLE_UPDATE_REGISTRY disabled unless operationally required.
- If update_registry is enabled, set ORCHESTRATOR_ADMIN_SECRET.
- Run with immutable image tags and pinned lockfile updates.
- Monitor logs for fallback and low-confidence routing patterns.
- Add CI checks for tests and linting before release.

## Local HTTPS For Browser Testing

Generate certs:

```bash
./scripts/generate-local-self-signed-cert.sh
```

Start with helper script:

```bash
./scripts/start-playground-orchestrator.sh
```

Example overrides:

```bash
ORCHESTRATOR_PORT=8010 ORCHESTRATOR_ALLOWED_ORIGINS="https://localhost:8080" ./scripts/start-playground-orchestrator.sh
```

## Testing

Run tests:

```bash
uv run pytest -q
```

Current suite covers:

- intent scoring and confidence bounds
- alias resolution
- tie and ambiguity handling
- fallback behavior
- registry validation and hot reload
- LLM classifier plugin behavior

## Development Notes

- Primary test file: tests/test_orchestrator.py
- Lint/editor settings: .vscode/settings.json
- Dependency lock updates: uv lock

## Extending The Router

To add a new downstream MCP server, update mcp_registry.json with:

- id
- endpoint
- categories
- tools
- keywords
- weight

Then validate behavior with targeted tests in tests/test_orchestrator.py.

## References

- Model Context Protocol Python SDK: https://github.com/modelcontextprotocol/python-sdk
