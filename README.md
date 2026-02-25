# Orchestrator MCP Server

An MCP orchestrator server that classifies chat context and recommends which downstream MCP server should handle the request.

It does **not** depend on `categoryService.info`.

## Features

- Keyword/regex-style context classification with recency, density, and weighting.
- Ranked route suggestions across registered MCP servers.
- Safe fallback with clarifying question suggestions.
- Extensible LLM-classifier plug-in interface (`ENABLE_LLM_CLASSIFIER=true`).
- MCP tools:
	- `classify_context`
	- `suggest_route`
	- `route_and_forward` (phase-1 stub)
	- `update_registry` (optional admin-only)
- Registry schema validation on load/update.
- Optional registry hot reload.

## Project Layout

- `src/server/server.py`: MCP server and tool endpoints.
- `src/server/config.py`: environment-driven settings.
- `src/server/schemas.py`: registry schema models.
- `src/server/registry.py`: registry load/save/update with locking.
- `src/server/classifier.py`: scoring/classification engine.
- `src/server/router.py`: route recommendation and fallback logic.
- `mcp_registry.json`: default registry.
- `tests/test_orchestrator.py`: unit/integration tests.

## Run

Install dependencies:

```bash
uv sync --all-groups
```

Run server in MCP inspector/dev mode:

```bash
uv run mcp dev src/server/server.py
```

Run server for browser clients (Playground) with CORS-enabled MCP streamable HTTP app:

```bash
ORCHESTRATOR_ALLOWED_ORIGINS="https://localhost:8080,https://127.0.0.1:8080" \
ORCHESTRATOR_TLS_CERT_FILE="./certs/localhost.crt" \
ORCHESTRATOR_TLS_KEY_FILE="./certs/localhost.key" \
uv run uvicorn src.server.server:app --host 127.0.0.1 --port 8000
```

This serves the MCP streamable HTTP transport (`/mcp`) with CORS headers so the frontend can call orchestrator tools directly.

Or use the one-command launcher script (recommended):

```bash
./scripts/start-playground-orchestrator.sh
```

Optional overrides:

```bash
ORCHESTRATOR_PORT=8010 ORCHESTRATOR_ALLOWED_ORIGINS="https://localhost:8080" ./scripts/start-playground-orchestrator.sh
```

### Local self-signed TLS (dev/test)

Generate a local self-signed cert and key:

```bash
./scripts/generate-local-self-signed-cert.sh
```

If `mkcert` is installed, the script will generate a locally trusted cert automatically.
If `mkcert` is not installed, it falls back to a plain self-signed cert that browsers may reject until trusted manually.

Then start the orchestrator over HTTPS:

```bash
./scripts/start-playground-orchestrator.sh
```

The launcher auto-uses `./certs/localhost.crt` and `./certs/localhost.key` when present.

If your browser blocks the cert, trust it in your local trust store for smoother MCP testing.

## MCP Tools Contract

### `classify_context`

Input:

```json
{
	"messages": [{ "role": "user", "content": "..." }],
	"locale": "en-CA",
	"metadata": { "request_id": "abc" }
}
```

Output:

```json
{
	"categories": [
		{ "name": "database", "confidence": 0.84, "matched_keywords": ["sql", "query"] }
	],
	"explanation": "Top category selected from keyword evidence.",
	"timestamp": "2026-02-24T00:00:00Z"
}
```

### `suggest_route`

Input:

```json
{
	"messages": [{ "role": "user", "content": "..." }],
	"max_recommendations": 3,
	"require_single_best": false
}
```

Output:

```json
{
	"recommendations": [
		{
			"mcp_server_id": "db_mcp",
			"endpoint": "https://db-mcp.example.com/mcp",
			"category": "database",
			"confidence": 0.88,
			"matched_keywords": ["sql", "table"],
			"rationale": "Matched keywords ..."
		}
	],
	"fallback": {
		"reason": "No clear match. Ask a clarifying question.",
		"suggestions_for_user": ["Are you trying to query a database or search the web?"]
	},
	"plan": null,
	"timestamp": "2026-02-24T00:00:00Z"
}
```

### `route_and_forward` (phase 1)

- Returns selected route plus a mock forwarding response.
- Provides a `plan` extension object for future chained workflows.

### `update_registry` (optional admin-only)

- Disabled by default.
- Enable via `ORCHESTRATOR_ENABLE_UPDATE_REGISTRY=true`.
- If `ORCHESTRATOR_ADMIN_SECRET` is set, `admin_secret` must match.

## Registry Schema

Path:
- `ORCHESTRATOR_REGISTRY_PATH` or fallback `./mcp_registry.json`

Example is provided in `mcp_registry.json` and follows:

```json
{
	"version": "1.0",
	"mcp_servers": [
		{
			"id": "web_search_mcp",
			"endpoint": "https://web-search-mcp.example.com/mcp",
			"categories": ["web-search", "news", "research"],
			"tools": ["search", "get_page"],
			"keywords": ["search", "google", "web", "news", "article", "research", "find online"],
			"weight": 1.0
		}
	],
	"category_aliases": { "web": "web-search" },
	"routing_rules": {
		"max_recommendations": 3,
		"tie_breaker": "weight_then_keyword_density",
		"default_fallback": {
			"category": "general",
			"message": "No clear match. Ask a clarifying question."
		}
	}
}
```

If missing, the file is auto-created with empty `mcp_servers` and sensible routing defaults.

## Configuration

- `ORCHESTRATOR_REGISTRY_PATH` (default: `./mcp_registry.json`)
- `ORCHESTRATOR_MAX_MESSAGES` (default: `10`)
- `ORCHESTRATOR_MIN_CONFIDENCE` (default: `0.4`)
- `ENABLE_LLM_CLASSIFIER` (default: `false`)
- `ORCHESTRATOR_LLM_BLEND_ALPHA` (default: `0.35`)
- `VERBOSE_LOGGING` (default: `false`)
- `ORCHESTRATOR_REDACT_SENSITIVE` (default: `true`)
- `ORCHESTRATOR_MAX_MESSAGE_CHARS` (default: `4000`)
- `ORCHESTRATOR_MAX_TOTAL_CHARS` (default: `20000`)
- `ORCHESTRATOR_ENABLE_HOT_RELOAD` (default: `false`)
- `ORCHESTRATOR_ENABLE_UPDATE_REGISTRY` (default: `false`)
- `ORCHESTRATOR_ADMIN_SECRET` (optional)
- `ORCHESTRATOR_TLS_CERT_FILE` (optional; enables local HTTPS when paired with key)
- `ORCHESTRATOR_TLS_KEY_FILE` (optional; enables local HTTPS when paired with cert)

## Tests

Run:

```bash
uv run pytest
```

Coverage includes:

- Keyword matching/scoring
- Alias resolution
- Confidence bounds
- Tie-breaking and weights
- Registry validation and error handling
- Empty-registry fallback
- Single strong-match prompts
- Near-tie ranked output
- Low-confidence disambiguation
- Hot-reload behavior

## How to Extend

- Add a new downstream MCP server by appending an entry to `mcp_registry.json` with:
	- unique `id`
	- `endpoint`
	- `categories`, `tools`, `keywords`, `weight`
- Add aliases in `category_aliases`.
- Optional LLM classification:
	- set `ENABLE_LLM_CLASSIFIER=true`
	- implement a custom classifier class with `classify_with_llm(messages)` returning category confidence map.

## Documentation

- [Model Context Protocol Python SDK](https://github.com/modelcontextprotocol/python-sdk)
