#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CERT_FILE="$ROOT_DIR/certs/localhost.crt"
DEFAULT_KEY_FILE="$ROOT_DIR/certs/localhost.key"

export ORCHESTRATOR_ALLOWED_ORIGINS="${ORCHESTRATOR_ALLOWED_ORIGINS:-http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,https://localhost:8080,https://127.0.0.1:8080,https://localhost:5173,https://127.0.0.1:5173}"
export ORCHESTRATOR_HOST="${ORCHESTRATOR_HOST:-127.0.0.1}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8000}"
export ORCHESTRATOR_GRACEFUL_SHUTDOWN_SECONDS="${ORCHESTRATOR_GRACEFUL_SHUTDOWN_SECONDS:-2}"
export ORCHESTRATOR_TLS_CERT_FILE="${ORCHESTRATOR_TLS_CERT_FILE:-}"
export ORCHESTRATOR_TLS_KEY_FILE="${ORCHESTRATOR_TLS_KEY_FILE:-}"
export ORCHESTRATOR_AUTO_TLS="${ORCHESTRATOR_AUTO_TLS:-false}"

if [[ "$ORCHESTRATOR_AUTO_TLS" == "true" && -z "$ORCHESTRATOR_TLS_CERT_FILE" && -z "$ORCHESTRATOR_TLS_KEY_FILE" ]]; then
	if [[ -f "$DEFAULT_CERT_FILE" && -f "$DEFAULT_KEY_FILE" ]]; then
		ORCHESTRATOR_TLS_CERT_FILE="$DEFAULT_CERT_FILE"
		ORCHESTRATOR_TLS_KEY_FILE="$DEFAULT_KEY_FILE"
	fi
fi

echo "Starting playground orchestrator on https://${ORCHESTRATOR_HOST}:${ORCHESTRATOR_PORT}/mcp"
echo "Allowed origins: ${ORCHESTRATOR_ALLOWED_ORIGINS}"
echo "Graceful shutdown timeout: ${ORCHESTRATOR_GRACEFUL_SHUTDOWN_SECONDS}s"

UVICORN_ARGS=(
	src.server.server:app
	--host "$ORCHESTRATOR_HOST"
	--port "$ORCHESTRATOR_PORT"
	--timeout-graceful-shutdown "$ORCHESTRATOR_GRACEFUL_SHUTDOWN_SECONDS"
)

if [[ -n "$ORCHESTRATOR_TLS_CERT_FILE" && -n "$ORCHESTRATOR_TLS_KEY_FILE" ]]; then
	echo "TLS enabled with cert/key files"
	UVICORN_ARGS+=(--ssl-certfile "$ORCHESTRATOR_TLS_CERT_FILE" --ssl-keyfile "$ORCHESTRATOR_TLS_KEY_FILE")
else
	echo "TLS disabled; serving local MCP over http://${ORCHESTRATOR_HOST}:${ORCHESTRATOR_PORT}/mcp"
	echo "To enable HTTPS locally, set ORCHESTRATOR_AUTO_TLS=true or set ORCHESTRATOR_TLS_CERT_FILE/ORCHESTRATOR_TLS_KEY_FILE"
fi

exec uv run uvicorn "${UVICORN_ARGS[@]}"
