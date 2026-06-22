"""FastMCP + HTTP entrypoint for orchestrator classification and routing tools.

This service exposes a dual-surface server:
- MCP tools for SDK clients.
- A lightweight HTTP endpoint used by playground pre-routing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import functools
import json
import os
from datetime import UTC, datetime

import anyio

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

try:
    from .config import settings
    from .logging_utils import get_logger
    from .registry import RegistryStore
    from .router import OrchestratorRouter
except ImportError:
    import sys
    from pathlib import Path

    current_dir = str(Path(__file__).resolve().parent)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from config import settings
    from logging_utils import get_logger
    from registry import RegistryStore
    from router import OrchestratorRouter

logger = get_logger("orchestrator.server")
store = RegistryStore(settings)
router = OrchestratorRouter(settings=settings, registry_store=store)

mcp = FastMCP("Orchestrator MCP Server")


def _log_startup_config_summary() -> None:
    if not settings.enable_llm_classifier:
        logger.info("LLM classifier disabled; routing uses keyword classification and generic fallback.")
        return

    missing: list[str] = []
    if not settings.litellm_proxy_url:
        missing.append("ORCHESTRATOR_LITELLM_PROXY_URL")
    if not settings.llm_model:
        missing.append("ORCHESTRATOR_LLM_MODEL")

    if missing:
        logger.warning(
            "LLM classifier enabled but configuration is incomplete (missing=%s). "
            "Orchestrator will fall back to keyword classification.",
            ",".join(missing),
        )
        return

    logger.info(
        "LLM classifier enabled (proxy_url=%s, model=%s, timeout_seconds=%.1f, min_confidence=%.2f).",
        settings.litellm_proxy_url,
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.min_confidence,
    )


_log_startup_config_summary()


def _allowed_origins_from_env() -> list[str]:
    # Keep CORS policy environment-driven so local playground variants can
    # connect without rebuilding the orchestrator image.
    raw = os.getenv(
        "ORCHESTRATOR_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,https://localhost:8080,https://127.0.0.1:8080,https://localhost:5173,https://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_http_app = mcp.streamable_http_app()
_cors_wrapped_mcp_app = CORSMiddleware(
    _http_app,
    allow_origins=_allowed_origins_from_env(),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
    allow_credentials=True,
)


@asynccontextmanager
async def orchestrator_lifespan(_: Starlette) -> AsyncIterator[None]:
    """Initialize FastMCP session manager for streamable HTTP transport."""
    async with mcp.session_manager.run():
        yield


def _error_response(
    code: str,
    message: str,
    details: str | None = None,
) -> dict[str, object]:
    """Build a uniform error envelope for both HTTP and MCP tool responses."""
    err: dict[str, object] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"error": err, "timestamp": datetime.now(UTC).isoformat()}


async def suggest_route_http(request: Request) -> JSONResponse:
    """HTTP helper endpoint for frontend pre-routing before LLM invocation."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_error_response("invalid_json", "Malformed JSON payload"), status_code=400)

    if not isinstance(body, dict):
        return JSONResponse(_error_response("invalid_input", "Request body must be a JSON object"), status_code=400)

    try:
        payload = SuggestRouteInput(
            messages=[ChatMessage(**m) for m in (body.get("messages") or [])],
            max_recommendations=body.get("max_recommendations"),
            require_single_best=bool(body.get("require_single_best", False)),
            locale=body.get("locale"),
            metadata=body.get("metadata"),
        )
    except Exception as exc:
        return JSONResponse(_error_response("invalid_input", "Invalid request payload", str(exc)), status_code=400)

    # router.suggest_route is synchronous (CPU/IO); run it in a thread so the
    # Starlette event loop is not blocked while classification runs.
    result = await anyio.to_thread.run_sync(
        functools.partial(
            router.suggest_route,
            payload.messages,
            max_recommendations=payload.max_recommendations,
            require_single_best=payload.require_single_best,
            locale=payload.locale,
            metadata=payload.metadata,
        )
    )

    # Return a meaningful HTTP status so clients can distinguish server errors
    # from successful (possibly ambiguous) classification responses.
    if "error" in result:
        code = result["error"].get("code", "")
        status = 400 if code in ("invalid_input", "invalid_json") else 500
        return JSONResponse(result, status_code=status)
    return JSONResponse(result)


app = Starlette(
    lifespan=orchestrator_lifespan,
    routes=[
        Route("/orchestrator/suggest-route", suggest_route_http, methods=["POST"]),
        Mount("/", app=_cors_wrapped_mcp_app),
    ]
)


class ChatMessage(BaseModel):
    role: str = Field(min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=20000)


class ClassifyContextInput(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    locale: str | None = None
    metadata: dict[str, object] | None = None


class SuggestRouteInput(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_recommendations: int | None = Field(default=None, ge=1, le=10)
    require_single_best: bool = False
    locale: str | None = None
    metadata: dict[str, object] | None = None


class ClassifyAndSuggestInput(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_recommendations: int | None = Field(default=None, ge=1, le=10)
    require_single_best: bool = False
    locale: str | None = None
    metadata: dict[str, object] | None = None


class RouteAndForwardInput(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    target_mcp_server_id: str | None = None
    tool_name: str | None = None
    payload: dict[str, object] | None = None
    locale: str | None = None
    metadata: dict[str, object] | None = None


class UpdateRegistryInput(BaseModel):
    upsert: list[dict[str, object]] | None = None
    remove: list[str] | None = None
    admin_secret: str | None = None


@mcp.resource("orchestrator://registry")
def get_registry_resource() -> str:
    """Returns the orchestrator registry as JSON."""
    registry = store.load_registry()
    return json.dumps(registry.model_dump(mode="json"), indent=2)


@mcp.tool()
def classify_context(
    messages: list[dict[str, str]],
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify chat context into one or more categories with confidence and keyword evidence."""
    try:
        payload = ClassifyContextInput(messages=[ChatMessage(**m) for m in messages], locale=locale, metadata=metadata)
    except Exception as error:
        return _error_response("invalid_input", "Malformed classify_context input", str(error))

    return router.classify_context(payload.messages, locale=payload.locale, metadata=payload.metadata)


@mcp.tool()
def suggest_route(
    messages: list[dict[str, str]],
    max_recommendations: int | None = None,
    require_single_best: bool = False,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Suggest one or more MCP server routes ranked by confidence and rationale."""
    try:
        payload = SuggestRouteInput(
            messages=[ChatMessage(**m) for m in messages],
            max_recommendations=max_recommendations,
            require_single_best=require_single_best,
            locale=locale,
            metadata=metadata,
        )
    except Exception as error:
        return _error_response("invalid_input", "Malformed suggest_route input", str(error))

    return router.suggest_route(
        payload.messages,
        max_recommendations=payload.max_recommendations,
        require_single_best=payload.require_single_best,
        locale=payload.locale,
        metadata=payload.metadata,
    )


@mcp.tool()
def classify_and_suggest(
    messages: list[dict[str, str]],
    max_recommendations: int | None = None,
    require_single_best: bool = False,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Single-pass classification + route suggestion using one scoring run."""
    try:
        payload = ClassifyAndSuggestInput(
            messages=[ChatMessage(**m) for m in messages],
            max_recommendations=max_recommendations,
            require_single_best=require_single_best,
            locale=locale,
            metadata=metadata,
        )
    except Exception as error:
        return _error_response("invalid_input", "Malformed classify_and_suggest input", str(error))

    return router.classify_and_suggest(
        payload.messages,
        max_recommendations=payload.max_recommendations,
        require_single_best=payload.require_single_best,
        locale=payload.locale,
        metadata=payload.metadata,
    )


@mcp.tool()
def route_and_forward(
    messages: list[dict[str, str]],
    target_mcp_server_id: str | None = None,
    tool_name: str | None = None,
    payload: dict[str, object] | None = None,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Phase-1 stub that returns selected route and a mock forwarding response."""
    try:
        req = RouteAndForwardInput(
            messages=[ChatMessage(**m) for m in messages],
            target_mcp_server_id=target_mcp_server_id,
            tool_name=tool_name,
            payload=payload,
            locale=locale,
            metadata=metadata,
        )
    except Exception as error:
        return _error_response("invalid_input", "Malformed route_and_forward input", str(error))

    return router.route_and_forward_stub(
        req.messages,
        target_mcp_server_id=req.target_mcp_server_id,
        tool_name=req.tool_name,
        payload=req.payload,
        locale=req.locale,
        metadata=req.metadata,
    )


@mcp.tool()
def update_registry(
    upsert: list[dict[str, object]] | None = None,
    remove: list[str] | None = None,
    admin_secret: str | None = None,
) -> dict[str, object]:
    """Admin-only registry upsert/remove with schema validation and file locking."""
    try:
        req = UpdateRegistryInput(upsert=upsert, remove=remove, admin_secret=admin_secret)
    except Exception as error:
        return _error_response("invalid_input", "Malformed update_registry input", str(error))

    try:
        updated = store.update_registry(upsert=req.upsert or [], remove=req.remove or [], provided_secret=req.admin_secret)
    except PermissionError as error:
        return _error_response("forbidden", str(error))
    except Exception as error:
        logger.exception("Registry update failed")
        return _error_response("registry_update_failed", str(error))

    return {
        "status": "ok",
        "registry": updated.model_dump(mode="json"),
        "timestamp": datetime.now(UTC).isoformat(),
    }
