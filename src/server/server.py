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
from typing import TypeVar

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


_T = TypeVar("_T", bound=BaseModel)


def _validate_input(
    model_cls: type[_T],
    tool_name: str,
    **kwargs: object,
) -> tuple[_T | None, dict[str, object] | None]:
    """Parse and validate *kwargs* into *model_cls*, returning an error envelope on failure."""
    try:
        return model_cls(**kwargs), None
    except Exception as error:
        return None, _error_response("invalid_input", f"Malformed {tool_name} input", str(error))


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
async def classify_context(
    messages: list[dict[str, str]],
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify chat context into one or more categories with confidence and keyword evidence."""
    payload, err = _validate_input(ClassifyContextInput, "classify_context", messages=messages, locale=locale, metadata=metadata)
    if err is not None:
        return err
    return await anyio.to_thread.run_sync(
        functools.partial(router.classify_context, payload.messages, locale=payload.locale, metadata=payload.metadata)  # type: ignore[union-attr]
    )


@mcp.tool()
async def suggest_route(
    messages: list[dict[str, str]],
    max_recommendations: int | None = None,
    require_single_best: bool = False,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Suggest one or more MCP server routes ranked by confidence and rationale."""
    payload, err = _validate_input(
        SuggestRouteInput, "suggest_route",
        messages=messages,
        max_recommendations=max_recommendations,
        require_single_best=require_single_best,
        locale=locale,
        metadata=metadata,
    )
    if err is not None:
        return err
    return await anyio.to_thread.run_sync(
        functools.partial(
            router.suggest_route,
            payload.messages,  # type: ignore[union-attr]
            max_recommendations=payload.max_recommendations,  # type: ignore[union-attr]
            require_single_best=payload.require_single_best,  # type: ignore[union-attr]
            locale=payload.locale,  # type: ignore[union-attr]
            metadata=payload.metadata,  # type: ignore[union-attr]
        )
    )


@mcp.tool()
async def classify_and_suggest(
    messages: list[dict[str, str]],
    max_recommendations: int | None = None,
    require_single_best: bool = False,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Single-pass classification + route suggestion using one scoring run."""
    payload, err = _validate_input(
        SuggestRouteInput, "classify_and_suggest",
        messages=messages,
        max_recommendations=max_recommendations,
        require_single_best=require_single_best,
        locale=locale,
        metadata=metadata,
    )
    if err is not None:
        return err
    return await anyio.to_thread.run_sync(
        functools.partial(
            router.classify_and_suggest,
            payload.messages,  # type: ignore[union-attr]
            max_recommendations=payload.max_recommendations,  # type: ignore[union-attr]
            require_single_best=payload.require_single_best,  # type: ignore[union-attr]
            locale=payload.locale,  # type: ignore[union-attr]
            metadata=payload.metadata,  # type: ignore[union-attr]
        )
    )


@mcp.tool()
async def route_and_forward(
    messages: list[dict[str, str]],
    target_mcp_server_id: str | None = None,
    tool_name: str | None = None,
    payload: dict[str, object] | None = None,
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Phase-1 stub that returns selected route and a mock forwarding response."""
    req, err = _validate_input(
        RouteAndForwardInput, "route_and_forward",
        messages=messages,
        target_mcp_server_id=target_mcp_server_id,
        tool_name=tool_name,
        payload=payload,
        locale=locale,
        metadata=metadata,
    )
    if err is not None:
        return err
    return await anyio.to_thread.run_sync(
        functools.partial(
            router.route_and_forward_stub,
            req.messages,  # type: ignore[union-attr]
            target_mcp_server_id=req.target_mcp_server_id,  # type: ignore[union-attr]
            tool_name=req.tool_name,  # type: ignore[union-attr]
            payload=req.payload,  # type: ignore[union-attr]
            locale=req.locale,  # type: ignore[union-attr]
            metadata=req.metadata,  # type: ignore[union-attr]
        )
    )


@mcp.tool()
async def update_registry(
    upsert: list[dict[str, object]] | None = None,
    remove: list[str] | None = None,
    admin_secret: str | None = None,
) -> dict[str, object]:
    """Admin-only registry upsert/remove with schema validation and file locking."""
    req, err = _validate_input(UpdateRegistryInput, "update_registry", upsert=upsert, remove=remove, admin_secret=admin_secret)
    if err is not None:
        return err

    try:
        updated = await anyio.to_thread.run_sync(
            functools.partial(
                store.update_registry,
                upsert=req.upsert or [],  # type: ignore[union-attr]
                remove=req.remove or [],  # type: ignore[union-attr]
                provided_secret=req.admin_secret,  # type: ignore[union-attr]
            )
        )
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
