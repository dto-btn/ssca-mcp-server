from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

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


def _allowed_origins_from_env() -> list[str]:
    raw = os.getenv(
        "ORCHESTRATOR_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,https://localhost:8080,https://127.0.0.1:8080,https://localhost:5173,https://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_http_app = mcp.streamable_http_app()
app = CORSMiddleware(
    _http_app,
    allow_origins=_allowed_origins_from_env(),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
    allow_credentials=True,
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
def classify_context(
    messages: list[dict[str, str]],
    locale: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify chat context into one or more categories with confidence and keyword evidence."""
    try:
        payload = ClassifyContextInput(messages=[ChatMessage(**m) for m in messages], locale=locale, metadata=metadata)
    except Exception as error:
        return {
            "error": {
                "code": "invalid_input",
                "message": "Malformed classify_context input",
                "details": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    result = router.classify_context(payload.messages, locale=payload.locale, metadata=payload.metadata)
    return result


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
        return {
            "error": {
                "code": "invalid_input",
                "message": "Malformed suggest_route input",
                "details": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    return router.suggest_route(
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
        return {
            "error": {
                "code": "invalid_input",
                "message": "Malformed route_and_forward input",
                "details": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

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
        return {
            "error": {
                "code": "invalid_input",
                "message": "Malformed update_registry input",
                "details": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    try:
        updated = store.update_registry(upsert=req.upsert or [], remove=req.remove or [], provided_secret=req.admin_secret)
    except PermissionError as error:
        return {
            "error": {
                "code": "forbidden",
                "message": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as error:
        logger.exception("Registry update failed")
        return {
            "error": {
                "code": "registry_update_failed",
                "message": str(error),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    return {
        "status": "ok",
        "registry": updated.model_dump(mode="json"),
        "timestamp": datetime.now(UTC).isoformat(),
    }
