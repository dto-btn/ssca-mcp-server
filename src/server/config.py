from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(value: str | None, default: bool = False) -> bool:
    """Parse common environment truthy values into bool."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    """Parse integer env var and fall back to default on invalid input."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    """Parse float env var and fall back to default on invalid input."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _first_non_empty(*values: str | None) -> str | None:
    """Return the first non-empty string from a precedence chain."""
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class OrchestratorSettings:
    registry_path: Path
    max_messages: int
    min_confidence: float
    enable_llm_classifier: bool
    azure_openai_endpoint: str | None
    azure_openai_api_version: str
    llm_model: str | None
    llm_timeout_seconds: float
    verbose_logging: bool
    redact_sensitive_tokens: bool
    max_message_chars: int
    max_total_chars: int
    enable_hot_reload: bool
    update_registry_enabled: bool
    admin_secret: str | None


def load_settings() -> OrchestratorSettings:
    """Load orchestrator settings from environment with safe bounds."""
    registry_path = Path(os.getenv("ORCHESTRATOR_REGISTRY_PATH", "./mcp_registry.json")).expanduser().resolve()

    return OrchestratorSettings(
        registry_path=registry_path,
        max_messages=max(1, _to_int(os.getenv("ORCHESTRATOR_MAX_MESSAGES"), 10)),
        min_confidence=max(0.0, min(1.0, _to_float(os.getenv("ORCHESTRATOR_MIN_CONFIDENCE"), 0.4))),
        enable_llm_classifier=_to_bool(os.getenv("ENABLE_LLM_CLASSIFIER"), False),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview"),
        llm_model=_first_non_empty(
            os.getenv("ORCHESTRATOR_LLM_MODEL"),
            os.getenv("GPT40_DEPLOYMENT_NAME"),
            os.getenv("DEFAULT_DEPLOYMENT_NAME"),
        ),
        llm_timeout_seconds=max(1.0, _to_float(os.getenv("ORCHESTRATOR_LLM_TIMEOUT_SECONDS"), 8.0)),
        verbose_logging=_to_bool(os.getenv("VERBOSE_LOGGING"), False),
        redact_sensitive_tokens=_to_bool(os.getenv("ORCHESTRATOR_REDACT_SENSITIVE"), True),
        max_message_chars=max(200, _to_int(os.getenv("ORCHESTRATOR_MAX_MESSAGE_CHARS"), 4000)),
        max_total_chars=max(2000, _to_int(os.getenv("ORCHESTRATOR_MAX_TOTAL_CHARS"), 20000)),
        enable_hot_reload=_to_bool(os.getenv("ORCHESTRATOR_ENABLE_HOT_RELOAD"), False),
        update_registry_enabled=_to_bool(os.getenv("ORCHESTRATOR_ENABLE_UPDATE_REGISTRY"), False),
        admin_secret=os.getenv("ORCHESTRATOR_ADMIN_SECRET"),
    )


settings = load_settings()