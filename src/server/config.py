from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class OrchestratorSettings:
    registry_path: Path
    max_messages: int
    min_confidence: float
    enable_llm_classifier: bool
    llm_blend_alpha: float
    verbose_logging: bool
    redact_sensitive_tokens: bool
    max_message_chars: int
    max_total_chars: int
    enable_hot_reload: bool
    update_registry_enabled: bool
    admin_secret: str | None


def load_settings() -> OrchestratorSettings:
    registry_path = Path(os.getenv("ORCHESTRATOR_REGISTRY_PATH", "./mcp_registry.json")).expanduser().resolve()
    llm_blend_alpha = _to_float(os.getenv("ORCHESTRATOR_LLM_BLEND_ALPHA"), 0.35)
    if llm_blend_alpha < 0:
        llm_blend_alpha = 0.0
    if llm_blend_alpha > 1:
        llm_blend_alpha = 1.0

    return OrchestratorSettings(
        registry_path=registry_path,
        max_messages=max(1, _to_int(os.getenv("ORCHESTRATOR_MAX_MESSAGES"), 10)),
        min_confidence=max(0.0, min(1.0, _to_float(os.getenv("ORCHESTRATOR_MIN_CONFIDENCE"), 0.4))),
        enable_llm_classifier=_to_bool(os.getenv("ENABLE_LLM_CLASSIFIER"), False),
        llm_blend_alpha=llm_blend_alpha,
        verbose_logging=_to_bool(os.getenv("VERBOSE_LOGGING"), False),
        redact_sensitive_tokens=_to_bool(os.getenv("ORCHESTRATOR_REDACT_SENSITIVE"), True),
        max_message_chars=max(200, _to_int(os.getenv("ORCHESTRATOR_MAX_MESSAGE_CHARS"), 4000)),
        max_total_chars=max(2000, _to_int(os.getenv("ORCHESTRATOR_MAX_TOTAL_CHARS"), 20000)),
        enable_hot_reload=_to_bool(os.getenv("ORCHESTRATOR_ENABLE_HOT_RELOAD"), False),
        update_registry_enabled=_to_bool(os.getenv("ORCHESTRATOR_ENABLE_UPDATE_REGISTRY"), False),
        admin_secret=os.getenv("ORCHESTRATOR_ADMIN_SECRET"),
    )


settings = load_settings()