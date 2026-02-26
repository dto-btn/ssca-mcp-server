from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse


class DefaultFallback(BaseModel):
    category: str = "generic"
    message: str = "No clear match. Ask a clarifying question."


class RoutingRules(BaseModel):
    max_recommendations: int = Field(default=3, ge=1, le=10)
    tie_breaker: str = "weight_then_keyword_density"
    default_fallback: DefaultFallback = Field(default_factory=DefaultFallback)


class RegistryServer(BaseModel):
    id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)

    @field_validator("categories", "tools", "keywords")
    @classmethod
    def _strip_values(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value and value.strip()]

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("endpoint")
    @classmethod
    def _require_https_endpoint(cls, endpoint: str) -> str:
        normalized = endpoint.strip()
        parsed = urlparse(normalized)
        host = (parsed.hostname or "").lower()

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("endpoint must use http:// or https:// transport")

        if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("http:// endpoints are allowed only for local development hosts (localhost/127.0.0.1)")

        if "/mcp" not in normalized.lower():
            raise ValueError("endpoint must target an MCP HTTP path (for example, https://host/mcp)")
        return normalized


class RegistryModel(BaseModel):
    version: str = "1.0"
    mcp_servers: list[RegistryServer] = Field(default_factory=list)
    category_aliases: dict[str, str] = Field(default_factory=dict)
    routing_rules: RoutingRules = Field(default_factory=RoutingRules)

    @field_validator("category_aliases")
    @classmethod
    def _normalize_aliases(cls, aliases: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in aliases.items():
            k = key.strip().lower()
            v = value.strip().lower()
            if k and v:
                normalized[k] = v
        return normalized


def default_registry() -> RegistryModel:
    return RegistryModel()