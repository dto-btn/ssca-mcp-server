from __future__ import annotations

from datetime import UTC, datetime

try:
    from .classifier import KeywordClassifier, resolve_alias
    from .config import OrchestratorSettings
    from .logging_utils import get_logger, redact_text
    from .registry import RegistryStore
except ImportError:
    from classifier import KeywordClassifier, resolve_alias
    from config import OrchestratorSettings
    from logging_utils import get_logger, redact_text
    from registry import RegistryStore

logger = get_logger("orchestrator.router")


class OrchestratorRouter:
    def __init__(self, settings: OrchestratorSettings, registry_store: RegistryStore):
        self.settings = settings
        self.registry_store = registry_store
        self.classifier = KeywordClassifier(settings=settings)

    def _normalize_messages(self, messages: list[object]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            if not role or not isinstance(content, str):
                continue
            normalized.append({"role": str(role), "content": content})
        return normalized

    def classify_context(
        self,
        messages: list[object],
        locale: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized = self._normalize_messages(messages)
        try:
            registry = self.registry_store.load_registry()
            categories = self.classifier.classify_categories(normalized, registry)
            if not categories:
                fallback_category = registry.routing_rules.default_fallback.category
                fallback_message = registry.routing_rules.default_fallback.message
                categories_data = [
                    {
                        "name": fallback_category,
                        "confidence": 0.0,
                        "matched_keywords": [],
                        "classification_method": "fallback",
                    }
                ]
                explanation = (
                    "No category had enough keyword evidence. "
                    f"Fallback selected: {fallback_message}"
                )
                classification_method = "fallback"
            else:
                categories_data = [
                    {
                        "name": cat.name,
                        "confidence": round(cat.confidence, 4),
                        "matched_keywords": cat.matched_keywords,
                        "classification_method": cat.classification_method,
                    }
                    for cat in categories
                ]
                top = categories[0]
                classification_method = top.classification_method
                uncertainty = ""
                if top.confidence < self.settings.min_confidence:
                    uncertainty = " Confidence is low; ask a clarifying question."
                explanation = (
                    f"Top category '{top.name}' selected via {top.classification_method} classification with evidence: "
                    f"{', '.join(top.matched_keywords[:5]) or 'none'}.{uncertainty}"
                )

            if self.settings.verbose_logging:
                snippet = " ".join(msg["content"][:120] for msg in normalized)
                if self.settings.redact_sensitive_tokens:
                    snippet = redact_text(snippet)
                logger.info("Classification complete locale=%s metadata=%s context=%s", locale, metadata or {}, snippet)

            return {
                "categories": categories_data,
                "explanation": explanation,
                "classification_method": classification_method,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as error:
            logger.exception("classify_context failed")
            return {
                "categories": [],
                "explanation": "Classification failed; fallback guidance returned.",
                "error": {
                    "code": "classification_failed",
                    "message": str(error),
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def suggest_route(
        self,
        messages: list[object],
        max_recommendations: int | None = None,
        require_single_best: bool = False,
        locale: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized = self._normalize_messages(messages)
        try:
            registry = self.registry_store.load_registry()
            ranked = self.classifier.score_servers(normalized, registry)
            max_recos = max_recommendations or registry.routing_rules.max_recommendations

            if not ranked:
                fallback_category = registry.routing_rules.default_fallback.category
                fallback = {
                    "category": fallback_category,
                    "upstream": None,
                    "reason": registry.routing_rules.default_fallback.message,
                    "suggestions_for_user": [
                        "Are you trying to query a database or search the web?",
                        "Do you want help with calendar scheduling?",
                        "Can you share the main action you want to perform?",
                    ],
                }
                return {
                    "recommendations": [],
                    "fallback": fallback,
                    "classification_method": "fallback",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            top_conf = ranked[0].confidence
            tie_delta = 0.05
            filtered: list = []
            for item in ranked:
                if len(filtered) >= max_recos:
                    break
                if top_conf - item.confidence <= tie_delta or len(filtered) == 0:
                    filtered.append(item)

            if require_single_best:
                filtered = filtered[:1]

            recommendations = []
            for item in filtered:
                normalized_category = resolve_alias(item.category, registry.category_aliases)
                rationale = (
                    f"Matched keywords: {', '.join(item.matched_keywords[:5]) or 'none'}; "
                    f"weighted confidence={item.confidence:.3f}."
                )
                if require_single_best and item.confidence < 0.6:
                    rationale += " Confidence below 0.6; disambiguation recommended."

                recommendations.append(
                    {
                        "mcp_server_id": item.server.id,
                        "endpoint": item.server.endpoint,
                        "category": normalized_category,
                        "confidence": round(item.confidence, 4),
                        "matched_keywords": item.matched_keywords,
                        "classification_method": item.classification_method,
                        "rationale": rationale,
                    }
                )

            response: dict[str, object] = {
                "recommendations": recommendations,
                "classification_method": (
                    str(recommendations[0].get("classification_method")) if recommendations else "fallback"
                ),
                "plan": None,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if require_single_best and recommendations and recommendations[0]["confidence"] < 0.6:
                response["disambiguation_note"] = (
                    "Top route has low confidence. Ask whether user wants web search, DB operation, or calendar action."
                )

            if self.settings.verbose_logging:
                logger.info(
                    "Routing complete locale=%s metadata=%s recommendation_count=%s",
                    locale,
                    metadata or {},
                    len(recommendations),
                )
            return response
        except Exception as error:
            logger.exception("suggest_route failed")
            return {
                "recommendations": [],
                "fallback": {
                    "reason": "Routing failed due to a server-side error.",
                    "suggestions_for_user": ["Please clarify if you need web, database, or scheduling help."],
                },
                "error": {
                    "code": "routing_failed",
                    "message": str(error),
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def route_and_forward_stub(
        self,
        messages: list[object],
        target_mcp_server_id: str | None = None,
        tool_name: str | None = None,
        payload: dict[str, object] | None = None,
        locale: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        routing = self.suggest_route(
            messages=messages,
            max_recommendations=1,
            require_single_best=True,
            locale=locale,
            metadata=metadata,
        )
        selected = target_mcp_server_id
        if not selected and routing.get("recommendations"):
            selected = str(routing["recommendations"][0]["mcp_server_id"])

        return {
            "status": "stub",
            "selected_mcp_server_id": selected,
            "selected_category": (
                str(routing["recommendations"][0]["category"])
                if routing.get("recommendations")
                else str((routing.get("fallback") or {}).get("category", "generic"))
            ),
            "selected_tool": tool_name,
            "payload": payload or {},
            "plan": {
                "supported": True,
                "description": "Future workflow: chain multiple MCP calls with per-step reasoning.",
                "steps": [],
            },
            "forward_result": {
                "message": "Forwarding is not yet implemented. This is a phase-1 stub.",
                "next_step": "Implement MCP client call dispatch in route_and_forward.",
            },
            "route_suggestion": routing,
            "timestamp": datetime.now(UTC).isoformat(),
        }