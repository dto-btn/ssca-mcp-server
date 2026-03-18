"""Routing orchestration layer that converts classification scores into MCP recommendations.

Uses a deterministic fallback and ambiguity strategy so clients can safely
continue even when confidence is low.
"""

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
        """Compose classification and registry services for routing operations."""
        self.settings = settings
        self.registry_store = registry_store
        self.classifier = KeywordClassifier(settings=settings)

    def _normalize_messages(self, messages: list[object]) -> list[dict[str, str]]:
        """Normalize mixed message objects/dicts into ``{role, content}`` records."""
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

    def _build_category_response(
        self,
        ranked: list,
        registry,
    ) -> tuple[list[dict[str, object]], str, str]:
        """Build category-oriented output with explanation text for callers."""
        if not ranked:
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
            return categories_data, "fallback", explanation

        best_by_category: dict[str, object] = {}
        for item in ranked:
            # Preserve only the strongest server evidence for each category.
            existing = best_by_category.get(item.category)
            if existing is None or item.confidence > existing.confidence:
                best_by_category[item.category] = item

        categories_data = [
            {
                "name": category,
                "confidence": round(score.confidence, 4),
                "matched_keywords": score.matched_keywords,
                "classification_method": score.classification_method,
            }
            for category, score in best_by_category.items()
        ]
        categories_data.sort(key=lambda item: float(item["confidence"]), reverse=True)

        top = categories_data[0]
        top_confidence = float(top["confidence"])
        if top_confidence < self.settings.min_confidence:
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
                "Top category confidence below threshold. "
                f"Fallback selected: {fallback_message}"
            )
            return categories_data, "fallback", explanation

        explanation = (
            f"Top category '{top['name']}' selected via {top['classification_method']} classification with evidence: "
            f"{', '.join(top['matched_keywords'][:5]) or 'none'}."
        )
        return categories_data, str(top["classification_method"]), explanation

    def _build_route_response(
        self,
        ranked: list,
        registry,
        max_recommendations: int,
        require_single_best: bool,
    ) -> dict[str, object]:
        """Build downstream route recommendations from ranked server scores.

        The response includes fallback guidance when there is no reliable match,
        and optional disambiguation hints when confidence is low.
        """
        if not ranked:
            # Explicitly returning upstream=None indicates "no MCP call" rather
            # than a transport failure, which lets clients continue model-only.
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
            }

        top_conf = ranked[0].confidence
        if top_conf < self.settings.min_confidence:
            # Keep the same no-upstream contract for low-confidence outcomes.
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
            }
        tie_delta = 0.05
        filtered: list = []
        seen_categories: set[str] = set()
        seen_server_ids: set[str] = set()
        for item in ranked:
            if len(filtered) >= max_recommendations:
                break
            if item.server.id in seen_server_ids:
                continue
            normalized_category = resolve_alias(item.category, registry.category_aliases)
            is_first = len(filtered) == 0
            is_near_tie = top_conf - item.confidence <= tie_delta
            is_new_category = normalized_category not in seen_categories

            # Keep near ties and also retain distinct categories so compound
            # intents can route to more than one MCP server.
            if is_first or is_near_tie or is_new_category:
                filtered.append(item)
                seen_categories.add(normalized_category)
                seen_server_ids.add(item.server.id)

        if require_single_best:
            filtered = filtered[:1]

        ambiguous_categories: set[str] = set()
        ambiguous_keywords: set[str] = set()
        ambiguity_candidates = [
            item
            for item in ranked
            if (top_conf - item.confidence <= tie_delta) and item.confidence >= self.settings.min_confidence
        ]
        for idx, left in enumerate(ambiguity_candidates):
            left_category = resolve_alias(left.category, registry.category_aliases)
            left_keywords = {keyword.strip().lower() for keyword in left.matched_keywords if keyword.strip()}
            if not left_keywords:
                continue
            for right in ambiguity_candidates[idx + 1 :]:
                right_category = resolve_alias(right.category, registry.category_aliases)
                if left_category == right_category:
                    continue
                right_keywords = {keyword.strip().lower() for keyword in right.matched_keywords if keyword.strip()}
                overlaps = left_keywords & right_keywords
                if overlaps:
                    ambiguous_categories.update({left_category, right_category})
                    ambiguous_keywords.update(overlaps)

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
        }

        if len(ambiguous_categories) >= 2 and ambiguous_keywords:
            categories_text = " or ".join(sorted(ambiguous_categories))
            keyword_list = sorted(ambiguous_keywords)
            keyword_text = ", ".join(keyword_list[:3])
            response["disambiguation_note"] = (
                "Ambiguous intent detected: one or more keywords map to multiple categories."
            )
            response["clarifying_question"] = (
                f"I noticed keyword(s) like '{keyword_text}' could map to multiple categories. "
                f"Did you mean {categories_text}?"
            )

        if require_single_best and recommendations and recommendations[0]["confidence"] < 0.6:
            response["disambiguation_note"] = (
                "Top route has low confidence. Ask whether user wants web search, DB operation, or calendar action."
            )
        return response

    def classify_context(
        self,
        messages: list[object],
        locale: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Classify conversation messages into categories with confidence."""
        normalized = self._normalize_messages(messages)
        try:
            registry = self.registry_store.load_registry()
            ranked = self.classifier.score_servers(normalized, registry)
            categories_data, classification_method, explanation = self._build_category_response(ranked, registry)

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
        """Recommend MCP targets for the current conversation context."""
        normalized = self._normalize_messages(messages)
        try:
            registry = self.registry_store.load_registry()
            ranked = self.classifier.score_servers(normalized, registry)
            max_recos = max_recommendations or registry.routing_rules.max_recommendations
            response = self._build_route_response(
                ranked=ranked,
                registry=registry,
                max_recommendations=max_recos,
                require_single_best=require_single_best,
            )
            response["timestamp"] = datetime.now(UTC).isoformat()

            if self.settings.verbose_logging:
                logger.info(
                    "Routing complete locale=%s metadata=%s recommendation_count=%s",
                    locale,
                    metadata or {},
                    len(response.get("recommendations", [])),
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

    def classify_and_suggest(
        self,
        messages: list[object],
        max_recommendations: int | None = None,
        require_single_best: bool = False,
        locale: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Run one scoring pass and return both category and route outputs."""
        normalized = self._normalize_messages(messages)
        try:
            registry = self.registry_store.load_registry()
            ranked = self.classifier.score_servers(normalized, registry)
            max_recos = max_recommendations or registry.routing_rules.max_recommendations

            categories_data, classification_method, explanation = self._build_category_response(ranked, registry)
            route_response = self._build_route_response(
                ranked=ranked,
                registry=registry,
                max_recommendations=max_recos,
                require_single_best=require_single_best,
            )

            response: dict[str, object] = {
                "categories": categories_data,
                "explanation": explanation,
                "classification_method": (
                    str(route_response.get("classification_method"))
                    if route_response.get("classification_method")
                    else classification_method
                ),
                "recommendations": route_response.get("recommendations", []),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if "fallback" in route_response:
                response["fallback"] = route_response["fallback"]
            if "plan" in route_response:
                response["plan"] = route_response["plan"]
            if "disambiguation_note" in route_response:
                response["disambiguation_note"] = route_response["disambiguation_note"]

            if self.settings.verbose_logging:
                logger.info(
                    "Classify+route complete locale=%s metadata=%s recommendation_count=%s",
                    locale,
                    metadata or {},
                    len(response.get("recommendations", [])),
                )

            return response
        except Exception as error:
            logger.exception("classify_and_suggest failed")
            return {
                "categories": [],
                "recommendations": [],
                "fallback": {
                    "category": "general",
                    "reason": "Classification and routing failed due to a server-side error.",
                    "upstream": None,
                },
                "error": {
                    "code": "classify_and_suggest_failed",
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
        """Return route selection plus a placeholder forward-plan payload.

        This intentionally does not execute downstream MCP calls yet; it surfaces
        the selected route and a deterministic envelope for future expansion.
        """
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
                else str((routing.get("fallback") or {}).get("category", "general"))
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