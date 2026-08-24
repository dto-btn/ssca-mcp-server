"""Routing orchestration layer that converts classification scores into MCP recommendations.

Uses a deterministic fallback and ambiguity strategy so clients can safely
continue even when confidence is low.
"""

from __future__ import annotations

from datetime import UTC, datetime
import re

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

TITLE_MIN_WORDS = 2
TITLE_MAX_WORDS = 5
TITLE_MAX_CHARS = 80
_TITLE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
_TITLE_SANITIZE_PATTERN = re.compile(r"[`*_~>#\[\](){}|]")
_TITLE_PUNCT_ONLY_PATTERN = re.compile(r"^[^A-Za-z0-9]+$")
_POLITE_PREFIX_PATTERN = re.compile(
    r"^(?:please\s+)?(?:can you|could you|would you|help me|i need to|i want to|i'd like to)\s+",
    re.IGNORECASE,
)

_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "help", "i", "i'm", "if", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "show", "tell", "that", "the", "to", "us", "we", "what",
    "with", "you", "your", "can", "could", "would", "should", "need", "want", "about", "so", "more", "make",
}
DEFAULT_CHAT_TITLE = "General Request"

_TITLE_ACTIONS = (
    "rewrite",
    "summarize",
    "draft",
    "write",
    "edit",
    "improve",
    "fix",
    "translate",
    "explain",
    "analyze",
    "compare",
    "schedule",
    "plan",
    "create",
    "query",
    "search",
    "find",
)
_TITLE_ACTION_LABELS = {
    "rewrite": "Rewrite",
    "summarize": "Summary",
    "draft": "Draft",
    "write": "Draft",
    "edit": "Edit",
    "improve": "Improvement",
    "fix": "Fix",
    "translate": "Translation",
    "explain": "Explanation",
    "analyze": "Analysis",
    "compare": "Comparison",
    "schedule": "Scheduling",
    "plan": "Plan",
    "create": "Creation",
    "query": "Query",
    "search": "Search",
    "find": "Lookup",
}
_TITLE_OBJECT_HINTS = (
    "email",
    "message",
    "document",
    "report",
    "policy",
    "query",
    "meeting",
    "summary",
    "proposal",
    "plan",
    "code",
)


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

    def _latest_user_message(self, messages: list[dict[str, str]]) -> str:
        """Return the most recent non-empty user message content."""
        for message in reversed(messages):
            if str(message.get("role", "")).strip().lower() != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return ""

    def _latest_message_content(self, messages: list[dict[str, str]]) -> str:
        """Return latest non-empty content from any role as a last-resort source."""
        for message in reversed(messages):
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return ""

    def _sanitize_title_text(self, value: str) -> str:
        """Normalize title candidate text and remove unsupported formatting."""
        cleaned = value.strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
        cleaned = cleaned.replace("```", " ")
        cleaned = _TITLE_SANITIZE_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;!?-_")
        if len(cleaned) > TITLE_MAX_CHARS:
            cleaned = cleaned[:TITLE_MAX_CHARS].rstrip(" .,:;!?-_")
        return cleaned

    def _is_safe_title(self, title: str) -> bool:
        """Validate title contract constraints for frontend auto-rename."""
        if not title:
            return False
        if len(title) > TITLE_MAX_CHARS:
            return False
        if _TITLE_PUNCT_ONLY_PATTERN.match(title):
            return False
        words = _TITLE_TOKEN_PATTERN.findall(title)
        return TITLE_MIN_WORDS <= len(words) <= TITLE_MAX_WORDS

    def _format_title_for_display(self, title: str) -> str:
        """Format normalized title into a readable short label."""
        tokens = _TITLE_TOKEN_PATTERN.findall(title)
        if not tokens:
            return ""

        formatted: list[str] = []
        for token in tokens:
            if token.isupper() and len(token) <= 5:
                formatted.append(token)
                continue
            formatted.append(token[:1].upper() + token[1:].lower())
        return " ".join(formatted)

    def _title_from_latest_user_message(self, latest_user_message: str) -> str:
        """Derive a deterministic short title from the latest user prompt."""
        sanitized = self._sanitize_title_text(latest_user_message)
        if not sanitized:
            return ""

        # Keep first sentence-like clause to focus on immediate intent.
        clause = re.split(r"[.!?\n;:]+", sanitized, maxsplit=1)[0].strip()
        if not clause:
            clause = sanitized

        clause = _POLITE_PREFIX_PATTERN.sub("", clause).strip()
        if not clause:
            clause = sanitized

        lower_clause = clause.lower()

        action = ""
        for verb in _TITLE_ACTIONS:
            if re.search(rf"\b{re.escape(verb)}\b", lower_clause):
                action = verb
                break

        obj = ""
        for hint in _TITLE_OBJECT_HINTS:
            if re.search(rf"\b{re.escape(hint)}\b", lower_clause):
                obj = hint
                break

        style_match = re.search(r"\b(?:more|less)\s+([a-z][a-z-]{2,})\b", lower_clause)
        style = style_match.group(1) if style_match else ""

        if style and obj and action:
            action_label = _TITLE_ACTION_LABELS.get(action, action.title())
            pattern_title = self._format_title_for_display(f"{style} {obj} {action_label}")
            if self._is_safe_title(pattern_title):
                return pattern_title

        if obj and action:
            action_label = _TITLE_ACTION_LABELS.get(action, action.title())
            pattern_title = self._format_title_for_display(f"{obj} {action_label}")
            if self._is_safe_title(pattern_title):
                return pattern_title

        all_tokens = _TITLE_TOKEN_PATTERN.findall(clause)
        informative_tokens = [token for token in all_tokens if token.lower() not in _TITLE_STOPWORDS]

        chosen = informative_tokens if len(informative_tokens) >= TITLE_MIN_WORDS else all_tokens
        if len(chosen) < TITLE_MIN_WORDS:
            return ""

        title = " ".join(chosen[:TITLE_MAX_WORDS])
        title = self._sanitize_title_text(title)
        return self._format_title_for_display(title)

    def _generate_chat_title(
        self,
        messages: list[dict[str, str]],
        categories_data: list[dict[str, object]],
        llm_title: str | None = None,
    ) -> tuple[str, str]:
        """Generate an optional chat title from latest user intent.

        LLM generation is attempted first when available; deterministic fallback
        ensures stable titles when LLM is disabled or does not return a usable
        title. The source is returned so clients can distinguish the outcomes.
        """
        latest_user_message = self._latest_user_message(messages)
        if not latest_user_message:
            latest_user_message = self._latest_message_content(messages)
        if not latest_user_message:
            return DEFAULT_CHAT_TITLE, "deterministic"

        if isinstance(llm_title, str):
            llm_title = self._sanitize_title_text(llm_title)
            llm_title = self._format_title_for_display(llm_title)
            if self._is_safe_title(llm_title):
                return llm_title, "ai"

        fallback_title = self._title_from_latest_user_message(latest_user_message)
        if self._is_safe_title(fallback_title):
            return fallback_title, "deterministic"

        if categories_data:
            category_name = str(categories_data[0].get("name", "")).strip()
            if category_name:
                category_words = category_name.replace("_", " ").replace("-", " ")
                category_title = self._format_title_for_display(f"{category_words} request")
                if self._is_safe_title(category_title):
                    return category_title, "deterministic"

        return DEFAULT_CHAT_TITLE, "deterministic"

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
            ranked, llm_title = self.classifier.score_servers_with_title(normalized, registry)
            max_recos = max_recommendations or registry.routing_rules.max_recommendations

            categories_data, classification_method, explanation = self._build_category_response(ranked, registry)
            route_response = self._build_route_response(
                ranked=ranked,
                registry=registry,
                max_recommendations=max_recos,
                require_single_best=require_single_best,
            )
            final_classification_method = str(
                route_response.get("classification_method") or classification_method
            )
            try:
                chat_title, chat_title_source = self._generate_chat_title(
                    normalized,
                    categories_data,
                    llm_title,
                )
            except Exception:
                logger.exception("chat title generation failed; returning routing response without title")
                chat_title = DEFAULT_CHAT_TITLE
                chat_title_source = "deterministic"

            response: dict[str, object] = {
                "categories": categories_data,
                "explanation": explanation,
                "classification_method": final_classification_method,
                "recommendations": route_response.get("recommendations", []),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            response["chat_title"] = chat_title
            response["chatTitle"] = chat_title
            response["chat_title_source"] = chat_title_source
            response["chatTitleSource"] = chat_title_source

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