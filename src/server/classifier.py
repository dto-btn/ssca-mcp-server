"""Keyword and optional LLM-assisted category classification for orchestrator routing.

Includes robust JSON parsing for model responses and a hybrid scoring pipeline
that can map one user request to multiple candidate MCP categories.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

try:
    from .config import OrchestratorSettings
    from .logging_utils import get_logger
    from .schemas import RegistryModel, RegistryServer
except ImportError:
    from config import OrchestratorSettings
    from logging_utils import get_logger
    from schemas import RegistryModel, RegistryServer

WORD_PATTERN = re.compile(r"[a-z0-9]+")
logger = get_logger("orchestrator.classifier")

MAX_LLM_CATEGORIES = 3
MAX_LLM_RATIONALE_CHARS = 160


def _extract_response_text(response: object) -> str:
    """Extract assistant text from OpenAI Responses API payload variants."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        chunks: list[str] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type != "message":
                continue
            contents = getattr(item, "content", None)
            if not isinstance(contents, list):
                continue
            for content in contents:
                content_type = getattr(content, "type", None)
                if content_type in {"output_text", "text"}:
                    text_value = getattr(content, "text", None)
                    if isinstance(text_value, str) and text_value:
                        chunks.append(text_value)
        if chunks:
            return "".join(chunks)

    return ""


def _try_parse_json_object(content: str) -> dict[str, object] | None:
    """Best-effort parse for LLM JSON responses.

    Handles common non-strict variants (markdown fences, wrapped prose, and
    trailing commas) without raising, so routing can continue gracefully.
    """
    if not content:
        return None

    candidates: list[str] = []
    trimmed = content.strip()
    if trimmed:
        candidates.append(trimmed)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, flags=re.IGNORECASE)
    if fenced and fenced.group(1).strip():
        candidates.append(fenced.group(1).strip())

    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start >= 0 and end > start:
        candidates.append(trimmed[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # Retry with trailing commas removed: {"a":1,} / [1,2,]
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            if cleaned == candidate:
                continue
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


@dataclass
class CategoryMatch:
    name: str
    confidence: float
    matched_keywords: list[str]
    raw_score: float
    classification_method: str


@dataclass
class ServerScore:
    server: RegistryServer
    category: str
    confidence: float
    matched_keywords: list[str]
    raw_score: float
    density: float
    classification_method: str


class LlmClassifierPlugin:
    def __init__(self, settings: OrchestratorSettings):
        """Initialize LiteLLM proxy client used for intent category inference.

        The plugin is intentionally resilient: configuration problems or SDK import
        failures only disable LLM classification and allow deterministic keyword
        routing to continue.
        """
        self.settings = settings
        self._client = None
        self._enabled = settings.enable_llm_classifier
        if not self._enabled:
            return
        if not settings.litellm_proxy_url:
            logger.warning(
                "LLM classifier enabled but not configured (missing ORCHESTRATOR_LITELLM_PROXY_URL)."
            )
            return

        try:
            from openai import OpenAI

            base_url = settings.litellm_proxy_url.rstrip("/")
            self._client = OpenAI(
                base_url=base_url,
                api_key=settings.litellm_proxy_api_key or "#unused-when-auth-via-bearer",
                timeout=settings.llm_timeout_seconds,
            )
        except Exception:
            logger.exception("Failed to initialize LiteLLM proxy client for LLM classifier.")
            self._client = None

    def _resolve_auth_headers(self) -> dict[str, str]:
        """Build per-request auth headers for standalone LiteLLM proxy calls."""
        headers: dict[str, str] = {}
        if self.settings.litellm_proxy_api_key:
            headers["x-api-key"] = self.settings.litellm_proxy_api_key

        static_bearer = self.settings.litellm_proxy_bearer_token
        if static_bearer:
            headers["Authorization"] = f"Bearer {static_bearer}"

        return headers

    def classify_with_llm(
        self,
        messages: list[dict[str, str]],
        allowed_categories: list[str] | None = None,
        server_context: list[dict[str, object]] | None = None,
    ) -> dict[str, tuple[float, str]]:
        """Classify conversation intent into categories via LiteLLM Responses API.

        Returns a map of ``category -> (confidence, rationale)``. If anything fails,
        the method returns an empty result so callers can fall back to keyword scoring.
        """
        if not self._enabled or self._client is None or not self.settings.llm_model:
            return {}

        candidate_categories = [category.strip().lower() for category in (allowed_categories or []) if category.strip()]
        if "general" not in candidate_categories:
            candidate_categories.append("general")

        transcript_lines = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            transcript_lines.append(f"{role}: {content}")

        if not transcript_lines:
            return {"general": (1.0, "Empty message context")}

        system_prompt = (
            "You classify user intent into one or more categories. "
            "Return ONLY a valid JSON object (no markdown, no prose). "
            "Use one of these exact shapes:\n"
            "{\"category\": \"general\", \"confidence\": 1.0, \"rationale\": \"brief reason\"}\n"
            "or\n"
            "{\"categories\": ["
            "{\"category\": \"general\", \"confidence\": 1.0, \"rationale\": \"brief reason\"}"
            "]}. "
            f"Only use one of these categories: {', '.join(candidate_categories)}. "
            f"Return at most {MAX_LLM_CATEGORIES} categories, sorted by confidence descending. "
            f"Each rationale must be <= {MAX_LLM_RATIONALE_CHARS} characters. "
            "Confidence must be a number between 0 and 1. "
            "If uncertain, choose general."
        )

        server_context_json = "[]"
        if server_context:
            try:
                server_context_json = json.dumps(server_context, ensure_ascii=False)
            except Exception:
                server_context_json = "[]"

        try:
            completion = self._client.responses.create(
                model=self.settings.llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Available MCP server context: {server_context_json}\n"
                            f"Conversation:\n{'\n'.join(transcript_lines)}"
                        ),
                    },
                ],
                temperature=0,
                max_output_tokens=260,
                text={"format": {"type": "json_object"}},
                extra_headers=self._resolve_auth_headers(),
            )
            content = (_extract_response_text(completion) or "{}").strip()
            parsed = _try_parse_json_object(content)
            if parsed is None:
                logger.warning("LLM classifier returned non-JSON payload; using keyword fallback.")
                return {}
            categories_payload = parsed.get("categories")
            if isinstance(categories_payload, list):
                results: dict[str, tuple[float, str]] = {}
                for entry in categories_payload:
                    if not isinstance(entry, dict):
                        continue
                    category = str(entry.get("category", "general")).strip().lower()
                    confidence = float(entry.get("confidence", 0.0))
                    rationale = str(entry.get("rationale", ""))[:MAX_LLM_RATIONALE_CHARS]
                    if category == "generic":
                        category = "general"
                    if category not in set(candidate_categories):
                        category = "general"
                    confidence = max(0.0, min(1.0, confidence))
                    results[category] = (confidence, rationale)
                    if len(results) >= MAX_LLM_CATEGORIES:
                        break
                return results

            category = str(parsed.get("category", "general")).strip().lower()
            confidence = float(parsed.get("confidence", 0.0))
            rationale = str(parsed.get("rationale", ""))[:MAX_LLM_RATIONALE_CHARS]
            if category == "generic":
                category = "general"
            if category not in set(candidate_categories):
                category = "general"
            confidence = max(0.0, min(1.0, confidence))
            return {category: (confidence, rationale)}
        except Exception:
            logger.exception("LLM classification failed. Falling back to keyword classifier.")
            return {}


def normalize_text(text: str) -> str:
    """Normalize free text for keyword matching and tokenization."""
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def simple_stem(token: str) -> str:
    """Apply a light stemmer tuned for routing keywords."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text and apply lightweight stemming."""
    tokens = WORD_PATTERN.findall(normalize_text(text))
    return [simple_stem(token) for token in tokens]


def _match_keyword(normalized_text: str, keyword: str) -> bool:
    """Return True if a keyword or phrase appears in normalized text."""
    key = normalize_text(keyword)
    if not key:
        return False
    if key in normalized_text:
        return True
    phrase_tokens = [simple_stem(token) for token in WORD_PATTERN.findall(key)]
    if len(phrase_tokens) <= 1:
        return False
    text_tokens = set(tokenize(normalized_text))
    return all(token in text_tokens for token in phrase_tokens)


def _recency_factor(index: int, total: int) -> float:
    """Bias scoring toward newer conversation turns."""
    if total <= 1:
        return 1.0
    relative = index / (total - 1)
    return 0.8 + (0.7 * relative)


def _confidence_from_score(score: float, density: float) -> float:
    """Convert weighted keyword evidence into calibrated confidence."""
    blended = (0.85 * score) + (0.15 * min(1.0, density * 10))
    confidence = 1 / (1 + math.exp(-((blended * 3.5) - 1.5)))
    return max(0.0, min(1.0, confidence))


class KeywordClassifier:
    def __init__(
        self,
        settings: OrchestratorSettings,
        llm_plugin: LlmClassifierPlugin | None = None,
    ):
        self.settings = settings
        self.llm_plugin = llm_plugin or LlmClassifierPlugin(settings=settings)

    def _collect_registry_categories(self, registry: RegistryModel) -> list[str]:
        """Collect normalized categories available for classification."""
        categories: set[str] = set()
        for server in registry.mcp_servers:
            for category in server.categories:
                normalized = category.strip().lower()
                if normalized:
                    categories.add(normalized)
        categories.add("general")
        return sorted(categories)

    def _llm_category_scores(
        self,
        messages: list[dict[str, str]],
        registry: RegistryModel,
    ) -> dict[str, tuple[float, str]] | None:
        """Ask the LLM for category scores constrained by registry categories."""
        if not self.settings.enable_llm_classifier:
            return None

        combined_text = "\n".join(message.get("content", "") for message in messages)
        normalized_text = normalize_text(combined_text)

        allowed_categories = self._collect_registry_categories(registry)
        server_context = [
            {
                "id": server.id,
                "description": server.description,
                "categories": server.categories,
                "tools": server.tools,
                "keywords": server.keywords[:10],
                "matched_keywords": [
                    keyword
                    for keyword in server.keywords
                    if _match_keyword(normalized_text, keyword)
                ][:10],
            }
            for server in registry.mcp_servers
        ]
        try:
            llm_scores = self.llm_plugin.classify_with_llm(messages, allowed_categories, server_context)
        except TypeError:
            llm_scores = self.llm_plugin.classify_with_llm(messages)

        if not llm_scores:
            return None

        return {
            category.strip().lower(): (float(confidence), rationale)
            for category, (confidence, rationale) in llm_scores.items()
        }

    def _score_servers_from_llm_category(
        self,
        registry: RegistryModel,
        category: str,
        llm_confidence: float,
    ) -> list[ServerScore]:
        """Translate one LLM category decision into ranked server scores.

        This phase keeps LLM output bounded by server metadata by only scoring
        servers that actually declare the selected category.
        """
        normalized_category = resolve_alias(category, registry.category_aliases)
        if normalized_category in {"generic", "general"}:
            return []

        matched_servers: list[RegistryServer] = []
        for server in registry.mcp_servers:
            server_categories = {value.strip().lower() for value in server.categories}
            if normalized_category in server_categories:
                matched_servers.append(server)

        if not matched_servers:
            return []

        max_weight = max((server.weight for server in matched_servers), default=1.0)
        if max_weight <= 0:
            max_weight = 1.0

        scores: list[ServerScore] = []
        for server in matched_servers:
            weight_factor = server.weight / max_weight
            confidence = max(0.0, min(1.0, (0.85 * llm_confidence) + (0.15 * weight_factor)))
            scores.append(
                ServerScore(
                    server=server,
                    category=normalized_category,
                    confidence=confidence,
                    matched_keywords=[f"llm:{normalized_category}"],
                    raw_score=weight_factor,
                    density=0.0,
                    classification_method="ai",
                )
            )

        scores.sort(key=lambda item: (item.confidence, item.server.weight), reverse=True)
        return scores

    def _score_servers_from_llm_categories(
        self,
        registry: RegistryModel,
        llm_scores: dict[str, tuple[float, str]],
    ) -> list[ServerScore]:
        """Score servers for every LLM-selected category above confidence cutoff."""
        scores: list[ServerScore] = []
        if not llm_scores:
            return scores

        tie_delta = 0.1
        top_confidence = max(confidence for confidence, _ in llm_scores.values())

        for category, (confidence, _rationale) in llm_scores.items():
            if confidence < self.settings.min_confidence:
                continue
            if top_confidence - confidence > tie_delta:
                continue
            scores.extend(self._score_servers_from_llm_category(registry, category, confidence))

        scores.sort(key=lambda item: (item.confidence, item.server.weight), reverse=True)
        return scores

    def _truncate_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Apply hard limits on context length before classification.

        Guards both message count and aggregate character volume so classifier
        performance remains predictable across large prompts.
        """
        trimmed = messages[-self.settings.max_messages :]
        safe_messages: list[dict[str, str]] = []
        total_chars = 0
        for message in trimmed:
            content = (message.get("content") or "")[: self.settings.max_message_chars]
            content_len = len(content)
            if total_chars + content_len > self.settings.max_total_chars:
                break
            total_chars += content_len
            safe_messages.append({"role": message.get("role", "user"), "content": content})
        return safe_messages

    def score_servers(
        self,
        messages: list[dict[str, str]],
        registry: RegistryModel,
    ) -> list[ServerScore]:
        """Rank candidate MCP servers from conversation context.

        Strategy:
        1) Try LLM-based single-category routing.
        2) If confidence is low or unavailable, run deterministic keyword scoring.
        3) Convert raw scores to calibrated confidence and sort descending.
        """
        scoped_messages = self._truncate_messages(messages)
        if not scoped_messages:
            return []

        llm_scores = self._llm_category_scores(scoped_messages, registry)
        if llm_scores:
            llm_server_scores = self._score_servers_from_llm_categories(registry, llm_scores)
            if llm_server_scores:
                return llm_server_scores

        per_server_score: dict[str, float] = {}
        per_server_keywords: dict[str, list[str]] = {}
        token_count = 0

        for idx, message in enumerate(scoped_messages):
            normalized = normalize_text(message.get("content", ""))
            tokens = tokenize(normalized)
            token_count += len(tokens)
            recency = _recency_factor(idx, len(scoped_messages))

            for server in registry.mcp_servers:
                for keyword in server.keywords:
                    # Keyword hits contribute recency-weighted evidence per server.
                    if _match_keyword(normalized, keyword):
                        per_server_score[server.id] = per_server_score.get(server.id, 0.0) + recency
                        seen = per_server_keywords.setdefault(server.id, [])
                        if keyword not in seen:
                            seen.append(keyword)

        if not per_server_score:
            return []

        max_raw = max(per_server_score.values()) if per_server_score else 1.0
        if max_raw <= 0:
            max_raw = 1.0
        token_count = max(1, token_count)

        by_category: dict[str, float] = {}
        for server in registry.mcp_servers:
            raw = per_server_score.get(server.id, 0.0)
            if raw <= 0:
                continue
            category = server.categories[0].lower() if server.categories else "general"
            # Track strongest server per category to avoid over-fragmented rankings.
            by_category[category] = max(by_category.get(category, 0.0), raw / max_raw)

        scores: list[ServerScore] = []
        for server in registry.mcp_servers:
            raw = per_server_score.get(server.id, 0.0)
            if raw <= 0:
                continue

            normalized_score = (raw / max_raw) * server.weight
            density = len(per_server_keywords.get(server.id, [])) / token_count
            category = server.categories[0].lower() if server.categories else "general"
            category_bonus = by_category.get(category, 0.0)
            confidence = _confidence_from_score(normalized_score, density)
            # Blend server-specific evidence with category-level consensus.
            confidence = max(0.0, min(1.0, (0.8 * confidence) + (0.2 * category_bonus)))

            scores.append(
                ServerScore(
                    server=server,
                    category=category,
                    confidence=confidence,
                    matched_keywords=per_server_keywords.get(server.id, []),
                    raw_score=normalized_score,
                    density=density,
                    classification_method="keyword",
                )
            )

        scores.sort(key=lambda item: (item.confidence, item.server.weight, item.density), reverse=True)
        return scores

    def classify_categories(
        self,
        messages: list[dict[str, str]],
        registry: RegistryModel,
    ) -> list[CategoryMatch]:
        """Aggregate server scores into top category matches."""
        server_scores = self.score_servers(messages, registry)
        if not server_scores:
            return []

        best_by_category: dict[str, ServerScore] = {}
        for item in server_scores:
            existing = best_by_category.get(item.category)
            if existing is None or item.confidence > existing.confidence:
                best_by_category[item.category] = item

        categories = [
            CategoryMatch(
                name=category,
                confidence=score.confidence,
                matched_keywords=score.matched_keywords,
                raw_score=score.raw_score,
                classification_method=score.classification_method,
            )
            for category, score in best_by_category.items()
        ]
        categories.sort(key=lambda item: item.confidence, reverse=True)
        return categories


def resolve_alias(term: str, aliases: dict[str, str]) -> str:
    """Normalize and map category aliases to canonical category names."""
    normalized = term.strip().lower()
    return aliases.get(normalized, normalized)