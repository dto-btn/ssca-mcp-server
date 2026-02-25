from __future__ import annotations

import math
import re
from dataclasses import dataclass

try:
    from .config import OrchestratorSettings
    from .schemas import RegistryModel, RegistryServer
except ImportError:
    from config import OrchestratorSettings
    from schemas import RegistryModel, RegistryServer

WORD_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class CategoryMatch:
    name: str
    confidence: float
    matched_keywords: list[str]
    raw_score: float


@dataclass
class ServerScore:
    server: RegistryServer
    category: str
    confidence: float
    matched_keywords: list[str]
    raw_score: float
    density: float


class LlmClassifierPlugin:
    def classify_with_llm(self, messages: list[dict[str, str]]) -> dict[str, tuple[float, str]]:
        return {}


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def simple_stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = WORD_PATTERN.findall(normalize_text(text))
    return [simple_stem(token) for token in tokens]


def _match_keyword(normalized_text: str, keyword: str) -> bool:
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
    if total <= 1:
        return 1.0
    relative = index / (total - 1)
    return 0.8 + (0.7 * relative)


def _confidence_from_score(score: float, density: float) -> float:
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
        self.llm_plugin = llm_plugin or LlmClassifierPlugin()

    def _truncate_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
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
        scoped_messages = self._truncate_messages(messages)
        if not scoped_messages:
            return []

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
            by_category[category] = max(by_category.get(category, 0.0), raw / max_raw)

        if self.settings.enable_llm_classifier:
            llm_scores = self.llm_plugin.classify_with_llm(scoped_messages)
            alpha = self.settings.llm_blend_alpha
            for category, (llm_conf, _rationale) in llm_scores.items():
                existing = by_category.get(category, 0.0)
                by_category[category] = ((1 - alpha) * existing) + (alpha * llm_conf)

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
            confidence = max(0.0, min(1.0, (0.8 * confidence) + (0.2 * category_bonus)))

            scores.append(
                ServerScore(
                    server=server,
                    category=category,
                    confidence=confidence,
                    matched_keywords=per_server_keywords.get(server.id, []),
                    raw_score=normalized_score,
                    density=density,
                )
            )

        scores.sort(key=lambda item: (item.confidence, item.server.weight, item.density), reverse=True)
        return scores

    def classify_categories(
        self,
        messages: list[dict[str, str]],
        registry: RegistryModel,
    ) -> list[CategoryMatch]:
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
            )
            for category, score in best_by_category.items()
        ]
        categories.sort(key=lambda item: item.confidence, reverse=True)
        return categories


def resolve_alias(term: str, aliases: dict[str, str]) -> str:
    normalized = term.strip().lower()
    return aliases.get(normalized, normalized)