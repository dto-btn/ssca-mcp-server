from __future__ import annotations

import json
import math
import os
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
        self.settings = settings
        self._client = None
        self._enabled = settings.enable_llm_classifier
        if not self._enabled:
            return
        if not settings.azure_openai_endpoint or not settings.llm_model:
            logger.warning(
                "LLM classifier enabled but not configured (missing AZURE_OPENAI_ENDPOINT or ORCHESTRATOR_LLM_MODEL)."
            )
            return

        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI

            tenant_hint = os.getenv("AZURE_TENANT_ID") or os.getenv("AZURE_AD_TENANT_ID")
            client_hint = os.getenv("AZURE_CLIENT_ID") or os.getenv("AZURE_AD_CLIENT_ID")
            if client_hint and client_hint.startswith("api://"):
                client_hint = client_hint.removeprefix("api://")

            credential_kwargs: dict[str, object] = {}
            if tenant_hint:
                credential_kwargs["shared_cache_tenant_id"] = tenant_hint
                credential_kwargs["interactive_browser_tenant_id"] = tenant_hint
            if client_hint and re.fullmatch(r"[0-9a-fA-F-]{32,36}", client_hint):
                credential_kwargs["managed_identity_client_id"] = client_hint

            credential = DefaultAzureCredential(**credential_kwargs)

            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )

            self._client = AzureOpenAI(
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
                azure_ad_token_provider=token_provider,
                timeout=settings.llm_timeout_seconds,
            )
        except Exception:
            logger.exception("Failed to initialize Azure OpenAI client for LLM classifier.")
            self._client = None

    def classify_with_llm(
        self,
        messages: list[dict[str, str]],
        allowed_categories: list[str] | None = None,
        server_context: list[dict[str, object]] | None = None,
    ) -> dict[str, tuple[float, str]]:
        if not self._enabled or self._client is None or not self.settings.llm_model:
            return {}

        candidate_categories = [category.strip().lower() for category in (allowed_categories or []) if category.strip()]
        if "generic" not in candidate_categories:
            candidate_categories.append("generic")

        transcript_lines = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            transcript_lines.append(f"{role}: {content}")

        if not transcript_lines:
            return {"generic": (1.0, "Empty message context")}

        system_prompt = (
            "You classify user intent into exactly one category. "
            "Return strict JSON with keys: category (string), confidence (number 0..1), rationale (string). "
            f"Only use one of these categories: {', '.join(candidate_categories)}. "
            "If uncertain, choose generic."
        )

        server_context_json = "[]"
        if server_context:
            try:
                server_context_json = json.dumps(server_context, ensure_ascii=False)
            except Exception:
                server_context_json = "[]"

        try:
            completion = self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
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
                max_tokens=140,
                response_format={"type": "json_object"},
            )
            content = (completion.choices[0].message.content or "{}").strip()
            parsed = json.loads(content)
            category = str(parsed.get("category", "generic")).strip().lower()
            confidence = float(parsed.get("confidence", 0.0))
            rationale = str(parsed.get("rationale", ""))
            if category not in set(candidate_categories):
                category = "generic"
            confidence = max(0.0, min(1.0, confidence))
            return {category: (confidence, rationale)}
        except Exception:
            logger.exception("LLM classification failed. Falling back to keyword classifier.")
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
        self.llm_plugin = llm_plugin or LlmClassifierPlugin(settings=settings)

    def _collect_registry_categories(self, registry: RegistryModel) -> list[str]:
        categories: set[str] = set()
        for server in registry.mcp_servers:
            for category in server.categories:
                normalized = category.strip().lower()
                if normalized:
                    categories.add(normalized)
        categories.add("generic")
        return sorted(categories)

    def _llm_top_category(
        self,
        messages: list[dict[str, str]],
        registry: RegistryModel,
    ) -> tuple[str, float, str] | None:
        if not self.settings.enable_llm_classifier:
            return None

        allowed_categories = self._collect_registry_categories(registry)
        server_context = [
            {
                "id": server.id,
                "description": server.description,
                "categories": server.categories,
                "tools": server.tools,
                "keywords": server.keywords[:10],
            }
            for server in registry.mcp_servers
        ]
        try:
            llm_scores = self.llm_plugin.classify_with_llm(messages, allowed_categories, server_context)
        except TypeError:
            llm_scores = self.llm_plugin.classify_with_llm(messages)

        if not llm_scores:
            return None

        category, (confidence, rationale) = max(llm_scores.items(), key=lambda item: item[1][0])
        return category.strip().lower(), float(confidence), rationale

    def _score_servers_from_llm_category(
        self,
        registry: RegistryModel,
        category: str,
        llm_confidence: float,
    ) -> list[ServerScore]:
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

        llm_choice = self._llm_top_category(scoped_messages, registry)
        if llm_choice:
            llm_category, llm_confidence, _llm_rationale = llm_choice
            llm_scores = self._score_servers_from_llm_category(registry, llm_category, llm_confidence)
            if llm_scores and llm_confidence >= self.settings.min_confidence:
                return llm_scores

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
    normalized = term.strip().lower()
    return aliases.get(normalized, normalized)