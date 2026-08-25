"""Chat title synthesis with a deterministic fallback.

Kept separate from routing so the heuristics can be unit-tested directly.
"""

from __future__ import annotations

import re

TITLE_MIN_WORDS = 2
TITLE_MAX_WORDS = 5
TITLE_MAX_CHARS = 80
DEFAULT_CHAT_TITLE = "General Request"

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


class ChatTitleGenerator:
    """Produce short chat titles from an LLM candidate or deterministic heuristics."""

    def sanitize(self, value: str) -> str:
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

    def is_safe(self, title: str) -> bool:
        """Validate title contract constraints for frontend auto-rename."""
        if not title:
            return False
        if len(title) > TITLE_MAX_CHARS:
            return False
        if _TITLE_PUNCT_ONLY_PATTERN.match(title):
            return False
        words = _TITLE_TOKEN_PATTERN.findall(title)
        return TITLE_MIN_WORDS <= len(words) <= TITLE_MAX_WORDS

    def format_for_display(self, title: str) -> str:
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

    def from_user_message(self, latest_user_message: str) -> str:
        """Derive a deterministic short title from the latest user prompt."""
        sanitized = self.sanitize(latest_user_message)
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
            pattern_title = self.format_for_display(f"{style} {obj} {action_label}")
            if self.is_safe(pattern_title):
                return pattern_title

        if obj and action:
            action_label = _TITLE_ACTION_LABELS.get(action, action.title())
            pattern_title = self.format_for_display(f"{obj} {action_label}")
            if self.is_safe(pattern_title):
                return pattern_title

        all_tokens = _TITLE_TOKEN_PATTERN.findall(clause)
        informative_tokens = [token for token in all_tokens if token.lower() not in _TITLE_STOPWORDS]

        chosen = informative_tokens if len(informative_tokens) >= TITLE_MIN_WORDS else all_tokens
        if len(chosen) < TITLE_MIN_WORDS:
            return ""

        title = " ".join(chosen[:TITLE_MAX_WORDS])
        title = self.sanitize(title)
        return self.format_for_display(title)

    def generate(
        self,
        messages: list[dict[str, str]],
        categories_data: list[dict[str, object]],
        llm_title: str | None = None,
    ) -> tuple[str, str]:
        """Generate a chat title and the source that produced it.

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
            candidate = self.format_for_display(self.sanitize(llm_title))
            if self.is_safe(candidate):
                return candidate, "ai"

        fallback_title = self.from_user_message(latest_user_message)
        if self.is_safe(fallback_title):
            return fallback_title, "deterministic"

        if categories_data:
            category_name = str(categories_data[0].get("name", "")).strip()
            if category_name:
                category_words = category_name.replace("_", " ").replace("-", " ")
                category_title = self.format_for_display(f"{category_words} request")
                if self.is_safe(category_title):
                    return category_title, "deterministic"

        return DEFAULT_CHAT_TITLE, "deterministic"

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
