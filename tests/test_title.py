"""Unit tests for deterministic chat title synthesis."""

from __future__ import annotations

from server.title import DEFAULT_CHAT_TITLE, ChatTitleGenerator


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_sanitize_strips_markdown_and_collapses_whitespace() -> None:
    generator = ChatTitleGenerator()
    assert generator.sanitize("  **Hello**\n\n  world  ") == "Hello world"


def test_sanitize_truncates_to_max_chars() -> None:
    generator = ChatTitleGenerator()
    assert len(generator.sanitize("word " * 40)) <= 80


def test_sanitize_returns_empty_for_punctuation_only_input() -> None:
    generator = ChatTitleGenerator()
    assert generator.sanitize("!!!") == ""


def test_is_safe_enforces_word_count_bounds() -> None:
    generator = ChatTitleGenerator()
    assert generator.is_safe("Formal Email Rewrite")
    assert not generator.is_safe("Solo")
    assert not generator.is_safe("One Two Three Four Five Six")


def test_is_safe_rejects_empty_and_punctuation_only() -> None:
    generator = ChatTitleGenerator()
    assert not generator.is_safe("")
    assert not generator.is_safe("!!!")


def test_format_for_display_preserves_short_acronyms() -> None:
    generator = ChatTitleGenerator()
    assert generator.format_for_display("SQL query help") == "SQL Query Help"


def test_from_user_message_uses_style_object_action_pattern() -> None:
    generator = ChatTitleGenerator()
    title = generator.from_user_message("Help me rewrite an email so it is more formal")
    assert title == "Formal Email Rewrite"


def test_from_user_message_uses_object_action_pattern() -> None:
    generator = ChatTitleGenerator()
    assert generator.from_user_message("Could you draft a report for the team?") == "Report Draft"


def test_from_user_message_drops_stopwords_when_no_pattern_matches() -> None:
    generator = ChatTitleGenerator()
    title = generator.from_user_message("What are the vacation carryover rules")
    assert title == "Vacation Carryover Rules"


def test_from_user_message_returns_empty_for_unusable_input() -> None:
    generator = ChatTitleGenerator()
    assert generator.from_user_message("") == ""
    assert generator.from_user_message("!!!") == ""


def test_generate_prefers_usable_llm_title() -> None:
    generator = ChatTitleGenerator()
    title, source = generator.generate(user("anything at all"), [], "Custom Chat Title")
    assert (title, source) == ("Custom Chat Title", "ai")


def test_generate_falls_back_when_llm_title_is_unusable() -> None:
    generator = ChatTitleGenerator()
    title, source = generator.generate(
        user("Please rewrite an email so it is more formal"),
        [],
        "!!!",
    )
    assert (title, source) == ("Formal Email Rewrite", "deterministic")


def test_generate_falls_back_to_top_category() -> None:
    generator = ChatTitleGenerator()
    title, source = generator.generate(user("!!!"), [{"name": "web_search"}], None)
    assert (title, source) == ("Web Search Request", "deterministic")


def test_generate_uses_non_user_message_as_last_resort() -> None:
    generator = ChatTitleGenerator()
    messages = [{"role": "assistant", "content": "Please draft a report"}]
    title, source = generator.generate(messages, [], None)
    assert (title, source) == ("Report Draft", "deterministic")


def test_generate_returns_default_without_content() -> None:
    generator = ChatTitleGenerator()
    assert generator.generate([], [], None) == (DEFAULT_CHAT_TITLE, "deterministic")
