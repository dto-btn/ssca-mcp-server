from __future__ import annotations

import json
from pathlib import Path

from server.classifier import KeywordClassifier, LlmClassifierPlugin, resolve_alias
from server.config import OrchestratorSettings
from server.registry import RegistryStore
from server.router import OrchestratorRouter


def make_settings(registry_path: Path, *, hot_reload: bool = False) -> OrchestratorSettings:
    return OrchestratorSettings(
        registry_path=registry_path,
        max_messages=10,
        min_confidence=0.4,
        enable_llm_classifier=False,
        llm_blend_alpha=0.35,
        azure_openai_endpoint=None,
        azure_openai_api_version="2024-05-01-preview",
        llm_model=None,
        llm_timeout_seconds=8.0,
        verbose_logging=False,
        redact_sensitive_tokens=True,
        max_message_chars=4000,
        max_total_chars=20000,
        enable_hot_reload=hot_reload,
        update_registry_enabled=True,
        admin_secret="secret",
    )


def write_registry(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sample_registry_payload() -> dict:
    return {
        "version": "1.0",
        "mcp_servers": [
            {
                "id": "web_search_mcp",
                "endpoint": "https://web-search-mcp.example.com/mcp",
                "categories": ["web-search", "news", "research"],
                "tools": ["search", "get_page"],
                "keywords": ["search", "web", "news", "article", "research", "find online"],
                "weight": 1.0,
            },
            {
                "id": "db_mcp",
                "endpoint": "https://db-mcp.example.com/mcp",
                "categories": ["database", "sql", "data-admin"],
                "tools": ["query", "schema"],
                "keywords": ["sql", "database", "table", "query", "postgres", "mysql", "join", "schema"],
                "weight": 1.0,
            },
            {
                "id": "calendar_mcp",
                "endpoint": "https://calendar-mcp.example.com/mcp",
                "categories": ["calendar", "scheduling"],
                "tools": ["create_event", "list_events"],
                "keywords": ["schedule", "meeting", "calendar", "appointment", "invite", "event"],
                "weight": 1.0,
            },
        ],
        "category_aliases": {
            "web": "web-search",
            "db": "database",
            "meet": "scheduling",
        },
        "routing_rules": {
            "max_recommendations": 3,
            "tie_breaker": "weight_then_keyword_density",
            "default_fallback": {
                "category": "generic",
                "message": "No clear match. Ask a clarifying question.",
            },
        },
    }


def make_router(tmp_path: Path, *, hot_reload: bool = False) -> tuple[OrchestratorRouter, RegistryStore, Path]:
    reg_path = tmp_path / "registry.json"
    write_registry(reg_path, sample_registry_payload())
    settings = make_settings(reg_path, hot_reload=hot_reload)
    store = RegistryStore(settings)
    router = OrchestratorRouter(settings=settings, registry_store=store)
    return router, store, reg_path


def msg(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_keyword_matching_and_scoring(tmp_path: Path) -> None:
    router, store, _ = make_router(tmp_path)
    classifier = KeywordClassifier(store.settings)
    registry = store.load_registry()

    scores = classifier.score_servers(msg("Run an SQL query on postgres users table"), registry)
    assert scores
    assert scores[0].server.id == "db_mcp"
    assert scores[0].confidence > 0.6


def test_alias_resolution() -> None:
    aliases = {"db": "database", "web": "web-search"}
    assert resolve_alias("db", aliases) == "database"
    assert resolve_alias("WEB", aliases) == "web-search"
    assert resolve_alias("calendar", aliases) == "calendar"


def test_confidence_calculation_bounds(tmp_path: Path) -> None:
    router, store, _ = make_router(tmp_path)
    classifier = KeywordClassifier(store.settings)
    registry = store.load_registry()
    scores = classifier.score_servers(msg("search web news article research"), registry)
    assert scores
    assert all(0.0 <= score.confidence <= 1.0 for score in scores)


def test_tie_breaking_and_weighting(tmp_path: Path) -> None:
    router, store, reg_path = make_router(tmp_path)
    payload = sample_registry_payload()
    payload["mcp_servers"][0]["weight"] = 1.2
    payload["mcp_servers"][1]["weight"] = 0.9
    write_registry(reg_path, payload)

    response = router.suggest_route(msg("search query news database"), max_recommendations=3)
    recos = response["recommendations"]
    assert recos
    assert recos[0]["mcp_server_id"] == "web_search_mcp"


def test_registry_validation_error_handling(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_registry.json"
    bad_path.write_text('{"version":"1.0","mcp_servers":[{"id":"x"}]}', encoding="utf-8")
    settings = make_settings(bad_path)
    store = RegistryStore(settings)
    router = OrchestratorRouter(settings=settings, registry_store=store)

    result = router.suggest_route(msg("search web"))
    assert "error" in result
    assert result["fallback"]["reason"] == "Routing failed due to a server-side error."


def test_empty_registry_fallback(tmp_path: Path) -> None:
    reg_path = tmp_path / "empty_registry.json"
    write_registry(
        reg_path,
        {
            "version": "1.0",
            "mcp_servers": [],
            "category_aliases": {},
            "routing_rules": {
                "max_recommendations": 3,
                "tie_breaker": "weight_then_keyword_density",
                "default_fallback": {
                    "category": "generic",
                    "message": "No clear match. Ask a clarifying question.",
                },
            },
        },
    )
    settings = make_settings(reg_path)
    store = RegistryStore(settings)
    router = OrchestratorRouter(settings=settings, registry_store=store)

    result = router.suggest_route(msg("anything"))
    assert result["recommendations"] == []
    assert "fallback" in result


def test_single_strong_match_schedule_prompt(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("Please schedule a meeting for Thursday afternoon with the team."))
    assert result["recommendations"][0]["mcp_server_id"] == "calendar_mcp"


def test_prompt_sql_users_last_month(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("Run an SQL query to list users who signed up last month."))
    assert result["recommendations"][0]["mcp_server_id"] == "db_mcp"


def test_prompt_recent_articles_ev_canada(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("Find recent articles about electric vehicles in Canada."))
    assert result["recommendations"][0]["mcp_server_id"] == "web_search_mcp"


def test_prompt_create_event_budget_review(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("Create an event called Budget Review next Monday at 10 AM."))
    assert result["recommendations"][0]["mcp_server_id"] == "calendar_mcp"


def test_prompt_schema_customers_postgres(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("What is the schema of the customers table in Postgres?"))
    assert result["recommendations"][0]["mcp_server_id"] == "db_mcp"


def test_prompt_find_data_online_renewable_energy(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("I need to find data online about renewable energy trends."))
    assert result["recommendations"][0]["mcp_server_id"] == "web_search_mcp"


def test_prompt_with_eps_routes_to_pmcoe(tmp_path: Path) -> None:
    reg_path = tmp_path / "pmcoe_registry.json"
    write_registry(
        reg_path,
        {
            "version": "1.0",
            "mcp_servers": [
                {
                    "id": "pmcoe_mcp",
                    "endpoint": "https://pmcoe-mcp.example.com/mcp",
                    "categories": ["pmcoe", "project", "project-management"],
                    "tools": ["search_pmcoe_resources"],
                    "keywords": ["pmcoe", "EPS", "project management"],
                    "weight": 1.0,
                },
                {
                    "id": "corporate_mcp",
                    "endpoint": "https://corporate-mcp.example.com/mcp",
                    "categories": ["corporate", "myssc"],
                    "tools": ["search_policy"],
                    "keywords": ["policy", "intranet"],
                    "weight": 1.0,
                },
            ],
            "category_aliases": {"project": "pmcoe"},
            "routing_rules": {
                "max_recommendations": 3,
                "tie_breaker": "weight_then_keyword_density",
                "default_fallback": {
                    "category": "generic",
                    "message": "No clear match. Ask a clarifying question.",
                },
            },
        },
    )

    settings = make_settings(reg_path)
    store = RegistryStore(settings)
    router = OrchestratorRouter(settings=settings, registry_store=store)

    result = router.suggest_route(msg("I need EPS guidance for a new project."))

    assert result["recommendations"]
    assert result["recommendations"][0]["mcp_server_id"] == "pmcoe_mcp"

    lower_result = router.suggest_route(msg("I need eps guidance for a new project."))

    assert lower_result["recommendations"]
    assert lower_result["recommendations"][0]["mcp_server_id"] == "pmcoe_mcp"


def test_multiple_near_ties_ranked_output(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("Search web resources and query table schema to compare news data."), max_recommendations=3)
    assert len(result["recommendations"]) >= 2


def test_low_confidence_single_best_triggers_disambiguation(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("help me maybe do something"), require_single_best=True)
    if result["recommendations"]:
        assert "disambiguation_note" in result or result["recommendations"][0]["confidence"] >= 0.6
    else:
        assert "fallback" in result


def test_unmatched_prompt_returns_generic_with_no_upstream(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.suggest_route(msg("zzzz qqqq yyyyy"), require_single_best=True)
    assert result["recommendations"] == []
    assert result["fallback"]["category"] == "generic"
    assert result["fallback"]["upstream"] is None


def test_hot_reload_registry(tmp_path: Path) -> None:
    router, store, reg_path = make_router(tmp_path, hot_reload=True)
    first = router.suggest_route(msg("schedule a meeting"))
    assert first["recommendations"][0]["mcp_server_id"] == "calendar_mcp"

    payload = sample_registry_payload()
    payload["mcp_servers"][2]["keywords"] = ["planner"]
    write_registry(reg_path, payload)
    second = router.suggest_route(msg("schedule a meeting"))

    if second["recommendations"]:
        assert second["recommendations"][0]["mcp_server_id"] != "calendar_mcp"
    else:
        assert "fallback" in second


class StubLlmPlugin(LlmClassifierPlugin):
    def __init__(self, result: dict[str, tuple[float, str]]):
        self.result = result

    def classify_with_llm(
        self,
        messages: list[dict[str, str]],
        allowed_categories: list[str] | None = None,
    ) -> dict[str, tuple[float, str]]:
        return self.result


class CountingStubLlmPlugin(LlmClassifierPlugin):
    def __init__(self, result: dict[str, tuple[float, str]]):
        self.result = result
        self.calls = 0

    def classify_with_llm(
        self,
        messages: list[dict[str, str]],
        allowed_categories: list[str] | None = None,
    ) -> dict[str, tuple[float, str]]:
        self.calls += 1
        return self.result


def test_llm_first_pass_selects_category_without_keywords(tmp_path: Path) -> None:
    _, store, _ = make_router(tmp_path)
    llm_settings = make_settings(store.settings.registry_path)
    llm_settings = OrchestratorSettings(**{**llm_settings.__dict__, "enable_llm_classifier": True})

    classifier = KeywordClassifier(
        llm_settings,
        llm_plugin=StubLlmPlugin({"database": (0.91, "Detected SQL task")}),
    )
    registry = store.load_registry()

    scores = classifier.score_servers(msg("please help"), registry)
    assert scores
    assert scores[0].server.id == "db_mcp"
    assert scores[0].matched_keywords == ["llm:database"]


def test_llm_generic_falls_back_to_keyword_scoring(tmp_path: Path) -> None:
    _, store, _ = make_router(tmp_path)
    llm_settings = make_settings(store.settings.registry_path)
    llm_settings = OrchestratorSettings(**{**llm_settings.__dict__, "enable_llm_classifier": True})

    classifier = KeywordClassifier(
        llm_settings,
        llm_plugin=StubLlmPlugin({"generic": (0.96, "Insufficient context")}),
    )
    registry = store.load_registry()

    scores = classifier.score_servers(msg("Please schedule a meeting tomorrow at 10"), registry)
    assert scores
    assert scores[0].server.id == "calendar_mcp"
    assert all(not keyword.startswith("llm:") for keyword in scores[0].matched_keywords)


def test_classify_and_suggest_returns_category_and_routes(tmp_path: Path) -> None:
    router, _, _ = make_router(tmp_path)
    result = router.classify_and_suggest(msg("Run an SQL query to list users"), max_recommendations=3)

    assert "categories" in result
    assert "recommendations" in result
    assert result["recommendations"]
    assert result["recommendations"][0]["mcp_server_id"] == "db_mcp"


def test_classify_and_suggest_uses_single_llm_call(tmp_path: Path) -> None:
    _, store, _ = make_router(tmp_path)
    llm_settings = make_settings(store.settings.registry_path)
    llm_settings = OrchestratorSettings(**{**llm_settings.__dict__, "enable_llm_classifier": True})

    counting_stub = CountingStubLlmPlugin({"database": (0.9, "Detected SQL task")})
    classifier = KeywordClassifier(llm_settings, llm_plugin=counting_stub)
    router = OrchestratorRouter(settings=llm_settings, registry_store=store)
    router.classifier = classifier

    result = router.classify_and_suggest(msg("Please help with SQL"), max_recommendations=3)

    assert "recommendations" in result
    assert counting_stub.calls == 1