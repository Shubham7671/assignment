"""Tests for the provider abstraction and the Gemini adapter.

As with ``test_llm.py``, every test uses a mock client — nothing here
reaches the network.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from leadqual.config import Settings, load_settings
from leadqual.exceptions import LLMCallError, LLMConfigurationError, LLMResponseError
from leadqual.llm.client import QualificationLLM
from leadqual.llm.providers import (
    PROVIDER_ANTHROPIC,
    PROVIDER_FALLBACK,
    PROVIDER_GEMINI,
    SUPPORTED_PROVIDERS,
    AnthropicProvider,
    GeminiProvider,
    build_provider,
    provider_availability,
)
from leadqual.llm.schema import QUALIFICATION_SCHEMA
from leadqual.nlp.extract import analyse_lead
from leadqual.scoring import score_completeness
from tests.conftest import MOCK_PAYLOAD, MockGeminiClient

LEAD = {
    "lead_id": 1,
    "full_name": "Test Enquirer",
    "email": "test@example.com",
    "source": "web_form",
    "programme_interest": "MSc Data Analytics",
}


def make_context(text: str = "I want an MBA. I can study at weekends. What are the fees?"):
    messages = [
        {"message_id": 1, "body": text, "direction": "inbound",
         "channel": "email", "sent_at": "2026-10-01T10:00:00+00:00"}
    ]
    nlp = analyse_lead(messages)
    return messages, nlp, score_completeness(nlp)


# ==========================================================================
# Defaults
# ==========================================================================
class TestCheapDefaults:
    """The out-of-the-box defaults must stay on the cheapest models.

    This project is run by a student without API credit, so an accidental
    change to a premium default would be a real cost. Changing a default is
    fine - but it has to be deliberate enough to update this test.
    """

    def test_anthropic_default_is_the_cheapest_claude(self):
        from leadqual.config import DEFAULT_MODEL

        assert DEFAULT_MODEL == "claude-haiku-5-5"

    def test_gemini_default_is_a_free_tier_flash_lite(self):
        from leadqual.config import DEFAULT_GEMINI_MODEL

        assert DEFAULT_GEMINI_MODEL == "gemini-3.5-flash-lite"
        assert "flash-lite" in DEFAULT_GEMINI_MODEL

    def test_defaults_apply_when_nothing_is_configured(self, monkeypatch, tmp_path):
        for var in ("LEADQUAL_MODEL", "LEADQUAL_GEMINI_MODEL", "LEADQUAL_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LEADQUAL_DB_PATH", str(tmp_path / "x.db"))
        resolved = load_settings(env_file=tmp_path / "missing.env")
        assert resolved.model == "claude-haiku-5-5"
        assert resolved.gemini_model == "gemini-3.5-flash-lite"


# ==========================================================================
# Settings / selection
# ==========================================================================
class TestProviderSelection:
    def test_build_provider_honours_the_setting(self, settings, gemini_settings):
        assert isinstance(build_provider(settings), AnthropicProvider)
        assert isinstance(build_provider(gemini_settings), GeminiProvider)

    def test_unknown_provider_is_rejected(self, settings):
        with pytest.raises(LLMConfigurationError, match="Unknown LEADQUAL_PROVIDER"):
            build_provider(replace(settings, provider="llama"))

    def test_supported_providers_all_build(self, settings):
        for name in SUPPORTED_PROVIDERS:
            assert build_provider(replace(settings, provider=name)) is not None

    def test_availability_lists_every_provider(self, settings):
        rows = provider_availability(settings)
        assert {row["provider"] for row in rows} == set(SUPPORTED_PROVIDERS)
        assert all("reason" in row for row in rows)

    @pytest.mark.parametrize(
        "provider,expected_model",
        [("anthropic", "claude-opus-5-5"), ("gemini", "gemini-3.5-flash-lite")],
    )
    def test_active_model_follows_the_provider(self, settings, provider, expected_model):
        chosen = replace(
            settings, provider=provider,
            model="claude-opus-5-5", gemini_model="gemini-3.5-flash-lite",
        )
        assert chosen.active_model == expected_model

    def test_active_key_follows_the_provider(self, settings):
        both = replace(settings, api_key="sk-ant-real", gemini_api_key="AIza-real")
        assert replace(both, provider="anthropic").has_active_key
        assert replace(both, provider="gemini").has_active_key
        no_gemini = replace(both, provider="gemini", gemini_api_key=None)
        assert not no_gemini.has_active_key
        assert no_gemini.has_api_key  # the other key is still there

    def test_placeholder_keys_are_rejected(self, settings):
        placeholder = replace(
            settings,
            api_key="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            gemini_api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        )
        assert not placeholder.has_api_key
        assert not placeholder.has_gemini_key


class TestAutoSelection:
    """With no LEADQUAL_PROVIDER set, pick whichever provider has a key."""

    def test_gemini_key_only_selects_gemini(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LEADQUAL_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-real-looking-key")
        monkeypatch.setenv("LEADQUAL_DB_PATH", str(tmp_path / "x.db"))
        assert load_settings(env_file=tmp_path / "missing.env").provider == "gemini"

    def test_anthropic_key_wins_when_both_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LEADQUAL_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-looking-key")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-real-looking-key")
        monkeypatch.setenv("LEADQUAL_DB_PATH", str(tmp_path / "x.db"))
        assert load_settings(env_file=tmp_path / "missing.env").provider == "anthropic"

    def test_explicit_setting_overrides_the_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LEADQUAL_PROVIDER", "gemini")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-looking-key")
        monkeypatch.setenv("LEADQUAL_DB_PATH", str(tmp_path / "x.db"))
        assert load_settings(env_file=tmp_path / "missing.env").provider == "gemini"

    def test_no_keys_falls_back_to_the_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LEADQUAL_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("LEADQUAL_DB_PATH", str(tmp_path / "x.db"))
        assert load_settings(env_file=tmp_path / "missing.env").provider == "anthropic"


# ==========================================================================
# Gemini adapter
# ==========================================================================
class TestGeminiRouting:
    def test_live_call_records_gemini_metadata(self, gemini_settings, mock_gemini_client):
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(
            gemini_settings, client=mock_gemini_client
        ).generate(LEAD, messages, nlp, score)

        assert metadata.provider == PROVIDER_GEMINI
        assert metadata.is_fallback is False
        assert metadata.model == "gemini-3.5-flash-lite"
        assert metadata.response_id == "gemini_mock_0001"
        assert metadata.stop_reason == "STOP"
        assert metadata.input_tokens == 2048
        assert metadata.output_tokens == 256
        assert metadata.latency_ms is not None and metadata.latency_ms >= 0
        assert payload["qualification_summary"] == MOCK_PAYLOAD["qualification_summary"]

    def test_no_key_uses_the_offline_generator(self, gemini_settings):
        messages, nlp, score = make_context()
        no_key = replace(gemini_settings, gemini_api_key=None)
        _, metadata = QualificationLLM(no_key).generate(LEAD, messages, nlp, score)
        assert metadata.provider == PROVIDER_FALLBACK
        assert metadata.is_fallback is True

    def test_status_names_gemini(self, gemini_settings, mock_gemini_client):
        status = QualificationLLM(gemini_settings, client=mock_gemini_client).status()
        assert status["provider"] == PROVIDER_GEMINI
        assert status["model"] == "gemini-3.5-flash-lite"

    def test_offline_mode_disables_the_live_call(self, gemini_settings):
        offline = replace(gemini_settings, offline=True)
        messages, nlp, score = make_context()
        _, metadata = QualificationLLM(offline).generate(LEAD, messages, nlp, score)
        assert metadata.is_fallback is True
        assert "OFFLINE" in (metadata.note or "").upper()

    def test_missing_key_on_the_live_path_raises(self, gemini_settings):
        broken = replace(gemini_settings, gemini_api_key=None)
        with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
            GeminiProvider(broken)._get_client()


class TestGeminiRequestShape:
    @pytest.fixture
    def request_sent(self, gemini_settings, mock_gemini_client):
        messages, nlp, score = make_context()
        QualificationLLM(gemini_settings, client=mock_gemini_client).generate(
            LEAD, messages, nlp, score
        )
        return mock_gemini_client.calls[0]

    def test_model_is_passed(self, request_sent):
        assert request_sent["model"] == "gemini-3.5-flash-lite"

    def test_user_prompt_is_the_contents(self, request_sent):
        assert "READINESS SCORE" in request_sent["contents"]
        assert "I want an MBA" in request_sent["contents"]

    def test_system_prompt_goes_in_the_config(self, request_sent):
        config = request_sent["config"]
        system = getattr(config, "system_instruction", None)
        assert system and "FAIRNESS CONSTRAINTS" in system

    def test_structured_output_is_requested(self, request_sent):
        config = request_sent["config"]
        assert getattr(config, "response_mime_type", None) == "application/json"
        assert getattr(config, "response_json_schema", None) == QUALIFICATION_SCHEMA

    def test_max_tokens_is_applied(self, request_sent):
        assert getattr(request_sent["config"], "max_output_tokens", None) == 1000

    def test_enquirer_identity_is_not_sent(self, request_sent):
        assert "Test Enquirer" not in request_sent["contents"]
        assert "test@example.com" not in request_sent["contents"]


class TestGeminiErrorHandling:
    @pytest.fixture
    def strict(self, gemini_settings):
        """Fallback disabled, so failures surface instead of being absorbed."""
        return replace(gemini_settings, allow_fallback=False)

    @pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"])
    def test_blocked_finish_reasons_raise(self, strict, reason):
        client = MockGeminiClient(finish_reason=reason)
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError, match="declined"):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_truncated_response_raises(self, strict):
        client = MockGeminiClient(finish_reason="MAX_TOKENS")
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError, match="max_output_tokens"):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_unparseable_response_raises(self, strict):
        client = MockGeminiClient(response_text="Sorry, I cannot help with that.")
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_sdk_exception_is_wrapped(self, strict):
        client = MockGeminiClient(raises=RuntimeError("connection reset"))
        messages, nlp, score = make_context()
        with pytest.raises(LLMCallError, match="connection reset"):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_failures_fall_back_when_allowed(self, gemini_settings):
        client = MockGeminiClient(raises=RuntimeError("boom"))
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(
            gemini_settings, client=client
        ).generate(LEAD, messages, nlp, score)
        assert metadata.is_fallback is True
        assert "boom" in (metadata.note or "")
        assert payload["qualification_summary"]

    def test_fenced_json_is_accepted(self, gemini_settings):
        import json as _json

        client = MockGeminiClient(
            response_text="```json\n" + _json.dumps(MOCK_PAYLOAD) + "\n```"
        )
        messages, nlp, score = make_context()
        _, metadata = QualificationLLM(gemini_settings, client=client).generate(
            LEAD, messages, nlp, score
        )
        assert metadata.is_fallback is False


# ==========================================================================
# Cross-provider behaviour
# ==========================================================================
class TestProviderParity:
    """The same prompts and contract go to both providers."""

    def test_both_providers_produce_the_same_payload_shape(
        self, live_settings, gemini_settings, mock_client, mock_gemini_client
    ):
        messages, nlp, score = make_context()
        anthropic_payload, anthropic_meta = QualificationLLM(
            live_settings, client=mock_client
        ).generate(LEAD, messages, nlp, score)
        gemini_payload, gemini_meta = QualificationLLM(
            gemini_settings, client=mock_gemini_client
        ).generate(LEAD, messages, nlp, score)

        assert anthropic_payload.keys() == gemini_payload.keys()
        assert anthropic_meta.provider == PROVIDER_ANTHROPIC
        assert gemini_meta.provider == PROVIDER_GEMINI

    def test_both_receive_the_identical_system_prompt(
        self, live_settings, gemini_settings, mock_client, mock_gemini_client
    ):
        messages, nlp, score = make_context()
        QualificationLLM(live_settings, client=mock_client).generate(
            LEAD, messages, nlp, score
        )
        QualificationLLM(gemini_settings, client=mock_gemini_client).generate(
            LEAD, messages, nlp, score
        )
        anthropic_system = mock_client.calls[0]["system"][0]["text"]
        gemini_system = mock_gemini_client.calls[0]["config"].system_instruction
        assert anthropic_system == gemini_system

    def test_both_receive_the_identical_schema(
        self, live_settings, gemini_settings, mock_client, mock_gemini_client
    ):
        messages, nlp, score = make_context()
        QualificationLLM(live_settings, client=mock_client).generate(
            LEAD, messages, nlp, score
        )
        QualificationLLM(gemini_settings, client=mock_gemini_client).generate(
            LEAD, messages, nlp, score
        )
        assert (
            mock_client.calls[0]["output_config"]["format"]["schema"]
            == mock_gemini_client.calls[0]["config"].response_json_schema
        )
