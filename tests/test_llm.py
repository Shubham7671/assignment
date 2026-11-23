"""Tests for the LLM layer: prompts, schema validation and the API client.

Every test here uses the mock client from ``conftest.py``. No test in this
suite reaches the network or spends API credit; the real call is exercised
in the live demonstration described in the README.
"""

from __future__ import annotations

import json

import pytest

from leadqual.exceptions import LLMCallError, LLMConfigurationError, LLMResponseError
from leadqual.fairness import REDACTION_TOKEN
from leadqual.llm.client import (
    PROVIDER_ANTHROPIC,
    PROVIDER_FALLBACK,
    QualificationLLM,
    build_offline_payload,
    parse_json_response,
)
from leadqual.llm.prompts import build_system_prompt, build_user_prompt
from leadqual.llm.schema import QUALIFICATION_SCHEMA, validate_payload
from leadqual.nlp.extract import analyse_lead
from leadqual.scoring import score_completeness
from tests.conftest import MOCK_PAYLOAD, MockAnthropicClient

LEAD = {
    "lead_id": 1,
    "full_name": "Test Enquirer",
    "email": "test@example.com",
    "source": "web_form",
    "programme_interest": "MSc Data Analytics",
}


def make_context(text: str = "I want an MBA. I can study at weekends. What are the fees?"):
    """Build (messages, nlp, score) for prompt and client tests."""
    messages = [
        {"message_id": 1, "body": text, "direction": "inbound",
         "channel": "email", "sent_at": "2026-10-01T10:00:00+00:00"}
    ]
    nlp = analyse_lead(messages)
    return messages, nlp, score_completeness(nlp)


# ==========================================================================
# Schema
# ==========================================================================
class TestSchema:
    def test_schema_is_closed_and_declares_required_keys(self):
        assert QUALIFICATION_SCHEMA["additionalProperties"] is False
        assert "qualification_summary" in QUALIFICATION_SCHEMA["required"]
        assert "suggested_followups" in QUALIFICATION_SCHEMA["required"]
        followups = QUALIFICATION_SCHEMA["properties"]["suggested_followups"]
        assert followups["items"]["additionalProperties"] is False

    def test_accepts_a_good_payload(self):
        cleaned = validate_payload(MOCK_PAYLOAD)
        assert cleaned["qualification_summary"]
        assert len(cleaned["suggested_followups"]) == 2
        assert cleaned["suggested_followups"][0]["field_name"] == "contact_preference"

    def test_clamps_priority_and_drops_unknown_field_names(self):
        cleaned = validate_payload(
            {
                "qualification_summary": "A" * 50,
                "suggested_followups": [
                    {"question": "A valid question here?", "priority": 99,
                     "field_name": "not_a_real_field", "rationale": "x"}
                ],
            }
        )
        followup = cleaned["suggested_followups"][0]
        assert followup["priority"] == 5
        assert followup["field_name"] is None

    def test_coerces_a_string_into_a_list(self):
        cleaned = validate_payload(
            {
                "qualification_summary": "A" * 50,
                "information_present": "Just one thing",
                "suggested_followups": [{"question": "A valid question?", "priority": 1}],
            }
        )
        assert cleaned["information_present"] == ["Just one thing"]

    def test_truncates_over_long_strings(self):
        cleaned = validate_payload(
            {
                "qualification_summary": "A" * 5000,
                "suggested_followups": [{"question": "B" * 900, "priority": 1}],
            }
        )
        assert len(cleaned["qualification_summary"]) <= 1200
        assert len(cleaned["suggested_followups"][0]["question"]) <= 300

    @pytest.mark.parametrize(
        "bad",
        [
            "a string, not an object",
            {"suggested_followups": [{"question": "Where?"}]},          # no summary
            {"qualification_summary": "A" * 50, "suggested_followups": []},  # no questions
            {"qualification_summary": "A" * 50,
             "suggested_followups": [{"question": "hi"}]},               # too short
        ],
    )
    def test_rejects_unusable_payloads(self, bad):
        with pytest.raises(LLMResponseError):
            validate_payload(bad)


# ==========================================================================
# JSON parsing
# ==========================================================================
class TestParsing:
    def test_bare_object(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_with_surrounding_prose(self):
        assert parse_json_response('Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    @pytest.mark.parametrize("bad", ["", "   ", "not json at all", "[1, 2, 3]"])
    def test_rejects_unparseable(self, bad):
        with pytest.raises(LLMResponseError):
            parse_json_response(bad)


# ==========================================================================
# Prompts
# ==========================================================================
class TestPrompts:
    def test_system_prompt_is_stable(self):
        """Byte-identical across calls, so the cached prefix stays valid."""
        assert build_system_prompt() == build_system_prompt()

    def test_system_prompt_carries_the_fairness_rules(self):
        prompt = build_system_prompt()
        assert "FAIRNESS CONSTRAINTS" in prompt
        assert "Never infer or comment on age" in prompt
        assert "deprioritising, declining or filtering out" in prompt

    def test_system_prompt_lists_every_field_and_an_example(self):
        prompt = build_system_prompt()
        for name in ("study_goal", "budget_indication", "open_questions"):
            assert name in prompt
        assert "EXAMPLE" in prompt

    def test_user_prompt_includes_analysis_and_messages(self):
        messages, nlp, score = make_context()
        prompt = build_user_prompt(LEAD, messages, nlp, score)
        assert "READINESS SCORE" in prompt
        assert "FIELDS MISSING" in prompt
        assert "I want an MBA" in prompt

    def test_user_prompt_omits_the_enquirer_name(self):
        messages, nlp, score = make_context()
        prompt = build_user_prompt(LEAD, messages, nlp, score)
        assert "Test Enquirer" not in prompt
        assert "test@example.com" not in prompt

    def test_user_prompt_redacts_protected_attributes(self):
        messages, nlp, score = make_context(
            "I am a 42 year old married woman and I want to study nursing."
        )
        prompt = build_user_prompt(LEAD, messages, nlp, score)
        assert "42 year old" not in prompt
        assert "married" not in prompt
        assert REDACTION_TOKEN in prompt
        assert "nursing" in prompt

    def test_user_prompt_flags_that_redaction_happened(self):
        messages, nlp, score = make_context("I am a 42 year old career changer.")
        prompt = build_user_prompt(LEAD, messages, nlp, score)
        assert "must play no part" in prompt

    def test_outbound_messages_are_excluded(self):
        messages, nlp, score = make_context()
        messages.append(
            {"message_id": 2, "body": "SECRET INTERNAL NOTE", "direction": "outbound",
             "channel": "email", "sent_at": "2026-10-02T10:00:00+00:00"}
        )
        prompt = build_user_prompt(LEAD, messages, nlp, score)
        assert "SECRET INTERNAL NOTE" not in prompt


# ==========================================================================
# Offline generator
# ==========================================================================
class TestOfflineGenerator:
    def test_output_satisfies_the_contract(self):
        _, nlp, score = make_context()
        payload = validate_payload(build_offline_payload(nlp, score))
        assert payload["qualification_summary"]
        assert payload["suggested_followups"]
        assert payload["recommended_next_step"]

    def test_it_says_it_is_not_a_model(self):
        _, nlp, score = make_context()
        payload = build_offline_payload(nlp, score)
        assert "not by a language model" in payload["confidence_note"]

    def test_it_is_deterministic(self):
        _, nlp, score = make_context()
        assert build_offline_payload(nlp, score) == build_offline_payload(nlp, score)

    def test_it_adapts_to_the_band(self):
        _, poor_nlp, poor_score = make_context("Hi. Thanks.")
        _, rich_nlp, rich_score = make_context(
            "I want to move into data science with an MSc in Data Analytics. "
            "Weekday evenings after 7pm, 10 hours a week. Starting January "
            "intake. Budget around £9,000. Online study. I completed a BSc in "
            "Statistics. Based in Manchester. Contact me via WhatsApp."
        )
        assert poor_score.band == "NOT_READY"
        assert "before booking consultation time" in (
            build_offline_payload(poor_nlp, poor_score)["recommended_next_step"]
        )
        assert rich_score.band == "READY"
        assert "Book the consultation" in (
            build_offline_payload(rich_nlp, rich_score)["recommended_next_step"]
        )


# ==========================================================================
# The client
# ==========================================================================
class TestClientRouting:
    def test_no_key_uses_the_offline_generator(self, settings):
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(settings).generate(LEAD, messages, nlp, score)
        assert metadata.provider == PROVIDER_FALLBACK
        assert metadata.is_fallback is True
        assert payload["qualification_summary"]

    def test_status_explains_why_it_is_offline(self, settings):
        status = QualificationLLM(settings).status()
        assert status["live_calls_enabled"] is False
        assert status["provider"] == PROVIDER_ANTHROPIC
        assert "No API key found" in status["reason"]

    def test_injected_client_is_used(self, live_settings, mock_client):
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(
            live_settings, client=mock_client
        ).generate(LEAD, messages, nlp, score)

        assert metadata.provider == PROVIDER_ANTHROPIC
        assert metadata.is_fallback is False
        assert metadata.model == "claude-opus-5-5"
        assert metadata.response_id == "msg_mock_0001"
        assert metadata.input_tokens == 1234
        assert metadata.output_tokens == 321
        assert metadata.latency_ms is not None and metadata.latency_ms >= 0
        assert payload["qualification_summary"] == MOCK_PAYLOAD["qualification_summary"]


class TestRequestShape:
    @pytest.fixture
    def request_sent(self, live_settings, mock_client):
        messages, nlp, score = make_context()
        QualificationLLM(live_settings, client=mock_client).generate(
            LEAD, messages, nlp, score
        )
        return mock_client.calls[0]

    def test_model_and_tokens(self, request_sent):
        assert request_sent["model"] == "claude-opus-5-5"
        assert request_sent["max_tokens"] == 1000

    def test_structured_output_is_requested(self, request_sent):
        output_format = request_sent["output_config"]["format"]
        assert output_format["type"] == "json_schema"
        assert output_format["schema"] == QUALIFICATION_SCHEMA

    def test_system_prompt_is_cached(self, request_sent):
        system = request_sent["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_no_deprecated_parameters(self, request_sent):
        """Opus 5.5 rejects a thinking budget and sampling parameters."""
        assert "thinking" not in request_sent
        assert "temperature" not in request_sent
        assert "top_p" not in request_sent

    def test_single_user_turn(self, request_sent):
        assert len(request_sent["messages"]) == 1
        assert request_sent["messages"][0]["role"] == "user"


class TestErrorHandling:
    def test_unparseable_response_falls_back(self, live_settings):
        client = MockAnthropicClient(response_text="I'm afraid I can't do that.")
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(
            live_settings, client=client
        ).generate(LEAD, messages, nlp, score)
        assert metadata.is_fallback is True
        assert "Live call failed" in (metadata.note or "")
        assert payload["qualification_summary"]

    def test_fallback_can_be_disabled(self, live_settings):
        from dataclasses import replace

        strict = replace(live_settings, allow_fallback=False)
        client = MockAnthropicClient(response_text="not json")
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_refusal_is_detected(self, live_settings):
        from dataclasses import replace

        strict = replace(live_settings, allow_fallback=False)
        client = MockAnthropicClient(stop_reason="refusal")
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError, match="declined"):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_truncated_response_is_detected(self, live_settings):
        from dataclasses import replace

        strict = replace(live_settings, allow_fallback=False)
        client = MockAnthropicClient(stop_reason="max_tokens")
        messages, nlp, score = make_context()
        with pytest.raises(LLMResponseError, match="max_tokens"):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_api_exception_falls_back(self, live_settings):
        client = MockAnthropicClient(raises=RuntimeError("connection reset"))
        messages, nlp, score = make_context()
        payload, metadata = QualificationLLM(
            live_settings, client=client
        ).generate(LEAD, messages, nlp, score)
        assert metadata.is_fallback is True
        assert "connection reset" in (metadata.note or "")

    def test_api_exception_raises_when_fallback_is_off(self, live_settings):
        from dataclasses import replace

        strict = replace(live_settings, allow_fallback=False)
        client = MockAnthropicClient(raises=RuntimeError("boom"))
        messages, nlp, score = make_context()
        with pytest.raises(LLMCallError):
            QualificationLLM(strict, client=client).generate(LEAD, messages, nlp, score)

    def test_missing_key_on_the_live_path_is_a_configuration_error(self, settings):
        from dataclasses import replace

        broken = replace(settings, offline=False, api_key=None)
        with pytest.raises(LLMConfigurationError):
            QualificationLLM(broken).provider._get_client()


class TestResponseVariants:
    def test_fenced_json_is_accepted(self, live_settings):
        client = MockAnthropicClient(
            response_text="```json\n" + json.dumps(MOCK_PAYLOAD) + "\n```"
        )
        messages, nlp, score = make_context()
        _, metadata = QualificationLLM(live_settings, client=client).generate(
            LEAD, messages, nlp, score
        )
        assert metadata.is_fallback is False

    def test_prose_wrapped_json_is_accepted(self, live_settings):
        client = MockAnthropicClient(
            response_text="Sure, here it is:\n" + json.dumps(MOCK_PAYLOAD)
        )
        messages, nlp, score = make_context()
        _, metadata = QualificationLLM(live_settings, client=client).generate(
            LEAD, messages, nlp, score
        )
        assert metadata.is_fallback is False
