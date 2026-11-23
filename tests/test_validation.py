"""Tests for input validation."""

from __future__ import annotations

import pytest

from leadqual.exceptions import ValidationError
from leadqual.validation import (
    collect_errors,
    validate_choice,
    validate_email,
    validate_external_ref,
    validate_lead_id,
    validate_lead_payload,
    validate_message_payload,
    validate_name,
    validate_phone,
    validate_review,
    validate_timestamp,
)


class TestNames:
    def test_normalises_whitespace(self):
        assert validate_name("  Priya   Raghavan  ") == "Priya Raghavan"

    @pytest.mark.parametrize("bad", ["", "   ", None, "A", "12345", 42])
    def test_rejects_bad_names(self, bad):
        with pytest.raises(ValidationError):
            validate_name(bad)


class TestEmails:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Test@Example.com", "test@example.com"),
            ("  a.b+tag@sub.example.co.uk ", "a.b+tag@sub.example.co.uk"),
        ],
    )
    def test_accepts_and_lowercases(self, raw, expected):
        assert validate_email(raw) == expected

    @pytest.mark.parametrize(
        "bad", ["", "nope", "no@domain", "@example.com", "a b@example.com", None]
    )
    def test_rejects_bad_emails(self, bad):
        with pytest.raises(ValidationError):
            validate_email(bad)

    def test_error_names_the_field(self):
        with pytest.raises(ValidationError) as excinfo:
            validate_email("nope")
        assert excinfo.value.field == "email"


class TestPhones:
    @pytest.mark.parametrize("raw", ["+44 7700 900111", "07700900111", "(020) 7946-0000"])
    def test_accepts_common_formats(self, raw):
        assert validate_phone(raw) == raw.strip()

    def test_none_allowed_by_default(self):
        assert validate_phone(None) is None
        assert validate_phone("") is None

    @pytest.mark.parametrize("bad", ["abc", "12", "+" * 10])
    def test_rejects_bad_phones(self, bad):
        with pytest.raises(ValidationError):
            validate_phone(bad)

    def test_can_require_a_value(self):
        with pytest.raises(ValidationError):
            validate_phone(None, allow_none=False)


class TestMisc:
    def test_external_ref_rules(self):
        assert validate_external_ref("CRM-1001") == "CRM-1001"
        assert validate_external_ref(None) is None
        with pytest.raises(ValidationError):
            validate_external_ref("bad ref!")

    def test_choice_is_case_insensitive_with_default(self):
        assert validate_choice("EMAIL", "channel", ("email", "sms")) == "email"
        assert validate_choice(None, "channel", ("email",), default="email") == "email"
        with pytest.raises(ValidationError):
            validate_choice("pigeon", "channel", ("email", "sms"))

    def test_timestamp_normalises_zulu(self):
        assert validate_timestamp("2026-10-02T09:14:00Z").startswith("2026-10-02T09:14:00")
        assert validate_timestamp(None) is None
        with pytest.raises(ValidationError):
            validate_timestamp("not-a-date")

    @pytest.mark.parametrize("value,expected", [(5, 5), ("7", 7)])
    def test_lead_id_accepts_ints_and_digit_strings(self, value, expected):
        assert validate_lead_id(value) == expected

    @pytest.mark.parametrize("bad", [0, -1, "abc", None, True, 1.5])
    def test_lead_id_rejects_everything_else(self, bad):
        with pytest.raises(ValidationError):
            validate_lead_id(bad)


class TestPayloads:
    def test_lead_payload_cleans_every_field(self):
        cleaned = validate_lead_payload(
            {
                "full_name": "  Ada  Lovelace ",
                "email": "ADA@example.com",
                "phone": "+44 7700 900123",
                "source": "WEB_FORM",
                "programme_interest": " MSc Computing ",
                "external_ref": "CRM-9",
            }
        )
        assert cleaned["full_name"] == "Ada Lovelace"
        assert cleaned["email"] == "ada@example.com"
        assert cleaned["source"] == "web_form"
        assert cleaned["status"] == "new"
        assert cleaned["programme_interest"] == "MSc Computing"

    def test_lead_payload_rejects_non_dict(self):
        with pytest.raises(ValidationError):
            validate_lead_payload(["not", "a", "dict"])

    def test_message_payload_takes_lead_id_override(self):
        cleaned = validate_message_payload({"body": "Hello there"}, lead_id=3)
        assert cleaned["lead_id"] == 3
        assert cleaned["direction"] == "inbound"
        assert cleaned["channel"] == "email"

    def test_message_payload_requires_a_body(self):
        with pytest.raises(ValidationError):
            validate_message_payload({"body": ""}, lead_id=1)


class TestReview:
    def test_valid_review(self):
        decision, reviewer, notes = validate_review("APPROVED", " A. Counsellor ", " ok ")
        assert (decision, reviewer, notes) == ("approved", "A. Counsellor", "ok")

    def test_notes_are_optional(self):
        assert validate_review("approved", "Reviewer", None)[2] is None

    def test_bad_decision_rejected(self):
        with pytest.raises(ValidationError):
            validate_review("maybe", "Reviewer")


class TestCollectErrors:
    def test_separates_good_from_bad(self):
        valid, errors = collect_errors(
            ["ada@example.com", "broken", "bob@example.com"], validate_email
        )
        assert valid == ["ada@example.com", "bob@example.com"]
        assert len(errors) == 1
        assert "record 1" in errors[0]
