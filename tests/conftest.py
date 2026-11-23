"""Shared pytest fixtures.

Every test runs against an in-memory SQLite database and a mocked LLM, so
the suite is fast, deterministic and never spends API credit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

# Make the project importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leadqual.config import Settings  # noqa: E402
from leadqual.db import Database  # noqa: E402


@pytest.fixture
def db() -> Iterator[Database]:
    """An initialised, empty, in-memory database."""
    database = Database(":memory:", create_dirs=False)
    database.initialise()
    yield database
    database.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a temporary file, with live calls disabled."""
    return Settings(
        db_path=tmp_path / "test.db",
        model="claude-opus-5-5",
        max_tokens=1000,
        api_key=None,
        offline=True,
        allow_fallback=True,
    )


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    """Settings that look configured, for exercising the live code path."""
    return Settings(
        db_path=tmp_path / "test.db",
        model="claude-opus-5-5",
        max_tokens=1000,
        api_key="sk-ant-test-key-not-real",
        offline=False,
        allow_fallback=True,
    )


@pytest.fixture
def sample_lead(db: Database) -> int:
    """A lead with one detailed inbound message. Returns its id."""
    lead_id = db.insert_lead(
        full_name="Test Enquirer",
        email="test.enquirer@example.com",
        phone="+44 7700 900000",
        source="web_form",
        programme_interest="MSc Data Analytics",
        external_ref="TEST-001",
    )
    db.insert_message(
        lead_id=lead_id,
        body=(
            "Hello, I am interested in the MSc in Data Analytics. I want to move "
            "into a data science role. I have completed a BSc in Statistics and I "
            "am currently working as an analyst. I would like to study online "
            "part-time, weekday evenings after 7pm, about 10 hours a week. My "
            "budget is around £9,000 and I could use an instalment plan. I would "
            "like to start in the January intake. I am based in Manchester. "
            "What documents do I need?"
        ),
        channel="web_form",
        direction="inbound",
        sent_at="2026-10-02T09:14:00+00:00",
    )
    return lead_id


@pytest.fixture
def sparse_lead(db: Database) -> int:
    """A lead whose enquiry says almost nothing. Returns its id."""
    lead_id = db.insert_lead(
        full_name="Sparse Enquirer",
        email="sparse@example.com",
        external_ref="TEST-002",
    )
    db.insert_message(
        lead_id=lead_id,
        body="Hi, please send me more information. Thanks.",
        channel="web_form",
        direction="inbound",
    )
    return lead_id


# --------------------------------------------------------------------------
# Mock Anthropic client
# --------------------------------------------------------------------------
MOCK_PAYLOAD = {
    "qualification_summary": (
        "This enquirer has named the programme and given a clear goal, study "
        "mode and weekly availability. Budget and start date are also on file, "
        "so a consultation can go straight to course detail. Only the contact "
        "preference is unstated."
    ),
    "information_present": ["Programme: MSc Data Analytics", "Goal: move into data science"],
    "information_missing": ["Contact preference"],
    "suggested_followups": [
        {
            "question": "What is the best way and time to reach you for a short call?",
            "field_name": "contact_preference",
            "priority": 1,
            "rationale": "Needed to book the consultation at a workable time.",
        },
        {
            "question": "Which specialisation within analytics interests you most?",
            "field_name": None,
            "priority": 2,
            "rationale": "Lets the counsellor prepare the right module detail.",
        },
    ],
    "recommended_next_step": "Book the consultation and confirm the contact window.",
    "confidence_note": "The budget figure is approximate.",
}


class MockMessages:
    """Stands in for ``client.messages``."""

    def __init__(self, owner: "MockAnthropicClient") -> None:
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.raises is not None:
            raise self._owner.raises
        text = (
            self._owner.response_text
            if self._owner.response_text is not None
            else json.dumps(MOCK_PAYLOAD)
        )
        return SimpleNamespace(
            id="msg_mock_0001",
            model=kwargs.get("model", "claude-opus-5-5"),
            stop_reason=self._owner.stop_reason,
            stop_details=None,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1234, output_tokens=321),
        )


class MockAnthropicClient:
    """A minimal stand-in for ``anthropic.Anthropic``.

    Records every request so tests can assert on the prompt and parameters.
    """

    def __init__(self, *, response_text: str | None = None,
                 stop_reason: str = "end_turn", raises: Exception | None = None) -> None:
        self.response_text = response_text
        self.stop_reason = stop_reason
        self.raises = raises
        self.calls: list[dict] = []
        self.messages = MockMessages(self)


@pytest.fixture
def mock_client() -> MockAnthropicClient:
    """A mock Anthropic client returning a valid payload."""
    return MockAnthropicClient()


# --------------------------------------------------------------------------
# Mock Gemini client
# --------------------------------------------------------------------------
class _MockFinishReason:
    """Stands in for the SDK's FinishReason enum, which exposes ``.name``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


class MockGeminiModels:
    """Stands in for ``client.models``."""

    def __init__(self, owner: "MockGeminiClient") -> None:
        self._owner = owner

    def generate_content(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.raises is not None:
            raise self._owner.raises
        text = (
            self._owner.response_text
            if self._owner.response_text is not None
            else json.dumps(MOCK_PAYLOAD)
        )
        return SimpleNamespace(
            text=text,
            response_id="gemini_mock_0001",
            model_version=kwargs.get("model", "gemini-3.5-flash-lite"),
            candidates=[
                SimpleNamespace(finish_reason=_MockFinishReason(self._owner.finish_reason))
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=2048,
                candidates_token_count=256,
                total_token_count=2304,
            ),
        )


class MockGeminiClient:
    """A minimal stand-in for ``google.genai.Client``."""

    def __init__(self, *, response_text: str | None = None,
                 finish_reason: str = "STOP", raises: Exception | None = None) -> None:
        self.response_text = response_text
        self.finish_reason = finish_reason
        self.raises = raises
        self.calls: list[dict] = []
        self.models = MockGeminiModels(self)


@pytest.fixture
def gemini_settings(tmp_path) -> Settings:
    """Settings selecting Gemini, with a key that looks configured."""
    return Settings(
        db_path=tmp_path / "test.db",
        provider="gemini",
        gemini_model="gemini-3.5-flash-lite",
        max_tokens=1000,
        api_key=None,
        gemini_api_key="AIza-test-key-not-real",
        offline=False,
        allow_fallback=True,
    )


@pytest.fixture
def mock_gemini_client() -> MockGeminiClient:
    """A mock Gemini client returning a valid payload."""
    return MockGeminiClient()
