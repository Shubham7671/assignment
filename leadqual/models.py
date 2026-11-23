"""Domain model: the qualification fields and the dataclasses moved between layers.

This module is the single source of truth for *what* a counsellor needs to
know before a consultation. The NLP layer fills these fields in, the scoring
layer weighs them, the LLM layer explains them and the UI displays them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


# --------------------------------------------------------------------------
# Qualification field catalogue
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """Definition of one piece of information a counsellor needs.

    Attributes:
        name: Machine key, stored in ``qualification_fields.field_name``.
        label: Human-readable label for the UI.
        weight: Contribution to the readiness score (all weights sum to 100).
        required: Whether a consultation is hard to run without it.
        description: What "present" means for this field.
        prompt_hint: A neutral example follow-up question for this gap.
    """

    name: str
    label: str
    weight: int
    required: bool
    description: str
    prompt_hint: str


#: The canonical catalogue. Weights sum to 100.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="study_goal",
        label="Study goal",
        weight=18,
        required=True,
        description="The outcome the enquirer wants (career change, promotion, exam, migration, degree).",
        prompt_hint="What would you most like to be able to do once the programme is finished?",
    ),
    FieldSpec(
        name="programme_interest",
        label="Programme of interest",
        weight=15,
        required=True,
        description="The subject, course or qualification they are asking about.",
        prompt_hint="Which programme or subject area are you considering at the moment?",
    ),
    FieldSpec(
        name="preferred_schedule",
        label="Preferred schedule",
        weight=14,
        required=True,
        description="Days, times or weekly hours they can commit to studying.",
        prompt_hint="Which days and times of the week usually work best for your study sessions?",
    ),
    FieldSpec(
        name="start_timeline",
        label="Start timeline",
        weight=12,
        required=True,
        description="When they would like to begin, or which intake they are targeting.",
        prompt_hint="Do you have an intake or start date in mind?",
    ),
    FieldSpec(
        name="budget_indication",
        label="Budget indication",
        weight=12,
        required=True,
        description="Any wording about fees, affordability, instalments, funding or sponsorship.",
        prompt_hint="Is there a fee range or payment arrangement you would like us to work within?",
    ),
    FieldSpec(
        name="study_mode",
        label="Study mode",
        weight=8,
        required=False,
        description="Online, on-campus, hybrid, part-time or full-time preference.",
        prompt_hint="Would you prefer to study online, on campus, or a mix of both?",
    ),
    FieldSpec(
        name="current_background",
        label="Current background",
        weight=8,
        required=False,
        description="Prior study, qualifications or work experience relevant to entry requirements.",
        prompt_hint="Could you tell me about your most recent qualification or role?",
    ),
    FieldSpec(
        name="location",
        label="Location / time zone",
        weight=5,
        required=False,
        description="Where they are based, for campus choice or scheduling across time zones.",
        prompt_hint="Which city or time zone are you based in?",
    ),
    FieldSpec(
        name="contact_preference",
        label="Contact preference",
        weight=5,
        required=False,
        description="How and when they would like to be contacted.",
        prompt_hint="What is the best way and time to reach you for a short call?",
    ),
    FieldSpec(
        name="open_questions",
        label="Open questions raised",
        weight=3,
        required=False,
        description="Questions the enquirer has asked that are still unanswered.",
        prompt_hint="Is there anything specific you would like the consultation to cover?",
    ),
)

#: Fast lookup by field name.
FIELD_SPECS_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}

#: Names of the fields a consultation really cannot run without.
REQUIRED_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.required)


# --------------------------------------------------------------------------
# Readiness bands
# --------------------------------------------------------------------------
#: (minimum score, band name, counsellor-facing meaning)
READINESS_BANDS: tuple[tuple[int, str, str], ...] = (
    (75, "READY", "Enough detail on file for a productive consultation."),
    (45, "NEEDS_INFO", "Worth a short qualifying call to fill specific gaps first."),
    (0, "NOT_READY", "Key details are missing; gather them before booking time."),
)


def band_for_score(score: float) -> tuple[str, str]:
    """Map a 0-100 readiness score onto a band name and its meaning."""
    for minimum, name, meaning in READINESS_BANDS:
        if score >= minimum:
            return name, meaning
    return "NOT_READY", READINESS_BANDS[-1][2]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Transfer objects
# --------------------------------------------------------------------------
@dataclass
class Lead:
    """A prospective student enquiry."""

    full_name: str
    email: str
    phone: str | None = None
    source: str = "web_form"
    programme_interest: str | None = None
    status: str = "new"
    external_ref: str | None = None
    lead_id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeadMessage:
    """One inbound or outbound message on a lead's history."""

    lead_id: int
    body: str
    channel: str = "email"
    direction: str = "inbound"
    sent_at: str | None = None
    message_id: int | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedField:
    """One qualification field as resolved by the NLP layer."""

    field_name: str
    is_present: bool
    value: str | None = None
    confidence: float = 0.0
    evidence_message_id: int | None = None
    evidence_text: str | None = None
    method: str = "rule"

    @property
    def spec(self) -> FieldSpec:
        return FIELD_SPECS_BY_NAME[self.field_name]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NLPResult:
    """Everything the NLP layer derived from a lead's message history."""

    fields: dict[str, ExtractedField] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)
    intent: str = "unknown"
    intent_confidence: float = 0.0
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    keywords: list[str] = field(default_factory=list)
    token_count: int = 0
    message_count: int = 0
    protected_attribute_hits: list[str] = field(default_factory=list)

    def present_fields(self) -> list[str]:
        return [name for name, f in self.fields.items() if f.is_present]

    def missing_fields(self) -> list[str]:
        return [name for name, f in self.fields.items() if not f.is_present]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "questions": self.questions,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "keywords": self.keywords,
            "token_count": self.token_count,
            "message_count": self.message_count,
            "protected_attribute_hits": self.protected_attribute_hits,
        }


@dataclass
class ScoreResult:
    """Outcome of completeness scoring."""

    score: float
    band: str
    band_meaning: str
    earned_weight: float
    total_weight: int
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    per_field: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FollowUpQuestion:
    """A suggested question for the counsellor to ask."""

    question: str
    field_name: str | None = None
    priority: int = 3
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMMetadata:
    """Provenance for an LLM call - recorded on every report for audit."""

    provider: str
    model: str
    is_fallback: bool = False
    response_id: str | None = None
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualificationReport:
    """The complete result of one qualification run on one lead."""

    lead_id: int
    summary: str
    readiness_score: float
    readiness_band: str
    nlp: NLPResult
    score: ScoreResult
    followups: list[FollowUpQuestion] = field(default_factory=list)
    metadata: LLMMetadata | None = None
    recommended_next_step: str = ""
    #: The raw (validated) payload returned by the LLM, kept for audit.
    llm_payload: dict[str, Any] = field(default_factory=dict)
    report_id: int | None = None
    status: str = "awaiting_review"
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "lead_id": self.lead_id,
            "summary": self.summary,
            "readiness_score": self.readiness_score,
            "readiness_band": self.readiness_band,
            "recommended_next_step": self.recommended_next_step,
            "status": self.status,
            "created_at": self.created_at,
            "nlp": self.nlp.to_dict(),
            "score": self.score.to_dict(),
            "followups": [f.to_dict() for f in self.followups],
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "llm_payload": self.llm_payload,
        }
