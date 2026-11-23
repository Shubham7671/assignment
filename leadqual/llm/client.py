"""Generating the qualification summary.

:class:`QualificationLLM` is provider-neutral: it builds the prompts, hands
them to whichever adapter in :mod:`leadqual.llm.providers` is configured,
parses and validates the reply, and records provenance.

Two safety nets, both clearly labelled in the stored metadata:

* **Offline generator.** When no provider is configured, ``LEADQUAL_OFFLINE``
  is set, or a live call fails and fallback is permitted, a deterministic
  generator produces the same JSON shape from the NLP and scoring results.
  Reports written this way carry ``provider='offline-fallback'`` and
  ``is_fallback=1`` — they are never passed off as model output.
* **Tolerant parsing.** The reply is schema-constrained, but it is still an
  external input: JSON is recovered from a fenced block or surrounding prose
  if necessary, then re-validated in Python.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..config import Settings, load_settings
from ..exceptions import LLMCallError, LLMConfigurationError, LLMResponseError
from ..models import FIELD_SPECS_BY_NAME, LLMMetadata, NLPResult, ScoreResult
from ..scoring import baseline_followups, rank_gaps
from .prompts import build_system_prompt, build_user_prompt
from .providers import (
    PROVIDER_ANTHROPIC,
    PROVIDER_FALLBACK,
    PROVIDER_GEMINI,
    BaseProvider,
    build_provider,
    provider_availability,
)
from .schema import validate_payload

__all__ = [
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GEMINI",
    "PROVIDER_FALLBACK",
    "QualificationLLM",
    "build_offline_payload",
    "generate_qualification",
    "parse_json_response",
    "provider_availability",
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# ==========================================================================
# Response parsing
# ==========================================================================
def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model output.

    Handles a bare object, a fenced ```json block, and an object with
    surrounding prose (which structured outputs should prevent, but a
    fallback path can still produce).

    Raises:
        LLMResponseError: If no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise LLMResponseError("The model returned an empty response.")

    candidates: list[str] = [text.strip()]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group("body"))
    loose = _JSON_OBJECT_RE.search(text)
    if loose:
        candidates.append(loose.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMResponseError(
        "Could not parse a JSON object from the model response.", raw=text[:2000]
    )


# ==========================================================================
# Offline deterministic generator
# ==========================================================================
def _describe_present(nlp: NLPResult) -> list[str]:
    items: list[str] = []
    for name in nlp.present_fields():
        spec = FIELD_SPECS_BY_NAME.get(name)
        if spec is None:
            continue
        value = (nlp.fields[name].value or "").strip()
        items.append(f"{spec.label}: {value}" if value else spec.label)
    return items[:12]


def _describe_missing(score: ScoreResult) -> list[str]:
    labels: list[str] = []
    for name in rank_gaps(score):
        spec = FIELD_SPECS_BY_NAME.get(name)
        if spec is not None:
            labels.append(spec.label)
    return labels[:12]


def build_offline_payload(nlp: NLPResult, score: ScoreResult) -> dict[str, Any]:
    """Produce a schema-shaped payload without calling any API.

    The wording is assembled from the deterministic analysis, so the result
    is reproducible and contains nothing the enquirer did not say.
    """
    present = _describe_present(nlp)
    missing = _describe_missing(score)

    present_phrase = (
        "has given " + ", ".join(
            FIELD_SPECS_BY_NAME[n].label.lower() for n in nlp.present_fields()[:4]
        )
        if nlp.present_fields()
        else "has not yet supplied any of the fields the CRM tracks"
    )
    missing_required_labels = [
        FIELD_SPECS_BY_NAME[n].label.lower() for n in score.missing_required
    ]

    if score.band == "READY":
        verdict = "The enquiry carries enough detail to run a productive consultation now."
        next_step = (
            "Book the consultation and use the questions below to confirm the "
            "remaining detail at the start of the call."
        )
    elif score.band == "NEEDS_INFO":
        verdict = (
            "A short qualifying exchange would make the consultation considerably "
            "more productive."
        )
        next_step = (
            "Send a brief qualifying message covering "
            + (", ".join(missing_required_labels[:3]) if missing_required_labels
               else "the gaps listed below")
            + ", then book the consultation."
        )
    else:
        verdict = (
            "Key details are still missing, so a consultation booked now would be "
            "spent gathering basics."
        )
        next_step = (
            "Make a short qualifying call or send a structured email covering "
            + (", ".join(missing_required_labels[:3]) if missing_required_labels
               else "the required fields")
            + " before booking consultation time."
        )

    summary = (
        f"This enquirer {present_phrase}. "
        f"The CRM scores the enquiry at {score.score:.0f} out of 100 for "
        f"information completeness ({score.band}), with "
        f"{len(score.missing_required)} required and {len(score.missing_optional)} "
        f"optional field(s) still unanswered. {verdict}"
    )
    if nlp.questions:
        summary += (
            f" The enquirer has also asked {len(nlp.questions)} question(s) that "
            "should be answered during the consultation."
        )

    followups = [
        {
            "question": followup.question,
            "field_name": followup.field_name,
            "priority": followup.priority,
            "rationale": followup.rationale,
        }
        for followup in baseline_followups(score, limit=5)
    ]
    if not followups:
        followups = [
            {
                "question": (
                    "Is there anything else you would like us to cover in the "
                    "consultation?"
                ),
                "field_name": None,
                "priority": 1,
                "rationale": "Every tracked field is already on file; confirm nothing is missed.",
            }
        ]

    return {
        "qualification_summary": summary,
        "information_present": present,
        "information_missing": missing,
        "suggested_followups": followups,
        "recommended_next_step": next_step,
        "confidence_note": (
            "Generated offline by the CRM's deterministic rules, not by a language "
            "model. Wording is templated; the underlying analysis is the same."
        ),
    }


# ==========================================================================
# The orchestrator
# ==========================================================================
class QualificationLLM:
    """Generates qualification summaries via the configured provider.

    Args:
        settings: Application settings. Loaded from the environment when omitted.
        client: An injected SDK client, handed to the selected provider. Tests
            pass a mock here; leave it ``None`` in production.
        provider: An already-built provider adapter, overriding ``settings.provider``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        provider: BaseProvider | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.provider: BaseProvider = provider or build_provider(
            self.settings, client=client
        )

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when a live API call can be attempted."""
        return self.provider.available

    def status(self) -> dict[str, Any]:
        """Describe the active provider for the UI and CLI."""
        return self.provider.status()

    @staticmethod
    def all_provider_status(settings: Settings | None = None) -> list[dict[str, Any]]:
        """Status of every supported provider, not just the active one."""
        return provider_availability(settings or load_settings())

    # ------------------------------------------------------------------
    def generate(
        self,
        lead: dict[str, Any],
        messages: Sequence[dict[str, Any]],
        nlp: NLPResult,
        score: ScoreResult,
    ) -> tuple[dict[str, Any], LLMMetadata]:
        """Produce a qualification summary and follow-up questions.

        Args:
            lead: The lead record.
            messages: The lead's message history.
            nlp: Deterministic extraction results.
            score: Completeness scoring results.

        Returns:
            ``(payload, metadata)`` — a validated payload matching
            :data:`~leadqual.llm.schema.QUALIFICATION_SCHEMA`, and the
            provenance record to store alongside it.

        Raises:
            LLMCallError / LLMResponseError / LLMConfigurationError: Only when
            the live call fails *and* fallback is disabled.
        """
        if not self.available:
            return self._offline(nlp, score, note=self.provider.status_reason)

        try:
            return self._generate_live(lead, messages, nlp, score)
        except (LLMCallError, LLMResponseError, LLMConfigurationError) as exc:
            if not self.settings.allow_fallback:
                raise
            return self._offline(
                nlp, score, note=f"Live call failed, used offline generator: {exc}"
            )

    # ------------------------------------------------------------------
    def _offline(
        self, nlp: NLPResult, score: ScoreResult, *, note: str
    ) -> tuple[dict[str, Any], LLMMetadata]:
        return (
            build_offline_payload(nlp, score),
            LLMMetadata(
                provider=PROVIDER_FALLBACK,
                model="rule-based-generator",
                is_fallback=True,
                note=note,
            ),
        )

    def _generate_live(
        self,
        lead: dict[str, Any],
        messages: Sequence[dict[str, Any]],
        nlp: NLPResult,
        score: ScoreResult,
    ) -> tuple[dict[str, Any], LLMMetadata]:
        """Call the provider and validate the result."""
        response = self.provider.complete(
            build_system_prompt(), build_user_prompt(lead, messages, nlp, score)
        )
        payload = validate_payload(parse_json_response(response.text))

        metadata = LLMMetadata(
            provider=self.provider.name,
            model=response.model,
            is_fallback=False,
            response_id=response.response_id,
            stop_reason=response.stop_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            note=None,
        )
        return payload, metadata


# ==========================================================================
# Convenience function
# ==========================================================================
def generate_qualification(
    lead: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    nlp: NLPResult,
    score: ScoreResult,
    *,
    settings: Settings | None = None,
    client: Any | None = None,
) -> tuple[dict[str, Any], LLMMetadata]:
    """One-shot helper around :class:`QualificationLLM`."""
    return QualificationLLM(settings, client=client).generate(lead, messages, nlp, score)
