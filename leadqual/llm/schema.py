"""The structured-output contract for the qualification summary.

The model is constrained to this JSON Schema via the Messages API's
``output_config.format``. :func:`validate_payload` re-checks the result in
Python as well - a schema-constrained response is still an external input,
and the application must not trust it blindly.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import LLMResponseError
from ..models import FIELD_SPECS_BY_NAME

MAX_SUMMARY_CHARS = 1200
MAX_QUESTION_CHARS = 300
MAX_FOLLOWUPS = 6

#: Sent to the API as ``output_config.format.schema``.
QUALIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "qualification_summary",
        "information_present",
        "information_missing",
        "suggested_followups",
        "recommended_next_step",
    ],
    "properties": {
        "qualification_summary": {
            "type": "string",
            "minLength": 40,
            "maxLength": MAX_SUMMARY_CHARS,
            "description": (
                "2-4 sentences for the counsellor, describing what this enquiry "
                "does and does not tell us. Factual, neutral, no judgement of the person."
            ),
        },
        "information_present": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 200},
            "description": "Short phrases naming what the enquirer has already told us.",
        },
        "information_missing": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 200},
            "description": "Short phrases naming the gaps that matter for a consultation.",
        },
        "suggested_followups": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_FOLLOWUPS,
            "description": "Questions the counsellor should ask, most important first.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "priority", "rationale"],
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": MAX_QUESTION_CHARS,
                        "description": "A single, polite, open question to ask the enquirer.",
                    },
                    "field_name": {
                        "type": "string",
                        "description": "Which qualification field this question fills, if any.",
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "1 = ask first, 5 = ask if there is time.",
                    },
                    "rationale": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "Why this question matters, in one line, for the counsellor.",
                    },
                },
            },
        },
        "recommended_next_step": {
            "type": "string",
            "minLength": 10,
            "maxLength": 400,
            "description": (
                "One sentence telling the counsellor what to do next, e.g. book the "
                "consultation, send a short qualifying email, or call to confirm timings."
            ),
        },
        "confidence_note": {
            "type": "string",
            "maxLength": 400,
            "description": "Optional caveat about anything ambiguous in the enquiry.",
        },
    },
}


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate and normalise a model response against the contract.

    Checks types, required keys, lengths and value ranges, and silently
    repairs the harmless problems (clamping priority, trimming over-long
    strings, dropping unknown field names) rather than discarding a
    response that is 95% usable.

    Args:
        payload: The parsed JSON object from the model.

    Returns:
        A cleaned dict safe for the rest of the application.

    Raises:
        LLMResponseError: If the payload is structurally unusable.
    """
    if not isinstance(payload, dict):
        raise LLMResponseError(
            f"Expected a JSON object from the model, got {type(payload).__name__}."
        )

    summary = payload.get("qualification_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMResponseError("Model response is missing 'qualification_summary'.")
    summary = summary.strip()[:MAX_SUMMARY_CHARS]

    def _string_list(key: str) -> list[str]:
        raw = payload.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [item.strip()[:200] for item in raw if isinstance(item, str) and item.strip()][:12]

    raw_followups = payload.get("suggested_followups", [])
    if not isinstance(raw_followups, list):
        raise LLMResponseError("'suggested_followups' must be a list.")

    followups: list[dict[str, Any]] = []
    for entry in raw_followups[:MAX_FOLLOWUPS]:
        if not isinstance(entry, dict):
            continue
        question = entry.get("question")
        if not isinstance(question, str) or len(question.strip()) < 5:
            continue

        try:
            priority = int(entry.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3
        priority = max(1, min(5, priority))

        field_name = entry.get("field_name")
        if not isinstance(field_name, str) or field_name not in FIELD_SPECS_BY_NAME:
            field_name = None

        rationale = entry.get("rationale")
        rationale = rationale.strip()[:300] if isinstance(rationale, str) else ""

        followups.append(
            {
                "question": question.strip()[:MAX_QUESTION_CHARS],
                "field_name": field_name,
                "priority": priority,
                "rationale": rationale,
            }
        )

    if not followups:
        raise LLMResponseError("Model returned no usable follow-up questions.")

    next_step = payload.get("recommended_next_step")
    next_step = next_step.strip()[:400] if isinstance(next_step, str) else ""

    note = payload.get("confidence_note")
    note = note.strip()[:400] if isinstance(note, str) else ""

    return {
        "qualification_summary": summary,
        "information_present": _string_list("information_present"),
        "information_missing": _string_list("information_missing"),
        "suggested_followups": followups,
        "recommended_next_step": next_step,
        "confidence_note": note,
    }
