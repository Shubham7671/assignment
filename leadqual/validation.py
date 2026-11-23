"""Input validation.

Every value that enters the system from a human, a file or an API response
passes through here first. Validators return the *cleaned* value and raise
:class:`~leadqual.exceptions.ValidationError` on bad input.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable, Sequence

from .exceptions import ValidationError

# Deliberately permissive - we are cleaning CRM data, not enforcing RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
# Allows a leading "+" or "(" so both "+44 7700 900111" and "(020) 7946-0000"
# are accepted; the digit-count check below is what actually bounds the value.
_PHONE_RE = re.compile(r"^[+(]?[0-9][0-9\s\-().]{4,20}$")
_EXTERNAL_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,31}$")

VALID_CHANNELS = ("email", "web_form", "phone", "whatsapp", "chat", "sms", "in_person")
VALID_DIRECTIONS = ("inbound", "outbound")
VALID_LEAD_STATUSES = ("new", "in_progress", "qualified", "needs_info", "closed")
VALID_REVIEW_DECISIONS = ("approved", "edited", "rejected", "escalated")

MAX_NAME_LEN = 120
MAX_EMAIL_LEN = 254
MAX_BODY_LEN = 20_000
MAX_NOTE_LEN = 4_000


def require_text(
    value: Any,
    field: str,
    *,
    min_len: int = 1,
    max_len: int = 1_000,
    allow_none: bool = False,
) -> str | None:
    """Validate and normalise a free-text value.

    Args:
        value: Raw input.
        field: Field name, used in the error message.
        min_len: Minimum length after stripping.
        max_len: Maximum allowed length.
        allow_none: When True, ``None``/blank returns ``None`` instead of raising.

    Returns:
        The stripped string, or ``None`` when ``allow_none`` and input is blank.

    Raises:
        ValidationError: If the value is missing, the wrong type or out of range.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return None
        raise ValidationError("a value is required", field, value)

    if not isinstance(value, str):
        raise ValidationError(f"expected text, got {type(value).__name__}", field, value)

    cleaned = value.strip()
    if len(cleaned) < min_len:
        raise ValidationError(f"must be at least {min_len} characters", field, value)
    if len(cleaned) > max_len:
        raise ValidationError(f"must be at most {max_len} characters", field, cleaned[:60])
    return cleaned


def validate_name(value: Any) -> str:
    """Validate a person's full name."""
    cleaned = require_text(value, "full_name", min_len=2, max_len=MAX_NAME_LEN)
    assert cleaned is not None  # narrowed by allow_none=False
    if not any(ch.isalpha() for ch in cleaned):
        raise ValidationError("must contain at least one letter", "full_name", value)
    return " ".join(cleaned.split())


def validate_email(value: Any) -> str:
    """Validate and lower-case an email address."""
    cleaned = require_text(value, "email", min_len=5, max_len=MAX_EMAIL_LEN)
    assert cleaned is not None
    cleaned = cleaned.lower()
    if not _EMAIL_RE.match(cleaned):
        raise ValidationError("is not a valid email address", "email", value)
    return cleaned


def validate_phone(value: Any, *, allow_none: bool = True) -> str | None:
    """Validate an optional phone number."""
    cleaned = require_text(value, "phone", min_len=6, max_len=24, allow_none=allow_none)
    if cleaned is None:
        return None
    digits = sum(1 for ch in cleaned if ch.isdigit())
    if not _PHONE_RE.match(cleaned) or not 6 <= digits <= 15:
        raise ValidationError(
            "must contain 6-15 digits, optionally with +, spaces, dashes or brackets",
            "phone",
            value,
        )
    return cleaned


def validate_external_ref(value: Any) -> str | None:
    """Validate an optional CRM reference such as ``CRM-00042``."""
    cleaned = require_text(value, "external_ref", max_len=32, allow_none=True)
    if cleaned is None:
        return None
    if not _EXTERNAL_REF_RE.match(cleaned):
        raise ValidationError(
            "may contain only letters, digits, hyphen and underscore", "external_ref", value
        )
    return cleaned


def validate_choice(value: Any, field: str, allowed: Sequence[str], *, default: str | None = None) -> str:
    """Validate that ``value`` is one of ``allowed`` (case-insensitive)."""
    if (value is None or (isinstance(value, str) and not value.strip())) and default is not None:
        return default
    cleaned = require_text(value, field, max_len=64)
    assert cleaned is not None
    lowered = cleaned.lower()
    if lowered not in allowed:
        raise ValidationError(f"must be one of {', '.join(allowed)}", field, value)
    return lowered


def validate_message_body(value: Any) -> str:
    """Validate the text of a lead message."""
    cleaned = require_text(value, "body", min_len=2, max_len=MAX_BODY_LEN)
    assert cleaned is not None
    return cleaned


def validate_timestamp(value: Any, field: str = "sent_at", *, allow_none: bool = True) -> str | None:
    """Validate an ISO-8601 timestamp string and return it normalised."""
    cleaned = require_text(value, field, max_len=40, allow_none=allow_none)
    if cleaned is None:
        return None
    candidate = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValidationError(f"is not a valid ISO-8601 timestamp ({exc})", field, value) from exc
    return parsed.isoformat()


def validate_lead_id(value: Any) -> int:
    """Validate a positive integer lead identifier."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValidationError("must be a positive integer", "lead_id", value)
    try:
        lead_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("must be a positive integer", "lead_id", value) from exc
    if lead_id <= 0:
        raise ValidationError("must be a positive integer", "lead_id", value)
    return lead_id


def validate_lead_payload(payload: Any) -> dict[str, Any]:
    """Validate a raw lead dictionary (e.g. one record from a JSON import).

    Returns:
        A dict with cleaned keys ready for :func:`leadqual.db.insert_lead`.
    """
    if not isinstance(payload, dict):
        raise ValidationError(f"expected an object, got {type(payload).__name__}", "lead")

    return {
        "full_name": validate_name(payload.get("full_name")),
        "email": validate_email(payload.get("email")),
        "phone": validate_phone(payload.get("phone")),
        "source": validate_choice(payload.get("source"), "source", VALID_CHANNELS, default="web_form"),
        "programme_interest": require_text(
            payload.get("programme_interest"), "programme_interest", max_len=160, allow_none=True
        ),
        "status": validate_choice(
            payload.get("status"), "status", VALID_LEAD_STATUSES, default="new"
        ),
        "external_ref": validate_external_ref(payload.get("external_ref")),
    }


def validate_message_payload(payload: Any, *, lead_id: int | None = None) -> dict[str, Any]:
    """Validate a raw lead-message dictionary."""
    if not isinstance(payload, dict):
        raise ValidationError(f"expected an object, got {type(payload).__name__}", "message")

    resolved_lead_id = lead_id if lead_id is not None else payload.get("lead_id")
    return {
        "lead_id": validate_lead_id(resolved_lead_id),
        "body": validate_message_body(payload.get("body")),
        "channel": validate_choice(
            payload.get("channel"), "channel", VALID_CHANNELS, default="email"
        ),
        "direction": validate_choice(
            payload.get("direction"), "direction", VALID_DIRECTIONS, default="inbound"
        ),
        "sent_at": validate_timestamp(payload.get("sent_at")),
    }


def validate_review(
    decision: Any, reviewer: Any, notes: Any = None
) -> tuple[str, str, str | None]:
    """Validate a counsellor's review submission."""
    clean_decision = validate_choice(decision, "decision", VALID_REVIEW_DECISIONS)
    clean_reviewer = require_text(reviewer, "reviewer", min_len=2, max_len=MAX_NAME_LEN)
    clean_notes = require_text(notes, "notes", max_len=MAX_NOTE_LEN, allow_none=True)
    assert clean_reviewer is not None
    return clean_decision, clean_reviewer, clean_notes


def collect_errors(items: Iterable[Any], validator) -> tuple[list[Any], list[str]]:
    """Run ``validator`` over ``items``, separating successes from failures.

    Useful for bulk imports where one bad record should not abort the batch.

    Returns:
        ``(valid_results, error_messages)``.
    """
    valid: list[Any] = []
    errors: list[str] = []
    for index, item in enumerate(items):
        try:
            valid.append(validator(item))
        except ValidationError as exc:
            errors.append(f"record {index}: {exc}")
    return valid, errors
