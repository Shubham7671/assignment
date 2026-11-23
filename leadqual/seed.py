"""Loading synthetic sample data into the database.

The bundled dataset (``data/sample_leads.json``) is entirely invented: no
real person, email address or phone number appears in it, and every address
uses the reserved ``example.com`` domain. It is written to exercise the
whole range of the workflow - from a one-line SMS with almost no detail to a
multi-message thread that scores as READY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .db import Database
from .exceptions import ValidationError
from .validation import validate_lead_payload, validate_message_payload

DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_leads.json"


def load_sample_file(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read and parse the sample-lead JSON file.

    Args:
        path: Path to the JSON file; the bundled dataset when omitted.

    Returns:
        The list of raw lead dicts (each may carry a ``messages`` list).

    Raises:
        ValidationError: If the file is missing, unreadable or malformed.
    """
    sample_path = Path(path) if path else DEFAULT_SAMPLE_PATH
    try:
        raw = sample_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(
            f"Sample data file not found at {sample_path}", "path", str(sample_path)
        ) from exc
    except OSError as exc:
        raise ValidationError(f"Could not read {sample_path}: {exc}", "path") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{sample_path} is not valid JSON: {exc}", "path") from exc

    if isinstance(document, list):
        leads = document
    elif isinstance(document, dict):
        leads = document.get("leads", [])
    else:
        raise ValidationError(
            f"Expected an object or list in {sample_path}, got {type(document).__name__}"
        )

    if not isinstance(leads, list) or not leads:
        raise ValidationError(f"No leads found in {sample_path}")
    return leads


def seed_database(
    db: Database,
    *,
    path: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Insert the sample leads and their message histories.

    Each record is validated before it is written, and a bad record is
    reported rather than aborting the whole load.

    Args:
        db: An initialised database.
        path: Optional path to an alternative dataset.
        skip_existing: Skip leads whose ``external_ref`` is already present.

    Returns:
        A summary dict with counts and any per-record errors.
    """
    raw_leads = load_sample_file(path)

    inserted_leads = 0
    inserted_messages = 0
    skipped = 0
    errors: list[str] = []

    existing_refs = {
        row["external_ref"]
        for row in db.list_leads(limit=10_000)
        if row.get("external_ref")
    }

    for index, raw_lead in enumerate(raw_leads):
        try:
            if not isinstance(raw_lead, dict):
                raise ValidationError(
                    f"expected an object, got {type(raw_lead).__name__}", f"lead[{index}]"
                )

            raw_messages = raw_lead.get("messages", [])
            if not isinstance(raw_messages, list):
                raise ValidationError("'messages' must be a list", f"lead[{index}]")

            cleaned_lead = validate_lead_payload(raw_lead)

            if skip_existing and cleaned_lead["external_ref"] in existing_refs:
                skipped += 1
                continue

            lead_id = db.insert_lead(**cleaned_lead)
            inserted_leads += 1
            if cleaned_lead["external_ref"]:
                existing_refs.add(cleaned_lead["external_ref"])

            for message_index, raw_message in enumerate(raw_messages):
                try:
                    cleaned_message = validate_message_payload(raw_message, lead_id=lead_id)
                    db.insert_message(**cleaned_message)
                    inserted_messages += 1
                except ValidationError as exc:
                    errors.append(f"lead[{index}].messages[{message_index}]: {exc}")

        except ValidationError as exc:
            errors.append(f"lead[{index}]: {exc}")

    db.log_audit(
        "database",
        None,
        "seeded",
        details={
            "leads": inserted_leads,
            "messages": inserted_messages,
            "skipped": skipped,
            "errors": len(errors),
        },
    )

    return {
        "leads_inserted": inserted_leads,
        "messages_inserted": inserted_messages,
        "leads_skipped": skipped,
        "errors": errors,
        "source": str(Path(path) if path else DEFAULT_SAMPLE_PATH),
    }
