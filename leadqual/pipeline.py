"""The end-to-end qualification workflow.

    read lead history -> NLP extraction -> completeness scoring
    -> LLM summary & follow-up questions -> save counsellor review

:func:`qualify_lead` runs the first four steps and persists a report;
:func:`record_review` closes the loop with the counsellor's decision.
"""

from __future__ import annotations

from typing import Any, Sequence

from .config import Settings, load_settings
from .db import Database
from .exceptions import (
    LLMError,
    PipelineError,
    RecordNotFoundError,
    ValidationError,
)
from .llm.client import QualificationLLM
from .models import (
    FIELD_SPECS_BY_NAME,
    FollowUpQuestion,
    QualificationReport,
    utc_now_iso,
)
from .nlp.extract import analyse_lead
from .nlp.similarity import find_similar_leads
from .scoring import baseline_followups, rank_gaps, score_completeness
from .validation import validate_lead_id, validate_review

MAX_FOLLOWUPS = 6


def _merge_followups(
    llm_followups: Sequence[dict[str, Any]],
    score,
    *,
    limit: int = MAX_FOLLOWUPS,
) -> list[FollowUpQuestion]:
    """Combine the model's questions with the deterministic baseline.

    The model's questions lead, because they are tailored to what the
    enquirer actually wrote. Any *required* gap the model did not cover is
    then topped up from the baseline, so a counsellor is never left without
    a question for a field the consultation depends on.
    """
    merged: list[FollowUpQuestion] = []
    seen_fields: set[str] = set()
    seen_questions: set[str] = set()

    for entry in llm_followups:
        question = str(entry.get("question", "")).strip()
        if not question:
            continue
        key = question.lower().rstrip("?. ")
        if key in seen_questions:
            continue
        seen_questions.add(key)
        field_name = entry.get("field_name")
        if isinstance(field_name, str) and field_name in FIELD_SPECS_BY_NAME:
            seen_fields.add(field_name)
        else:
            field_name = None
        merged.append(
            FollowUpQuestion(
                question=question,
                field_name=field_name,
                priority=int(entry.get("priority", 3)),
                rationale=str(entry.get("rationale", "")),
            )
        )

    uncovered_required = [
        name for name in score.missing_required if name not in seen_fields
    ]
    if uncovered_required:
        for fallback in baseline_followups(score, limit=limit):
            if fallback.field_name in uncovered_required:
                key = fallback.question.lower().rstrip("?. ")
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                merged.append(fallback)

    merged.sort(key=lambda f: (f.priority, f.field_name or "zz"))
    return merged[:limit]


def qualify_lead(
    db: Database,
    lead_id: int,
    *,
    settings: Settings | None = None,
    llm: QualificationLLM | None = None,
    persist: bool = True,
) -> QualificationReport:
    """Run the full qualification workflow for one lead.

    Args:
        db: An open database.
        lead_id: The lead to qualify.
        settings: Application settings; loaded from the environment if omitted.
        llm: An existing LLM client. Pass one when qualifying many leads so
            the SDK client and prompt are reused; tests pass a mock-backed one.
        persist: When False, the report is returned but not written - used by
            the UI's "preview" mode.

    Returns:
        The populated :class:`~leadqual.models.QualificationReport`.

    Raises:
        ValidationError: If ``lead_id`` is not a positive integer.
        RecordNotFoundError: If the lead does not exist.
        PipelineError: If the lead has no inbound messages to analyse, or an
            unexpected failure occurs partway through.
    """
    lead_id = validate_lead_id(lead_id)
    settings = settings or load_settings()
    llm = llm or QualificationLLM(settings)

    # --- 1. Read lead history ------------------------------------------
    lead = db.get_lead(lead_id)  # raises RecordNotFoundError
    messages = db.get_messages(lead_id)
    inbound = [m for m in messages if m.get("direction") == "inbound"]
    if not inbound:
        raise PipelineError(
            f"Lead {lead_id} has no inbound messages, so there is nothing to qualify. "
            "Add the enquirer's message first."
        )

    # --- 2. NLP extraction ----------------------------------------------
    try:
        nlp = analyse_lead(messages)
    except Exception as exc:  # noqa: BLE001 - surfaced as a pipeline failure
        raise PipelineError(f"NLP extraction failed for lead {lead_id}: {exc}") from exc

    # --- 3. Completeness scoring ----------------------------------------
    score = score_completeness(nlp)

    # --- 4. LLM summary and follow-up questions --------------------------
    payload, metadata = llm.generate(lead, messages, nlp, score)

    report = QualificationReport(
        lead_id=lead_id,
        summary=payload["qualification_summary"],
        readiness_score=score.score,
        readiness_band=score.band,
        nlp=nlp,
        score=score,
        followups=_merge_followups(payload["suggested_followups"], score),
        metadata=metadata,
        recommended_next_step=payload.get("recommended_next_step", ""),
        llm_payload=payload,
        status="awaiting_review",
    )

    if persist:
        db.save_report(report)
        # Keep the lead's workflow status aligned with the latest assessment.
        new_status = "qualified" if score.band == "READY" else "needs_info"
        if lead.get("status") not in ("closed", new_status):
            db.update_lead_status(lead_id, new_status, actor="pipeline")
    else:
        report.created_at = utc_now_iso()

    return report


def qualify_many(
    db: Database,
    lead_ids: Sequence[int] | None = None,
    *,
    settings: Settings | None = None,
    skip_errors: bool = True,
) -> tuple[list[QualificationReport], list[str]]:
    """Qualify several leads with one shared LLM client.

    Args:
        db: An open database.
        lead_ids: Specific leads, or ``None`` for every lead that has at
            least one inbound message.
        settings: Application settings.
        skip_errors: Collect per-lead failures instead of aborting the batch.

    Returns:
        ``(reports, errors)``.
    """
    settings = settings or load_settings()
    llm = QualificationLLM(settings)

    if lead_ids is None:
        lead_ids = [
            row["lead_id"]
            for row in db.list_leads(limit=10_000)
            if row.get("message_count", 0) > 0
        ]

    reports: list[QualificationReport] = []
    errors: list[str] = []
    for lead_id in lead_ids:
        try:
            reports.append(qualify_lead(db, lead_id, settings=settings, llm=llm))
        except (PipelineError, RecordNotFoundError, ValidationError, LLMError) as exc:
            # LLMError matters here: with LEADQUAL_ALLOW_FALLBACK=false a single
            # provider outage would otherwise abort the whole batch and discard
            # the leads already done.
            if not skip_errors:
                raise
            errors.append(f"lead {lead_id}: {exc}")
    return reports, errors


def record_review(
    db: Database,
    report_id: int,
    *,
    reviewer: str,
    decision: str,
    notes: str | None = None,
    lead_status: str | None = None,
) -> int:
    """Save a counsellor's review - the final step of the workflow.

    Args:
        db: An open database.
        report_id: The report being signed off.
        reviewer: Counsellor name or id.
        decision: ``approved`` | ``edited`` | ``rejected`` | ``escalated``.
        notes: Optional free-text notes.
        lead_status: Optionally move the lead to a new status at the same time.

    Returns:
        The new ``review_id``.

    Raises:
        ValidationError: If the submission fails validation.
        RecordNotFoundError: If the report does not exist.
    """
    clean_decision, clean_reviewer, clean_notes = validate_review(decision, reviewer, notes)
    return db.save_review(
        report_id=report_id,
        reviewer=clean_reviewer,
        decision=clean_decision,
        notes=clean_notes,
        lead_status=lead_status,
    )


def similar_past_leads(
    db: Database, lead_id: int, *, top_n: int = 3
) -> list[dict[str, Any]]:
    """Find previously qualified leads whose enquiries read similarly.

    Gives the counsellor precedent: what was asked last time, and how it went.

    Args:
        db: An open database.
        lead_id: The lead to compare against.
        top_n: How many neighbours to return.

    Returns:
        Candidate dicts with ``lead_id``, ``full_name``, ``similarity`` and,
        where available, the past readiness band.
    """
    lead_id = validate_lead_id(lead_id)
    query_text = "\n".join(
        m["body"] for m in db.get_messages(lead_id, direction="inbound")
    )
    if not query_text.strip():
        return []

    candidates: list[dict[str, Any]] = []
    for row in db.list_leads(limit=500):
        other_id = int(row["lead_id"])
        if other_id == lead_id or not row.get("message_count"):
            continue
        text = "\n".join(
            m["body"] for m in db.get_messages(other_id, direction="inbound")
        )
        if not text.strip():
            continue
        candidates.append(
            {
                "lead_id": other_id,
                "full_name": row.get("full_name"),
                "latest_band": row.get("latest_band"),
                "latest_score": row.get("latest_score"),
                "text": text,
            }
        )

    matches = find_similar_leads(query_text, candidates, top_n=top_n)
    for match in matches:
        match.pop("text", None)  # don't leak whole histories into the UI
    return matches


def pipeline_preview(db: Database, lead_id: int) -> dict[str, Any]:
    """Run only the deterministic half (NLP + scoring), with no LLM call.

    Useful for demonstrating the pipeline without spending API credit, and
    for the UI's "analyse first, then summarise" flow.
    """
    lead_id = validate_lead_id(lead_id)
    lead = db.get_lead(lead_id)
    messages = db.get_messages(lead_id)
    nlp = analyse_lead(messages)
    score = score_completeness(nlp)
    return {
        "lead": lead,
        "messages": messages,
        "nlp": nlp,
        "score": score,
        "ranked_gaps": rank_gaps(score),
        "baseline_followups": baseline_followups(score),
    }
