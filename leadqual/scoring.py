"""Completeness scoring.

The readiness score answers one question: *does this enquiry contain enough
information for a useful consultation?* It is a weighted completeness
measure over :data:`leadqual.models.FIELD_SPECS` - nothing about the person
enters the calculation.

Score = sum(weight x confidence) over fields that were found, out of 100,
with a penalty applied when a *required* field is missing, because an
enquiry that is rich in optional detail but has no stated goal is still not
ready for a consultation.
"""

from __future__ import annotations

from .models import (
    FIELD_SPECS,
    FIELD_SPECS_BY_NAME,
    ExtractedField,
    FollowUpQuestion,
    NLPResult,
    ScoreResult,
    band_for_score,
)

#: Confidence below this is treated as "mentioned, but not firmly stated".
PARTIAL_CONFIDENCE_FLOOR = 0.60

#: Each missing required field costs this many points on top of its weight.
MISSING_REQUIRED_PENALTY = 4.0

#: A field below the floor earns this share of its weight.
PARTIAL_CREDIT = 0.6


def _field_credit(field: ExtractedField | None, weight: int) -> float:
    """Points earned by one field.

    Full weight for a confident extraction, partial credit for a weak one,
    zero when absent.
    """
    if field is None or not field.is_present:
        return 0.0
    if field.confidence >= PARTIAL_CONFIDENCE_FLOOR:
        return float(weight)
    return float(weight) * PARTIAL_CREDIT


def score_completeness(nlp: NLPResult) -> ScoreResult:
    """Convert NLP extraction results into a readiness score.

    Args:
        nlp: The output of :func:`leadqual.nlp.extract.analyse_lead`.

    Returns:
        A :class:`~leadqual.models.ScoreResult` with the 0-100 score, its
        band, the per-field breakdown and the lists of missing fields.
    """
    total_weight = sum(spec.weight for spec in FIELD_SPECS)
    earned = 0.0
    per_field: dict[str, float] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for spec in FIELD_SPECS:
        field = nlp.fields.get(spec.name)
        credit = _field_credit(field, spec.weight)
        per_field[spec.name] = round(credit, 2)
        earned += credit

        if field is None or not field.is_present:
            (missing_required if spec.required else missing_optional).append(spec.name)

    penalty = MISSING_REQUIRED_PENALTY * len(missing_required)
    raw_score = (earned / total_weight) * 100.0 - penalty
    score = round(max(0.0, min(100.0, raw_score)), 1)
    band, meaning = band_for_score(score)

    return ScoreResult(
        score=score,
        band=band,
        band_meaning=meaning,
        earned_weight=round(earned, 2),
        total_weight=total_weight,
        missing_required=missing_required,
        missing_optional=missing_optional,
        per_field=per_field,
    )


def rank_gaps(score: ScoreResult) -> list[str]:
    """Order the missing fields by how much they would improve the score.

    Required fields come first (their weight plus the penalty they carry),
    then optional fields by weight.

    Returns:
        Field names, most valuable gap first.
    """
    def value(name: str) -> float:
        spec = FIELD_SPECS_BY_NAME[name]
        return spec.weight + (MISSING_REQUIRED_PENALTY if spec.required else 0.0)

    gaps = score.missing_required + score.missing_optional
    return sorted(gaps, key=value, reverse=True)


def baseline_followups(score: ScoreResult, limit: int = 5) -> list[FollowUpQuestion]:
    """Deterministic follow-up questions straight from the field catalogue.

    These are generated without the LLM. They guarantee the counsellor always
    has something usable - the LLM's job is to improve on them by tailoring
    the wording to what the enquirer actually wrote.

    Args:
        score: The completeness result.
        limit: Maximum number of questions.

    Returns:
        Up to ``limit`` questions, highest priority (1) first.
    """
    questions: list[FollowUpQuestion] = []
    for position, name in enumerate(rank_gaps(score)[:limit]):
        spec = FIELD_SPECS_BY_NAME[name]
        questions.append(
            FollowUpQuestion(
                question=spec.prompt_hint,
                field_name=name,
                priority=min(5, position + 1),
                rationale=(
                    f"'{spec.label}' is not stated anywhere in the enquiry and is "
                    f"{'required' if spec.required else 'useful'} for a productive consultation."
                ),
            )
        )
    return questions


def explain_score(score: ScoreResult) -> list[dict[str, object]]:
    """Row-by-row breakdown of how the score was reached, for the UI."""
    rows: list[dict[str, object]] = []
    for spec in FIELD_SPECS:
        earned = score.per_field.get(spec.name, 0.0)
        rows.append(
            {
                "field": spec.name,
                "label": spec.label,
                "required": spec.required,
                "weight": spec.weight,
                "earned": earned,
                "status": (
                    "present" if earned >= spec.weight
                    else "partial" if earned > 0
                    else "missing"
                ),
            }
        )
    rows.sort(key=lambda r: (not r["required"], -float(r["weight"])))  # type: ignore[arg-type]
    return rows
