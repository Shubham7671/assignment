"""Reporting layer: SQL results turned into pandas DataFrames.

The SQL lives in :mod:`leadqual.db`; this module shapes the results for the
CLI tables, the Streamlit dashboard and CSV export. pandas is used where it
earns its place - grouping, percentage columns, pivots and export - rather
than as a replacement for queries SQLite already does well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .db import Database
from .models import FIELD_SPECS, READINESS_BANDS


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame that still has the right columns when empty."""
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


# --------------------------------------------------------------------------
# Dashboards
# --------------------------------------------------------------------------
def overview_metrics(db: Database) -> dict[str, Any]:
    """Headline counts for the top of the dashboard."""
    metrics = db.query_pipeline_overview()
    total_reports = metrics.get("total_reports") or 0
    reviewed = (metrics.get("total_reports") or 0) - (metrics.get("awaiting_review") or 0)
    metrics["review_rate_pct"] = (
        round(100.0 * reviewed / total_reports, 1) if total_reports else 0.0
    )
    metrics["avg_score"] = metrics.get("avg_score") or 0.0
    return metrics


def readiness_distribution(db: Database) -> pd.DataFrame:
    """Lead counts per readiness band, with a share-of-total column.

    Bands with no leads are included at zero so the chart axis is stable.
    """
    rows = db.query_readiness_distribution()
    frame = _frame(rows, ["band", "lead_count", "avg_score", "min_score", "max_score"])

    all_bands = pd.DataFrame({"band": [name for _, name, _ in READINESS_BANDS]})
    frame = all_bands.merge(frame, on="band", how="left")
    # Coerce first: an all-empty merge leaves object columns, and calling
    # .fillna on those is deprecated in pandas 2.x.
    frame["lead_count"] = pd.to_numeric(frame["lead_count"], errors="coerce").fillna(0).astype(int)
    for column in ("avg_score", "min_score", "max_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    total = int(frame["lead_count"].sum())
    frame["share_pct"] = (
        (100.0 * frame["lead_count"] / total).round(1) if total else 0.0
    )
    return frame


def missing_field_report(db: Database) -> pd.DataFrame:
    """Which fields are most often missing, ordered worst first.

    Answers the operational question behind this project: *what do our
    enquiry forms and first-contact templates fail to capture?*
    """
    rows = db.query_missing_field_frequency()
    frame = _frame(
        rows,
        ["field_name", "field_label", "is_required", "weight",
         "missing_count", "total_count", "missing_pct"],
    )
    if frame.empty:
        return frame
    frame["is_required"] = frame["is_required"].astype(bool)
    frame["score_impact"] = (
        frame["weight"] * frame["missing_count"] / frame["total_count"].clip(lower=1)
    ).round(2)
    return frame.sort_values(
        ["is_required", "missing_pct", "weight"], ascending=[False, False, False]
    ).reset_index(drop=True)


def leads_needing_info(db: Database, *, threshold: float = 75.0) -> pd.DataFrame:
    """The counsellor work queue: leads below the readiness threshold."""
    rows = db.query_leads_needing_info(threshold=threshold)
    return _frame(
        rows,
        ["lead_id", "full_name", "email", "lead_status", "report_id",
         "readiness_score", "readiness_band", "created_at",
         "open_followups", "missing_required"],
    )


def provider_usage(db: Database) -> pd.DataFrame:
    """Token spend and latency per provider/model - the API audit view."""
    rows = db.query_provider_usage()
    frame = _frame(
        rows,
        ["provider", "model", "is_fallback", "run_count", "input_tokens",
         "output_tokens", "avg_latency_ms", "first_run", "last_run"],
    )
    if frame.empty:
        return frame
    frame["is_fallback"] = frame["is_fallback"].astype(bool)
    frame["total_tokens"] = frame["input_tokens"].fillna(0) + frame["output_tokens"].fillna(0)
    total_runs = int(frame["run_count"].sum())
    frame["share_of_runs_pct"] = (
        (100.0 * frame["run_count"] / total_runs).round(1) if total_runs else 0.0
    )
    return frame.sort_values("run_count", ascending=False).reset_index(drop=True)


def review_summary(db: Database) -> pd.DataFrame:
    """Counsellor decisions, counts and the average score they reviewed."""
    rows = db.query_review_summary()
    return _frame(
        rows, ["decision", "review_count", "avg_score_reviewed", "distinct_reviewers"]
    )


def field_coverage_matrix(db: Database, *, limit: int = 50) -> pd.DataFrame:
    """A lead x field grid of what is present - a pandas pivot over the
    per-field rows, useful for spotting patterns across the whole pipeline.

    Returns:
        A DataFrame indexed by lead, one column per qualification field,
        values ``1`` (present) or ``0`` (missing).
    """
    reports = db.list_reports(limit=limit)
    if not reports:
        return pd.DataFrame(columns=[spec.name for spec in FIELD_SPECS])

    records: list[dict[str, Any]] = []
    for report in reports:
        for field_row in db.get_report_fields(int(report["report_id"])):
            records.append(
                {
                    "lead": f"{report['lead_id']} - {report['full_name']}",
                    "field": field_row["field_name"],
                    "present": int(field_row["is_present"]),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=[spec.name for spec in FIELD_SPECS])

    pivot = frame.pivot_table(
        index="lead", columns="field", values="present", aggfunc="max", fill_value=0
    )
    ordered = [spec.name for spec in FIELD_SPECS if spec.name in pivot.columns]
    return pivot[ordered]


def report_to_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Turn one stored report's field rows into a display DataFrame."""
    fields = report.get("fields") or []
    if not fields:
        return pd.DataFrame(columns=["Field", "Required", "Weight", "Status", "Confidence", "Value"])
    return pd.DataFrame(
        [
            {
                "Field": row["field_label"],
                "Required": "yes" if row["is_required"] else "no",
                "Weight": row["weight"],
                "Status": "present" if row["is_present"] else "missing",
                "Confidence": round(float(row["confidence"]), 2),
                "Value": row["field_value"] or "",
            }
            for row in fields
        ]
    )


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_all(db: Database, directory: str | Path) -> dict[str, str]:
    """Write every analytical view to CSV.

    Args:
        db: An open database.
        directory: Output directory, created if needed.

    Returns:
        A mapping of view name to the file written.
    """
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    views: dict[str, pd.DataFrame] = {
        "readiness_distribution": readiness_distribution(db),
        "missing_fields": missing_field_report(db),
        "leads_needing_info": leads_needing_info(db),
        "provider_usage": provider_usage(db),
        "review_summary": review_summary(db),
        "field_coverage": field_coverage_matrix(db).reset_index(),
    }

    written: dict[str, str] = {}
    for name, frame in views.items():
        destination = output_dir / f"{name}.csv"
        frame.to_csv(destination, index=False, encoding="utf-8")
        written[name] = str(destination)
    return written
