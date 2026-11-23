"""Command-line interface.

Run ``python -m leadqual --help`` for the full command list. A typical
end-to-end demonstration::

    python -m leadqual init
    python -m leadqual seed
    python -m leadqual analyse 1          # NLP + scoring, no API call
    python -m leadqual qualify 1          # full pipeline, calls Claude
    python -m leadqual report 1
    python -m leadqual review 1 --reviewer "A. Counsellor" --decision approved
    python -m leadqual stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_settings
from .db import Database, open_database
from .exceptions import LeadQualError
from .llm.client import QualificationLLM
from .models import FIELD_SPECS_BY_NAME
from .nlp.preprocess import nlp_backend_status
from .nlp.similarity import INTENT_DESCRIPTIONS, similarity_backend_status
from .pipeline import (
    pipeline_preview,
    qualify_lead,
    qualify_many,
    record_review,
    similar_past_leads,
)
from .reporting import (
    export_all,
    leads_needing_info,
    missing_field_report,
    overview_metrics,
    provider_usage,
    readiness_distribution,
    review_summary,
)
from .scoring import explain_score
from .seed import seed_database
from .validation import validate_lead_payload, validate_message_payload

# --------------------------------------------------------------------------
# Output helpers (rich when installed, plain text otherwise)
# --------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    _RICH = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    _RICH = False


def out(message: str = "") -> None:
    """Print a line, with rich markup when available."""
    if _RICH and _console is not None:
        _console.print(message)
    else:
        # Strip the simplest rich tags so plain output stays readable.
        import re

        print(re.sub(r"\[/?[a-z0-9 _#]+\]", "", message))


def _cell(value: Any, max_width: int) -> str:
    """Stringify and truncate one cell so wide tables stay readable."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    # Plain ASCII ellipsis: a Windows console on cp1252 raises on "…".
    return text if len(text) <= max_width else text[: max_width - 3] + "..."


def show_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_cell: int = 38,
) -> None:
    """Render a table, falling back to aligned plain text."""
    if not rows:
        out(f"[dim]{title}: no rows.[/dim]")
        return

    rows = [[_cell(cell, max_cell) for cell in row] for row in rows]

    if _RICH and _console is not None and Table is not None:
        table = Table(title=title, header_style="bold cyan", title_style="bold")
        for column in columns:
            table.add_column(str(column))
        for row in rows:
            table.add_row(*["" if cell is None else str(cell) for cell in row])
        _console.print(table)
        return

    print(f"\n{title}")
    widths = [
        max(len(str(columns[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(columns))
    ]
    print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[i] if row[i] is not None else "").ljust(widths[i])
                         for i in range(len(columns))))


def frame_table(title: str, frame, columns: Sequence[str] | None = None) -> None:
    """Render a pandas DataFrame via :func:`show_table`.

    Args:
        title: Table caption.
        frame: The DataFrame.
        columns: Optional subset of columns to display, so the wide
            analytical frames stay legible in a terminal. The CSV export
            always keeps every column.
    """
    if frame is None or frame.empty:
        out(f"[dim]{title}: no data yet.[/dim]")
        return
    if columns:
        frame = frame[[c for c in columns if c in frame.columns]]
    show_table(title, list(frame.columns), frame.values.tolist())


def band_colour(band: str) -> str:
    return {"READY": "green", "NEEDS_INFO": "yellow", "NOT_READY": "red"}.get(band, "white")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace, db: Database) -> int:
    """Create the database schema."""
    if args.reset:
        db.reset()
        out("[yellow]Database reset.[/yellow]")
    else:
        db.initialise()
    out(f"[green]Schema ready[/green] at [bold]{db.path}[/bold]")
    return 0


def cmd_seed(args: argparse.Namespace, db: Database) -> int:
    """Load the synthetic sample dataset."""
    if args.reset:
        db.reset()
        out("[yellow]Database reset before seeding.[/yellow]")

    result = seed_database(db, path=args.file, skip_existing=not args.reset)
    out(
        f"[green]Seeded[/green] {result['leads_inserted']} lead(s) and "
        f"{result['messages_inserted']} message(s) from {result['source']}"
    )
    if result["leads_skipped"]:
        out(f"[dim]Skipped {result['leads_skipped']} lead(s) already present.[/dim]")
    for error in result["errors"]:
        out(f"[red]! {error}[/red]")
    return 0


def cmd_leads(args: argparse.Namespace, db: Database) -> int:
    """List leads."""
    rows = db.list_leads(status=args.status, limit=args.limit)
    show_table(
        "Leads",
        ["ID", "Ref", "Name", "Email", "Source", "Status", "Msgs", "Reports", "Score", "Band"],
        [
            [
                r["lead_id"], r["external_ref"], r["full_name"], r["email"],
                r["source"], r["status"], r["message_count"], r["report_count"],
                r["latest_score"] if r["latest_score"] is not None else "-",
                r["latest_band"] or "-",
            ]
            for r in rows
        ],
    )
    return 0


def cmd_show(args: argparse.Namespace, db: Database) -> int:
    """Show one lead and its message history."""
    lead = db.get_lead(args.lead_id)
    out(f"\n[bold]Lead {lead['lead_id']}[/bold] - {lead['full_name']} <{lead['email']}>")
    out(
        f"  ref={lead['external_ref']}  source={lead['source']}  status={lead['status']}\n"
        f"  programme={lead['programme_interest'] or '-'}  created={lead['created_at']}"
    )

    messages = db.get_messages(args.lead_id)
    show_table(
        "Message history",
        ["ID", "Direction", "Channel", "Sent", "Body"],
        [
            [
                m["message_id"], m["direction"], m["channel"], m["sent_at"],
                (m["body"][:90] + "...") if len(m["body"]) > 90 else m["body"],
            ]
            for m in messages
        ],
    )

    latest = db.get_latest_report(args.lead_id)
    if latest:
        out(
            f"\nLatest report: #{latest['report_id']} "
            f"[{band_colour(latest['readiness_band'])}]{latest['readiness_band']}"
            f"[/{band_colour(latest['readiness_band'])}] "
            f"({latest['readiness_score']}/100), status={latest['status']}"
        )
    else:
        out("\n[dim]No qualification report yet. Run: leadqual qualify "
            f"{args.lead_id}[/dim]")
    return 0


def cmd_analyse(args: argparse.Namespace, db: Database) -> int:
    """Run NLP + scoring only - no API call, no database write."""
    preview = pipeline_preview(db, args.lead_id)
    nlp, score = preview["nlp"], preview["score"]

    out(f"\n[bold]NLP analysis - lead {args.lead_id}[/bold] "
        f"({preview['lead']['full_name']})")
    out(
        f"  intent     : {nlp.intent} ({nlp.intent_confidence:.2f}) - "
        f"{INTENT_DESCRIPTIONS.get(nlp.intent, '')}\n"
        f"  tone       : {nlp.sentiment} ({nlp.sentiment_score:+.2f})\n"
        f"  messages   : {nlp.message_count}   tokens: {nlp.token_count}\n"
        f"  keywords   : {', '.join(nlp.keywords) or '-'}"
    )
    if nlp.protected_attribute_hits:
        out(
            "  [yellow]protected attributes mentioned: "
            f"{', '.join(nlp.protected_attribute_hits)} - excluded from scoring "
            "and redacted before any LLM call.[/yellow]"
        )

    show_table(
        "Field extraction",
        ["Field", "Req", "Weight", "Status", "Conf", "Value"],
        [
            [
                row["label"], "yes" if row["required"] else "", row["weight"],
                row["status"],
                f"{nlp.fields[row['field']].confidence:.2f}"
                if nlp.fields.get(row["field"]) else "0.00",
                (nlp.fields[row["field"]].value or "")[:60]
                if nlp.fields.get(row["field"]) else "",
            ]
            for row in explain_score(score)
        ],
    )

    colour = band_colour(score.band)
    out(
        f"\n[bold]Readiness: [{colour}]{score.score}/100 - {score.band}[/{colour}][/bold]\n"
        f"  {score.band_meaning}\n"
        f"  earned {score.earned_weight} of {score.total_weight} weight; "
        f"{len(score.missing_required)} required field(s) missing"
    )

    if nlp.questions:
        out("\n[bold]Questions the enquirer asked:[/bold]")
        for question in nlp.questions:
            out(f"  - {question}")

    out("\n[bold]Baseline follow-ups (no LLM):[/bold]")
    for followup in preview["baseline_followups"]:
        out(f"  [{followup.priority}] {followup.question}")
    out("\n[dim]Run 'qualify' to add the Claude-generated summary.[/dim]")
    return 0


def _print_report(report, db: Database | None = None) -> None:
    """Pretty-print a freshly generated report object."""
    colour = band_colour(report.readiness_band)
    meta = report.metadata
    out(
        f"\n[bold]Qualification report[/bold] for lead {report.lead_id}"
        + (f" (report #{report.report_id})" if report.report_id else "")
    )
    out(
        f"  readiness : [{colour}]{report.readiness_score}/100 - "
        f"{report.readiness_band}[/{colour}]"
    )
    if meta:
        tag = "[yellow]offline fallback[/yellow]" if meta.is_fallback else "[green]live API[/green]"
        out(f"  provider  : {meta.provider} / {meta.model}  {tag}")
        if meta.input_tokens is not None or meta.latency_ms is not None:
            out(
                f"  usage     : in={meta.input_tokens} out={meta.output_tokens} "
                f"latency={meta.latency_ms}ms stop={meta.stop_reason}"
            )
        if meta.response_id:
            out(f"  response  : {meta.response_id}")
        if meta.note:
            out(f"  note      : [dim]{meta.note}[/dim]")

    out(f"\n[bold]Summary[/bold]\n  {report.summary}")
    if report.recommended_next_step:
        out(f"\n[bold]Recommended next step[/bold]\n  {report.recommended_next_step}")

    out("\n[bold]Suggested follow-up questions[/bold]")
    for followup in report.followups:
        label = (
            FIELD_SPECS_BY_NAME[followup.field_name].label
            if followup.field_name in FIELD_SPECS_BY_NAME
            else "general"
        )
        out(f"  [{followup.priority}] ({label}) {followup.question}")
        if followup.rationale:
            out(f"      [dim]{followup.rationale}[/dim]")


def cmd_qualify(args: argparse.Namespace, db: Database) -> int:
    """Run the full pipeline for one lead or all leads."""
    settings = load_settings()
    status = QualificationLLM(settings).status()
    if not status["live_calls_enabled"]:
        out(f"[yellow]LLM: offline generator will be used - {status['reason']}[/yellow]")
    else:
        out(f"[dim]LLM: calling {status['provider']} / {status['model']}...[/dim]")

    if args.all:
        reports, errors = qualify_many(db, settings=settings)
        show_table(
            "Qualification run",
            ["Lead", "Report", "Score", "Band", "Provider", "Fallback"],
            [
                [
                    r.lead_id, r.report_id, r.readiness_score, r.readiness_band,
                    r.metadata.model if r.metadata else "-",
                    "yes" if (r.metadata and r.metadata.is_fallback) else "no",
                ]
                for r in reports
            ],
        )
        for error in errors:
            out(f"[red]! {error}[/red]")
        out(f"\n[green]Qualified {len(reports)} lead(s)[/green], {len(errors)} error(s).")
        return 0

    if args.lead_id is None:
        out("[red]Give a lead id, or use --all.[/red]")
        return 2

    report = qualify_lead(db, args.lead_id, settings=settings)
    _print_report(report, db)
    out(f"\n[green]Saved as report #{report.report_id}[/green] (status: {report.status})")
    return 0


def cmd_report(args: argparse.Namespace, db: Database) -> int:
    """Show a stored report."""
    report = db.get_report(args.report_id)
    colour = band_colour(report["readiness_band"])

    out(
        f"\n[bold]Report #{report['report_id']}[/bold] - lead {report['lead_id']} "
        f"({report['full_name']})\n"
        f"  created   : {report['created_at']}   status: {report['status']}\n"
        f"  readiness : [{colour}]{report['readiness_score']}/100 - "
        f"{report['readiness_band']}[/{colour}]\n"
        f"  provider  : {report['provider']} / {report['model']}"
        + ("  [yellow](offline fallback)[/yellow]" if report["is_fallback"] else "")
    )
    if report.get("input_tokens") is not None:
        out(
            f"  usage     : in={report['input_tokens']} out={report['output_tokens']} "
            f"latency={report['latency_ms']}ms"
        )

    out(f"\n[bold]Summary[/bold]\n  {report['summary']}")
    if report.get("recommended_next_step"):
        out(f"\n[bold]Recommended next step[/bold]\n  {report['recommended_next_step']}")

    show_table(
        "Fields",
        ["Field", "Req", "Weight", "Status", "Conf", "Value"],
        [
            [
                f["field_label"], "yes" if f["is_required"] else "", f["weight"],
                "present" if f["is_present"] else "missing",
                f"{f['confidence']:.2f}", (f["field_value"] or "")[:60],
            ]
            for f in report["fields"]
        ],
    )

    show_table(
        "Follow-up questions",
        ["ID", "Pri", "Field", "Status", "Question"],
        [
            [f["followup_id"], f["priority"], f["field_name"] or "-", f["status"], f["question"]]
            for f in report["followups"]
        ],
    )

    if report["reviews"]:
        show_table(
            "Counsellor reviews",
            ["ID", "Reviewer", "Decision", "When", "Notes"],
            [
                [r["review_id"], r["reviewer"], r["decision"], r["created_at"],
                 (r["notes"] or "")[:60]]
                for r in report["reviews"]
            ],
        )
    else:
        out("\n[dim]Not yet reviewed.[/dim]")

    if args.json:
        out("\n[bold]Raw LLM payload[/bold]")
        print(json.dumps(report.get("llm_payload") or {}, indent=2))
    return 0


def cmd_review(args: argparse.Namespace, db: Database) -> int:
    """Record a counsellor's review of a report."""
    review_id = record_review(
        db,
        args.report_id,
        reviewer=args.reviewer,
        decision=args.decision,
        notes=args.notes,
        lead_status=args.lead_status,
    )
    out(
        f"[green]Review #{review_id} saved[/green] - report #{args.report_id} "
        f"marked '{args.decision}' by {args.reviewer}."
    )
    return 0


def cmd_followups(args: argparse.Namespace, db: Database) -> int:
    """List or update suggested follow-up questions."""
    if args.set_status:
        if args.followup_id is None:
            out("[red]--set-status needs a followup id.[/red]")
            return 2
        db.update_followup_status(args.followup_id, args.set_status)
        out(f"[green]Follow-up #{args.followup_id} set to '{args.set_status}'.[/green]")
        return 0

    rows = db.get_followups(report_id=args.report_id, status=args.status)
    show_table(
        "Follow-up questions",
        ["ID", "Report", "Lead", "Pri", "Field", "Status", "Question"],
        [
            [r["followup_id"], r["report_id"], r["full_name"], r["priority"],
             r["field_name"] or "-", r["status"], r["question"]]
            for r in rows
        ],
    )
    return 0


def cmd_similar(args: argparse.Namespace, db: Database) -> int:
    """Find past enquiries similar to this one (TF-IDF cosine)."""
    matches = similar_past_leads(db, args.lead_id, top_n=args.top)
    show_table(
        f"Enquiries similar to lead {args.lead_id}",
        ["Lead", "Name", "Similarity", "Past band", "Past score"],
        [
            [m["lead_id"], m["full_name"], m["similarity"],
             m.get("latest_band") or "-", m.get("latest_score") or "-"]
            for m in matches
        ],
    )
    return 0


def cmd_stats(args: argparse.Namespace, db: Database) -> int:
    """Pipeline analytics."""
    metrics = overview_metrics(db)
    out("\n[bold]Pipeline overview[/bold]")
    for key, value in metrics.items():
        out(f"  {key.replace('_', ' '):<20} {value}")

    frame_table("Readiness distribution", readiness_distribution(db))
    frame_table(
        "Most frequently missing fields",
        missing_field_report(db),
        ["field_label", "is_required", "weight", "missing_count", "total_count",
         "missing_pct", "score_impact"],
    )
    frame_table(
        "Leads needing more information",
        leads_needing_info(db),
        ["lead_id", "full_name", "readiness_score", "readiness_band",
         "open_followups", "missing_required"],
    )
    frame_table(
        "Provider / model usage (audit)",
        provider_usage(db),
        ["provider", "model", "is_fallback", "run_count", "input_tokens",
         "output_tokens", "avg_latency_ms"],
    )
    frame_table("Counsellor review decisions", review_summary(db))
    out("\n[dim]Full, untruncated versions of these views: "
        "python -m leadqual export[/dim]")
    return 0


def cmd_export(args: argparse.Namespace, db: Database) -> int:
    """Export every analytical view to CSV."""
    written = export_all(db, args.directory)
    show_table("Exported", ["View", "File"], [[k, v] for k, v in written.items()])
    return 0


def cmd_add_lead(args: argparse.Namespace, db: Database) -> int:
    """Create a lead from the command line."""
    cleaned = validate_lead_payload(
        {
            "full_name": args.name,
            "email": args.email,
            "phone": args.phone,
            "source": args.source,
            "programme_interest": args.programme,
            "external_ref": args.ref,
        }
    )
    lead_id = db.insert_lead(**cleaned)
    out(f"[green]Lead #{lead_id} created[/green] for {cleaned['full_name']}.")
    if args.message:
        message = validate_message_payload(
            {"body": args.message, "channel": cleaned["source"], "direction": "inbound"},
            lead_id=lead_id,
        )
        message_id = db.insert_message(**message)
        out(f"[green]Message #{message_id} added.[/green]")
    return 0


def cmd_add_message(args: argparse.Namespace, db: Database) -> int:
    """Append a message to an existing lead."""
    db.get_lead(args.lead_id)  # existence check
    message = validate_message_payload(
        {
            "body": args.body,
            "channel": args.channel,
            "direction": args.direction,
            "sent_at": args.sent_at,
        },
        lead_id=args.lead_id,
    )
    message_id = db.insert_message(**message)
    out(f"[green]Message #{message_id} added to lead {args.lead_id}.[/green]")
    return 0


def cmd_audit(args: argparse.Namespace, db: Database) -> int:
    """Show the audit trail."""
    rows = db.get_audit_log(entity_type=args.entity, limit=args.limit)
    show_table(
        "Audit log",
        ["ID", "When", "Entity", "Entity ID", "Action", "Actor", "Details"],
        [
            [r["audit_id"], r["created_at"], r["entity_type"], r["entity_id"],
             r["action"], r["actor"], (r["details_json"] or "")[:70]]
            for r in rows
        ],
    )
    return 0


def cmd_status(args: argparse.Namespace, db: Database) -> int:
    """Show configuration and component health."""
    settings = load_settings()
    out("\n[bold]Configuration[/bold]")
    for key, value in settings.describe().items():
        out(f"  {key:<22} {value}")

    active = QualificationLLM(settings).status()
    out(f"\n[bold]Active LLM provider: {active['provider']}[/bold]")
    for key, value in active.items():
        out(f"  {key:<22} {value}")

    out("\n[bold]All providers[/bold]")
    show_table(
        "Provider availability",
        ["Provider", "SDK", "Key", "Model", "Live", "Reason"],
        [
            [
                s["provider"],
                "yes" if s["sdk_installed"] else "no",
                "yes" if s["api_key_configured"] else "no",
                s["model"],
                "yes" if s["live_calls_enabled"] else "no",
                s["reason"],
            ]
            for s in QualificationLLM.all_provider_status(settings)
        ],
        max_cell=60,
    )

    out("\n[bold]NLP backends[/bold]")
    for key, value in nlp_backend_status().items():
        out(f"  {key:<22} {value}")
    for key, value in similarity_backend_status().items():
        out(f"  {key:<22} {value}")

    out("\n[bold]Database[/bold]")
    for key, value in db.query_pipeline_overview().items():
        out(f"  {key:<22} {value}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="leadqual",
        description=(
            "Lead Qualification & Readiness Assistant - assess whether an enquiry "
            "carries enough information for a useful consultation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", help="Path to the SQLite file (overrides LEADQUAL_DB_PATH).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create the database schema.")
    p.add_argument("--reset", action="store_true", help="Drop existing tables first.")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("seed", help="Load the synthetic sample dataset.")
    p.add_argument("--file", help="Alternative JSON dataset.")
    p.add_argument("--reset", action="store_true", help="Reset the database first.")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("leads", help="List leads.")
    p.add_argument("--status", help="Filter by lead status.")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_leads)

    p = sub.add_parser("show", help="Show one lead and its messages.")
    p.add_argument("lead_id", type=int)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("analyse", help="NLP + scoring only (no API call).")
    p.add_argument("lead_id", type=int)
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("qualify", help="Run the full pipeline and save a report.")
    p.add_argument("lead_id", type=int, nargs="?")
    p.add_argument("--all", action="store_true", help="Qualify every lead with messages.")
    p.set_defaults(func=cmd_qualify)

    p = sub.add_parser("report", help="Show a stored qualification report.")
    p.add_argument("report_id", type=int)
    p.add_argument("--json", action="store_true", help="Also print the raw LLM payload.")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("review", help="Save a counsellor review.")
    p.add_argument("report_id", type=int)
    p.add_argument("--reviewer", required=True)
    p.add_argument(
        "--decision", required=True,
        choices=["approved", "edited", "rejected", "escalated"],
    )
    p.add_argument("--notes")
    p.add_argument(
        "--lead-status", dest="lead_status",
        choices=["new", "in_progress", "qualified", "needs_info", "closed"],
        help="Optionally move the lead to this status.",
    )
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("followups", help="List or update follow-up questions.")
    p.add_argument("--report-id", dest="report_id", type=int)
    p.add_argument("--status", help="Filter by follow-up status.")
    p.add_argument("--followup-id", dest="followup_id", type=int)
    p.add_argument(
        "--set-status", dest="set_status",
        choices=["suggested", "accepted", "dismissed", "asked", "answered"],
    )
    p.set_defaults(func=cmd_followups)

    p = sub.add_parser("similar", help="Find similar past enquiries (TF-IDF).")
    p.add_argument("lead_id", type=int)
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(func=cmd_similar)

    p = sub.add_parser("stats", help="Pipeline analytics.")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="Export analytics to CSV.")
    p.add_argument("directory", nargs="?", default="exports")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("add-lead", help="Create a lead.")
    p.add_argument("--name", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--phone")
    p.add_argument("--source", default="web_form")
    p.add_argument("--programme")
    p.add_argument("--ref")
    p.add_argument("--message", help="Also add this as the first inbound message.")
    p.set_defaults(func=cmd_add_lead)

    p = sub.add_parser("add-message", help="Append a message to a lead.")
    p.add_argument("lead_id", type=int)
    p.add_argument("body")
    p.add_argument("--channel", default="email")
    p.add_argument("--direction", default="inbound", choices=["inbound", "outbound"])
    p.add_argument("--sent-at", dest="sent_at")
    p.set_defaults(func=cmd_add_message)

    p = sub.add_parser("audit", help="Show the audit trail.")
    p.add_argument("--entity", help="Filter by entity type.")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("status", help="Show configuration and component health.")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    db_path = Path(args.db) if args.db else settings.db_path

    try:
        with open_database(db_path) as db:
            return int(args.func(args, db) or 0)
    except LeadQualError as exc:
        out(f"[red]Error:[/red] {exc}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        out("\n[yellow]Interrupted.[/yellow]")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
