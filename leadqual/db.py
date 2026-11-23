"""SQLite persistence using the standard-library ``sqlite3`` module.

Entities
--------
``leads``                 one row per enquiry
``lead_messages``         the conversation history (1:N from leads)
``qualification_reports`` one row per qualification run (1:N from leads)
``qualification_fields``  per-field extraction results (1:N from reports)
``followups``             suggested questions (1:N from reports)
``counsellor_reviews``    the human sign-off (1:N from reports)
``audit_log``             append-only record of who did what, when

Every table has a primary key and timestamps; child tables carry foreign keys
with ``ON DELETE CASCADE`` so removing a lead cleans up after itself.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .exceptions import DatabaseError, RecordNotFoundError
from .models import FIELD_SPECS, FollowUpQuestion, utc_now_iso

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

-- ---------------------------------------------------------------- leads ---
CREATE TABLE IF NOT EXISTS leads (
    lead_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    external_ref       TEXT UNIQUE,
    full_name          TEXT NOT NULL,
    email              TEXT NOT NULL,
    phone              TEXT,
    source             TEXT NOT NULL DEFAULT 'web_form',
    programme_interest TEXT,
    status             TEXT NOT NULL DEFAULT 'new'
                       CHECK (status IN ('new','in_progress','qualified','needs_info','closed')),
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_email  ON leads(email);

-- -------------------------------------------------------- lead_messages ---
CREATE TABLE IF NOT EXISTS lead_messages (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id     INTEGER NOT NULL REFERENCES leads(lead_id) ON DELETE CASCADE,
    channel     TEXT NOT NULL DEFAULT 'email',
    direction   TEXT NOT NULL DEFAULT 'inbound'
                CHECK (direction IN ('inbound','outbound')),
    body        TEXT NOT NULL,
    sent_at     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_lead ON lead_messages(lead_id, sent_at);

-- ------------------------------------------------ qualification_reports ---
CREATE TABLE IF NOT EXISTS qualification_reports (
    report_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id               INTEGER NOT NULL REFERENCES leads(lead_id) ON DELETE CASCADE,
    readiness_score       REAL NOT NULL CHECK (readiness_score BETWEEN 0 AND 100),
    readiness_band        TEXT NOT NULL
                          CHECK (readiness_band IN ('READY','NEEDS_INFO','NOT_READY')),
    summary               TEXT NOT NULL,
    recommended_next_step TEXT,
    missing_required_json TEXT NOT NULL DEFAULT '[]',
    nlp_json              TEXT NOT NULL DEFAULT '{}',
    score_json            TEXT NOT NULL DEFAULT '{}',
    llm_payload_json      TEXT NOT NULL DEFAULT '{}',
    -- provider / model metadata (audit)
    provider              TEXT NOT NULL,
    model                 TEXT NOT NULL,
    is_fallback           INTEGER NOT NULL DEFAULT 0 CHECK (is_fallback IN (0,1)),
    response_id           TEXT,
    stop_reason           TEXT,
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    latency_ms            INTEGER,
    provider_note         TEXT,
    status                TEXT NOT NULL DEFAULT 'awaiting_review'
                          CHECK (status IN ('awaiting_review','reviewed','archived')),
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_lead   ON qualification_reports(lead_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_status ON qualification_reports(status);

-- ------------------------------------------------- qualification_fields ---
CREATE TABLE IF NOT EXISTS qualification_fields (
    field_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id           INTEGER NOT NULL REFERENCES qualification_reports(report_id) ON DELETE CASCADE,
    lead_id             INTEGER NOT NULL REFERENCES leads(lead_id) ON DELETE CASCADE,
    field_name          TEXT NOT NULL,
    field_label         TEXT NOT NULL,
    is_present          INTEGER NOT NULL CHECK (is_present IN (0,1)),
    field_value         TEXT,
    confidence          REAL NOT NULL DEFAULT 0 CHECK (confidence BETWEEN 0 AND 1),
    weight              INTEGER NOT NULL,
    is_required         INTEGER NOT NULL CHECK (is_required IN (0,1)),
    extraction_method   TEXT NOT NULL DEFAULT 'rule',
    evidence_message_id INTEGER REFERENCES lead_messages(message_id) ON DELETE SET NULL,
    evidence_text       TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (report_id, field_name)
);
CREATE INDEX IF NOT EXISTS idx_fields_report ON qualification_fields(report_id);
CREATE INDEX IF NOT EXISTS idx_fields_name   ON qualification_fields(field_name, is_present);

-- ------------------------------------------------------------ followups ---
CREATE TABLE IF NOT EXISTS followups (
    followup_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL REFERENCES qualification_reports(report_id) ON DELETE CASCADE,
    lead_id     INTEGER NOT NULL REFERENCES leads(lead_id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    field_name  TEXT,
    priority    INTEGER NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    rationale   TEXT,
    status      TEXT NOT NULL DEFAULT 'suggested'
                CHECK (status IN ('suggested','accepted','dismissed','asked','answered')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_followups_report ON followups(report_id, priority);
CREATE INDEX IF NOT EXISTS idx_followups_status ON followups(status);

-- --------------------------------------------------- counsellor_reviews ---
CREATE TABLE IF NOT EXISTS counsellor_reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL REFERENCES qualification_reports(report_id) ON DELETE CASCADE,
    lead_id     INTEGER NOT NULL REFERENCES leads(lead_id) ON DELETE CASCADE,
    reviewer    TEXT NOT NULL,
    decision    TEXT NOT NULL
                CHECK (decision IN ('approved','edited','rejected','escalated')),
    notes       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_report ON counsellor_reviews(report_id);

-- ------------------------------------------------------------ audit_log ---
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type  TEXT NOT NULL,
    entity_id    INTEGER,
    action       TEXT NOT NULL,
    actor        TEXT NOT NULL DEFAULT 'system',
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
"""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


class Database:
    """A thin, explicit wrapper around a SQLite connection.

    Usage::

        with Database("data/leadqual.db") as db:
            db.initialise()
            lead_id = db.insert_lead(...)

    The class is deliberately not an ORM: every statement is visible SQL so
    the schema, keys and queries can be inspected directly.
    """

    def __init__(self, path: str | Path, *, create_dirs: bool = True) -> None:
        self.path = Path(path)
        if create_dirs and str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(
                str(self.path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:  # pragma: no cover - environment specific
            raise DatabaseError(f"Could not open database at {self.path}: {exc}") from exc

        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self.conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Run a block of statements atomically.

        Commits on success, rolls back and re-raises as
        :class:`~leadqual.exceptions.DatabaseError` on failure.
        """
        cursor = self.conn.cursor()
        try:
            yield cursor
        except sqlite3.Error as exc:
            self.conn.rollback()
            raise DatabaseError(f"Transaction failed: {exc}") from exc
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
        finally:
            cursor.close()

    def initialise(self) -> None:
        """Create the schema if it does not already exist (idempotent)."""
        try:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not initialise schema: {exc}") from exc

    def reset(self) -> None:
        """Drop every application table and recreate the schema."""
        tables = (
            "audit_log",
            "counsellor_reviews",
            "followups",
            "qualification_fields",
            "qualification_reports",
            "lead_messages",
            "leads",
            "schema_meta",
        )
        try:
            self.conn.execute("PRAGMA foreign_keys = OFF;")
            for table in tables:
                self.conn.execute(f"DROP TABLE IF EXISTS {table};")
            self.conn.commit()
            self.conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not reset database: {exc}") from exc
        self.initialise()

    # ==================================================================
    # CRUD - leads
    # ==================================================================
    def insert_lead(
        self,
        *,
        full_name: str,
        email: str,
        phone: str | None = None,
        source: str = "web_form",
        programme_interest: str | None = None,
        status: str = "new",
        external_ref: str | None = None,
    ) -> int:
        """Create a lead and return its new ``lead_id``."""
        now = utc_now_iso()
        try:
            with self.transaction() as cur:
                cur.execute(
                    """
                    INSERT INTO leads (external_ref, full_name, email, phone, source,
                                       programme_interest, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (external_ref, full_name, email, phone, source,
                     programme_interest, status, now, now),
                )
                lead_id = int(cur.lastrowid or 0)
        except DatabaseError as exc:
            if "UNIQUE" in str(exc).upper():
                raise DatabaseError(
                    f"A lead with external_ref {external_ref!r} already exists."
                ) from exc
            raise
        self.log_audit("lead", lead_id, "created", details={"email": email, "source": source})
        return lead_id

    def get_lead(self, lead_id: int) -> dict[str, Any]:
        """Fetch one lead by id.

        Raises:
            RecordNotFoundError: If no such lead exists.
        """
        row = self.conn.execute(
            "SELECT * FROM leads WHERE lead_id = ?", (lead_id,)
        ).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise RecordNotFoundError("lead", lead_id)
        return result

    def find_lead_by_email(self, email: str) -> dict[str, Any] | None:
        """Fetch the most recent lead with this email, or ``None``."""
        row = self.conn.execute(
            "SELECT * FROM leads WHERE email = ? ORDER BY lead_id DESC LIMIT 1",
            (email.lower(),),
        ).fetchone()
        return _row_to_dict(row)

    def list_leads(
        self, *, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List leads with message counts and latest readiness, newest first."""
        sql = """
            SELECT l.*,
                   (SELECT COUNT(*) FROM lead_messages m WHERE m.lead_id = l.lead_id)
                       AS message_count,
                   (SELECT COUNT(*) FROM qualification_reports r WHERE r.lead_id = l.lead_id)
                       AS report_count,
                   (SELECT r.readiness_score FROM qualification_reports r
                     WHERE r.lead_id = l.lead_id ORDER BY r.report_id DESC LIMIT 1)
                       AS latest_score,
                   (SELECT r.readiness_band FROM qualification_reports r
                     WHERE r.lead_id = l.lead_id ORDER BY r.report_id DESC LIMIT 1)
                       AS latest_band
            FROM leads l
        """
        params: list[Any] = []
        if status:
            sql += " WHERE l.status = ?"
            params.append(status)
        sql += " ORDER BY l.lead_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return _rows_to_dicts(self.conn.execute(sql, params).fetchall())

    def update_lead_status(self, lead_id: int, status: str, *, actor: str = "system") -> None:
        """Update a lead's workflow status."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE leads SET status = ?, updated_at = ? WHERE lead_id = ?",
                (status, utc_now_iso(), lead_id),
            )
            if cur.rowcount == 0:
                raise RecordNotFoundError("lead", lead_id)
        self.log_audit("lead", lead_id, "status_changed", actor=actor,
                       details={"status": status})

    def delete_lead(self, lead_id: int, *, actor: str = "system") -> None:
        """Delete a lead and every child record (cascade)."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
            if cur.rowcount == 0:
                raise RecordNotFoundError("lead", lead_id)
        self.log_audit("lead", lead_id, "deleted", actor=actor)

    # ==================================================================
    # CRUD - messages
    # ==================================================================
    def insert_message(
        self,
        *,
        lead_id: int,
        body: str,
        channel: str = "email",
        direction: str = "inbound",
        sent_at: str | None = None,
    ) -> int:
        """Append a message to a lead's history and return its id."""
        now = utc_now_iso()
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO lead_messages (lead_id, channel, direction, body, sent_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lead_id, channel, direction, body, sent_at or now, now),
            )
            message_id = int(cur.lastrowid or 0)
            cur.execute(
                "UPDATE leads SET updated_at = ? WHERE lead_id = ?", (now, lead_id)
            )
        return message_id

    def get_messages(
        self, lead_id: int, *, direction: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a lead's messages in chronological order."""
        sql = "SELECT * FROM lead_messages WHERE lead_id = ?"
        params: list[Any] = [lead_id]
        if direction:
            sql += " AND direction = ?"
            params.append(direction)
        sql += " ORDER BY COALESCE(sent_at, created_at) ASC, message_id ASC"
        return _rows_to_dicts(self.conn.execute(sql, params).fetchall())

    def delete_message(self, message_id: int) -> None:
        """Delete a single message."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM lead_messages WHERE message_id = ?", (message_id,))
            if cur.rowcount == 0:
                raise RecordNotFoundError("lead_message", message_id)

    # ==================================================================
    # CRUD - reports (+ fields and followups, written atomically)
    # ==================================================================
    def save_report(self, report) -> int:
        """Persist a :class:`~leadqual.models.QualificationReport`.

        The report, its per-field rows and its follow-up questions are
        written in a single transaction so a partial run never lands.

        Args:
            report: A populated ``QualificationReport``.

        Returns:
            The new ``report_id``.
        """
        now = utc_now_iso()
        meta = report.metadata
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO qualification_reports (
                    lead_id, readiness_score, readiness_band, summary,
                    recommended_next_step, missing_required_json, nlp_json, score_json,
                    llm_payload_json,
                    provider, model, is_fallback, response_id, stop_reason,
                    input_tokens, output_tokens, latency_ms, provider_note,
                    status, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    report.lead_id,
                    float(report.readiness_score),
                    report.readiness_band,
                    report.summary,
                    report.recommended_next_step,
                    json.dumps(report.score.missing_required),
                    json.dumps(report.nlp.to_dict()),
                    json.dumps(report.score.to_dict()),
                    json.dumps(report.llm_payload or {}, default=str),
                    meta.provider if meta else "unknown",
                    meta.model if meta else "unknown",
                    1 if (meta and meta.is_fallback) else 0,
                    meta.response_id if meta else None,
                    meta.stop_reason if meta else None,
                    meta.input_tokens if meta else None,
                    meta.output_tokens if meta else None,
                    meta.latency_ms if meta else None,
                    meta.note if meta else None,
                    report.status,
                    now,
                ),
            )
            report_id = int(cur.lastrowid or 0)

            for spec in FIELD_SPECS:
                extracted = report.nlp.fields.get(spec.name)
                cur.execute(
                    """
                    INSERT INTO qualification_fields (
                        report_id, lead_id, field_name, field_label, is_present,
                        field_value, confidence, weight, is_required,
                        extraction_method, evidence_message_id, evidence_text, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        report_id,
                        report.lead_id,
                        spec.name,
                        spec.label,
                        1 if (extracted and extracted.is_present) else 0,
                        extracted.value if extracted else None,
                        float(extracted.confidence) if extracted else 0.0,
                        spec.weight,
                        1 if spec.required else 0,
                        extracted.method if extracted else "rule",
                        extracted.evidence_message_id if extracted else None,
                        extracted.evidence_text if extracted else None,
                        now,
                    ),
                )

            for followup in report.followups:
                cur.execute(
                    """
                    INSERT INTO followups (report_id, lead_id, question, field_name,
                                           priority, rationale, status, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        report_id,
                        report.lead_id,
                        followup.question,
                        followup.field_name,
                        int(followup.priority),
                        followup.rationale,
                        "suggested",
                        now,
                        now,
                    ),
                )

        report.report_id = report_id
        report.created_at = now
        self.log_audit(
            "qualification_report",
            report_id,
            "created",
            details={
                "lead_id": report.lead_id,
                "score": report.readiness_score,
                "band": report.readiness_band,
                "provider": meta.provider if meta else "unknown",
                "model": meta.model if meta else "unknown",
                "is_fallback": bool(meta and meta.is_fallback),
            },
        )
        return report_id

    def get_report(self, report_id: int) -> dict[str, Any]:
        """Fetch one report with its fields, follow-ups and reviews attached."""
        row = self.conn.execute(
            """
            SELECT r.*, l.full_name, l.email, l.programme_interest
            FROM qualification_reports r
            JOIN leads l ON l.lead_id = r.lead_id
            WHERE r.report_id = ?
            """,
            (report_id,),
        ).fetchone()
        report = _row_to_dict(row)
        if report is None:
            raise RecordNotFoundError("qualification_report", report_id)

        report["fields"] = self.get_report_fields(report_id)
        report["followups"] = self.get_followups(report_id)
        report["reviews"] = self.get_reviews(report_id)
        for key in ("missing_required_json", "nlp_json", "score_json", "llm_payload_json"):
            try:
                report[key.removesuffix("_json")] = json.loads(report[key])
            except (json.JSONDecodeError, TypeError):
                report[key.removesuffix("_json")] = None
        return report

    def get_report_fields(self, report_id: int) -> list[dict[str, Any]]:
        """Per-field extraction rows for a report, heaviest weight first."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                SELECT * FROM qualification_fields
                WHERE report_id = ?
                ORDER BY is_required DESC, weight DESC, field_name ASC
                """,
                (report_id,),
            ).fetchall()
        )

    def get_latest_report(self, lead_id: int) -> dict[str, Any] | None:
        """The most recent report for a lead, or ``None`` if never run."""
        row = self.conn.execute(
            """
            SELECT report_id FROM qualification_reports
            WHERE lead_id = ? ORDER BY report_id DESC LIMIT 1
            """,
            (lead_id,),
        ).fetchone()
        return self.get_report(int(row["report_id"])) if row else None

    def list_reports(
        self, *, lead_id: int | None = None, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List report summaries, newest first."""
        sql = """
            SELECT r.report_id, r.lead_id, l.full_name, l.email,
                   r.readiness_score, r.readiness_band, r.status,
                   r.provider, r.model, r.is_fallback, r.created_at,
                   (SELECT COUNT(*) FROM followups f WHERE f.report_id = r.report_id)
                       AS followup_count
            FROM qualification_reports r
            JOIN leads l ON l.lead_id = r.lead_id
            WHERE 1 = 1
        """
        params: list[Any] = []
        if lead_id is not None:
            sql += " AND r.lead_id = ?"
            params.append(lead_id)
        if status:
            sql += " AND r.status = ?"
            params.append(status)
        sql += " ORDER BY r.report_id DESC LIMIT ?"
        params.append(limit)
        return _rows_to_dicts(self.conn.execute(sql, params).fetchall())

    def update_report_status(self, report_id: int, status: str) -> None:
        """Move a report through its lifecycle."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE qualification_reports SET status = ? WHERE report_id = ?",
                (status, report_id),
            )
            if cur.rowcount == 0:
                raise RecordNotFoundError("qualification_report", report_id)

    def delete_report(self, report_id: int) -> None:
        """Delete a report and its children."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM qualification_reports WHERE report_id = ?", (report_id,))
            if cur.rowcount == 0:
                raise RecordNotFoundError("qualification_report", report_id)
        self.log_audit("qualification_report", report_id, "deleted")

    # ==================================================================
    # CRUD - followups
    # ==================================================================
    def get_followups(
        self, report_id: int | None = None, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Follow-up questions, highest priority first."""
        sql = """
            SELECT f.*, l.full_name
            FROM followups f
            JOIN leads l ON l.lead_id = f.lead_id
            WHERE 1 = 1
        """
        params: list[Any] = []
        if report_id is not None:
            sql += " AND f.report_id = ?"
            params.append(report_id)
        if status:
            sql += " AND f.status = ?"
            params.append(status)
        sql += " ORDER BY f.priority ASC, f.followup_id ASC"
        return _rows_to_dicts(self.conn.execute(sql, params).fetchall())

    def update_followup_status(self, followup_id: int, status: str) -> None:
        """Accept, dismiss or tick off a suggested question."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE followups SET status = ?, updated_at = ? WHERE followup_id = ?",
                (status, utc_now_iso(), followup_id),
            )
            if cur.rowcount == 0:
                raise RecordNotFoundError("followup", followup_id)

    def add_followup(self, report_id: int, lead_id: int, followup: FollowUpQuestion) -> int:
        """Add a counsellor-authored question to an existing report."""
        now = utc_now_iso()
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO followups (report_id, lead_id, question, field_name,
                                       priority, rationale, status, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (report_id, lead_id, followup.question, followup.field_name,
                 int(followup.priority), followup.rationale, "accepted", now, now),
            )
            return int(cur.lastrowid or 0)

    # ==================================================================
    # CRUD - counsellor reviews
    # ==================================================================
    def save_review(
        self,
        *,
        report_id: int,
        reviewer: str,
        decision: str,
        notes: str | None = None,
        lead_status: str | None = None,
    ) -> int:
        """Record a counsellor's sign-off on a report.

        Also marks the report ``reviewed`` and, when ``lead_status`` is given,
        moves the lead on - all in one transaction.
        """
        row = self.conn.execute(
            "SELECT lead_id FROM qualification_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError("qualification_report", report_id)
        lead_id = int(row["lead_id"])
        now = utc_now_iso()

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO counsellor_reviews (report_id, lead_id, reviewer, decision,
                                                notes, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (report_id, lead_id, reviewer, decision, notes, now),
            )
            review_id = int(cur.lastrowid or 0)
            cur.execute(
                "UPDATE qualification_reports SET status = 'reviewed' WHERE report_id = ?",
                (report_id,),
            )
            if lead_status:
                cur.execute(
                    "UPDATE leads SET status = ?, updated_at = ? WHERE lead_id = ?",
                    (lead_status, now, lead_id),
                )

        self.log_audit(
            "counsellor_review",
            review_id,
            "created",
            actor=reviewer,
            details={"report_id": report_id, "decision": decision},
        )
        return review_id

    def get_reviews(self, report_id: int) -> list[dict[str, Any]]:
        """All reviews recorded against a report, newest first."""
        return _rows_to_dicts(
            self.conn.execute(
                "SELECT * FROM counsellor_reviews WHERE report_id = ? ORDER BY review_id DESC",
                (report_id,),
            ).fetchall()
        )

    # ==================================================================
    # Audit log
    # ==================================================================
    def log_audit(
        self,
        entity_type: str,
        entity_id: int | None,
        action: str,
        *,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> int:
        """Append an entry to the audit log (never raises on serialisation)."""
        try:
            payload = json.dumps(details or {}, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            payload = "{}"
        try:
            cur = self.conn.execute(
                """
                INSERT INTO audit_log (entity_type, entity_id, action, actor,
                                       details_json, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (entity_type, entity_id, action, actor, payload, utc_now_iso()),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not write audit log: {exc}") from exc

    def get_audit_log(
        self, *, entity_type: str | None = None, entity_id: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Read back the audit trail, newest first."""
        sql = "SELECT * FROM audit_log WHERE 1 = 1"
        params: list[Any] = []
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if entity_id is not None:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        sql += " ORDER BY audit_id DESC LIMIT ?"
        params.append(limit)
        return _rows_to_dicts(self.conn.execute(sql, params).fetchall())

    # ==================================================================
    # Analytical queries (joins / aggregates used by the reporting layer)
    # ==================================================================
    def query_readiness_distribution(self) -> list[dict[str, Any]]:
        """How many leads sit in each readiness band, using their latest report."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                WITH latest AS (
                    SELECT lead_id, MAX(report_id) AS report_id
                    FROM qualification_reports GROUP BY lead_id
                )
                SELECT r.readiness_band          AS band,
                       COUNT(*)                  AS lead_count,
                       ROUND(AVG(r.readiness_score), 1) AS avg_score,
                       MIN(r.readiness_score)    AS min_score,
                       MAX(r.readiness_score)    AS max_score
                FROM latest
                JOIN qualification_reports r ON r.report_id = latest.report_id
                GROUP BY r.readiness_band
                ORDER BY avg_score DESC
                """
            ).fetchall()
        )

    def query_missing_field_frequency(self) -> list[dict[str, Any]]:
        """Which qualification fields are most often absent (latest report only)."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                WITH latest AS (
                    SELECT lead_id, MAX(report_id) AS report_id
                    FROM qualification_reports GROUP BY lead_id
                )
                SELECT q.field_name,
                       q.field_label,
                       q.is_required,
                       q.weight,
                       SUM(CASE WHEN q.is_present = 0 THEN 1 ELSE 0 END) AS missing_count,
                       COUNT(*)                                          AS total_count,
                       ROUND(100.0 * SUM(CASE WHEN q.is_present = 0 THEN 1 ELSE 0 END)
                             / COUNT(*), 1)                              AS missing_pct
                FROM latest
                JOIN qualification_fields q ON q.report_id = latest.report_id
                GROUP BY q.field_name, q.field_label, q.is_required, q.weight
                ORDER BY missing_pct DESC, q.weight DESC
                """
            ).fetchall()
        )

    def query_leads_needing_info(self, *, threshold: float = 75.0) -> list[dict[str, Any]]:
        """Leads whose latest report falls below ``threshold``, worst first."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                WITH latest AS (
                    SELECT lead_id, MAX(report_id) AS report_id
                    FROM qualification_reports GROUP BY lead_id
                )
                SELECT l.lead_id, l.full_name, l.email, l.status AS lead_status,
                       r.report_id, r.readiness_score, r.readiness_band, r.created_at,
                       (SELECT COUNT(*) FROM followups f
                         WHERE f.report_id = r.report_id AND f.status = 'suggested')
                           AS open_followups,
                       (SELECT GROUP_CONCAT(q.field_label, '; ') FROM qualification_fields q
                         WHERE q.report_id = r.report_id AND q.is_present = 0
                           AND q.is_required = 1)
                           AS missing_required
                FROM latest
                JOIN qualification_reports r ON r.report_id = latest.report_id
                JOIN leads l ON l.lead_id = r.lead_id
                WHERE r.readiness_score < ?
                ORDER BY r.readiness_score ASC
                """,
                (threshold,),
            ).fetchall()
        )

    def query_provider_usage(self) -> list[dict[str, Any]]:
        """Token spend and latency grouped by provider/model - the audit view."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                SELECT provider,
                       model,
                       is_fallback,
                       COUNT(*)                        AS run_count,
                       SUM(COALESCE(input_tokens, 0))  AS input_tokens,
                       SUM(COALESCE(output_tokens, 0)) AS output_tokens,
                       ROUND(AVG(COALESCE(latency_ms, 0)), 0) AS avg_latency_ms,
                       MIN(created_at)                 AS first_run,
                       MAX(created_at)                 AS last_run
                FROM qualification_reports
                GROUP BY provider, model, is_fallback
                ORDER BY run_count DESC
                """
            ).fetchall()
        )

    def query_review_summary(self) -> list[dict[str, Any]]:
        """Counsellor decisions and the average score they were given."""
        return _rows_to_dicts(
            self.conn.execute(
                """
                SELECT cr.decision,
                       COUNT(*)                            AS review_count,
                       ROUND(AVG(r.readiness_score), 1)    AS avg_score_reviewed,
                       COUNT(DISTINCT cr.reviewer)         AS distinct_reviewers
                FROM counsellor_reviews cr
                JOIN qualification_reports r ON r.report_id = cr.report_id
                GROUP BY cr.decision
                ORDER BY review_count DESC
                """
            ).fetchall()
        )

    def query_pipeline_overview(self) -> dict[str, Any]:
        """Headline counts for the dashboard."""
        row = self.conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM leads)                                      AS total_leads,
              (SELECT COUNT(*) FROM lead_messages)                              AS total_messages,
              (SELECT COUNT(*) FROM qualification_reports)                      AS total_reports,
              (SELECT COUNT(*) FROM qualification_reports
                WHERE status = 'awaiting_review')                               AS awaiting_review,
              (SELECT COUNT(*) FROM counsellor_reviews)                         AS total_reviews,
              (SELECT COUNT(*) FROM followups WHERE status = 'suggested')       AS open_followups,
              (SELECT ROUND(AVG(readiness_score), 1) FROM qualification_reports) AS avg_score
            """
        ).fetchone()
        return _row_to_dict(row) or {}


def open_database(path: str | Path, *, initialise: bool = True) -> Database:
    """Convenience factory: open a database and ensure the schema exists."""
    db = Database(path)
    if initialise:
        db.initialise()
    return db
