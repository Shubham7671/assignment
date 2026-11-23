"""Tests for the SQLite persistence layer."""

from __future__ import annotations

import sqlite3

import pytest

from leadqual.db import Database
from leadqual.exceptions import DatabaseError, RecordNotFoundError
from leadqual.models import FollowUpQuestion, LLMMetadata, QualificationReport
from leadqual.nlp.extract import analyse_lead
from leadqual.scoring import score_completeness


def build_report(lead_id: int, *, score_value: float = 60.0) -> QualificationReport:
    """A minimal but valid report object for persistence tests."""
    nlp = analyse_lead(
        [{"message_id": 1, "body": "I want an MBA in January.", "direction": "inbound"}]
    )
    score = score_completeness(nlp)
    band = "READY" if score_value >= 75 else "NEEDS_INFO" if score_value >= 45 else "NOT_READY"
    return QualificationReport(
        lead_id=lead_id,
        summary="A test summary long enough to be realistic for the counsellor.",
        readiness_score=score_value,
        readiness_band=band,
        nlp=nlp,
        score=score,
        followups=[
            FollowUpQuestion("What is your budget?", "budget_indication", 1, "Needed."),
            FollowUpQuestion("When would you start?", "start_timeline", 2, "Needed."),
        ],
        metadata=LLMMetadata(
            provider="anthropic",
            model="claude-opus-5-5",
            response_id="msg_test",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            latency_ms=900,
        ),
        recommended_next_step="Book the consultation.",
        llm_payload={"qualification_summary": "A test summary."},
    )


class TestSchema:
    def test_initialise_is_idempotent(self, db: Database):
        db.initialise()
        db.initialise()
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "leads", "lead_messages", "qualification_reports",
            "qualification_fields", "followups", "counsellor_reviews", "audit_log",
        } <= tables

    def test_foreign_keys_are_enforced(self, db: Database):
        with pytest.raises(DatabaseError):
            db.insert_message(lead_id=9999, body="Orphan message")

    def test_check_constraint_rejects_bad_status(self, db: Database):
        lead_id = db.insert_lead(full_name="A B", email="a@example.com")
        with pytest.raises(DatabaseError):
            db.update_lead_status(lead_id, "banana")

    def test_reset_clears_everything(self, db: Database):
        db.insert_lead(full_name="A B", email="a@example.com")
        db.reset()
        assert db.list_leads() == []


class TestLeadCrud:
    def test_create_and_read(self, db: Database):
        lead_id = db.insert_lead(
            full_name="Ada Lovelace",
            email="ada@example.com",
            phone="+44 7700 900001",
            programme_interest="MSc Computing",
            external_ref="CRM-1",
        )
        lead = db.get_lead(lead_id)
        assert lead["full_name"] == "Ada Lovelace"
        assert lead["status"] == "new"
        assert lead["created_at"] and lead["updated_at"]

    def test_missing_lead_raises(self, db: Database):
        with pytest.raises(RecordNotFoundError):
            db.get_lead(4242)

    def test_duplicate_external_ref_is_rejected(self, db: Database):
        db.insert_lead(full_name="A B", email="a@example.com", external_ref="CRM-1")
        with pytest.raises(DatabaseError, match="already exists"):
            db.insert_lead(full_name="C D", email="c@example.com", external_ref="CRM-1")

    def test_update_status(self, db: Database):
        lead_id = db.insert_lead(full_name="A B", email="a@example.com")
        db.update_lead_status(lead_id, "qualified")
        assert db.get_lead(lead_id)["status"] == "qualified"

    def test_find_by_email_is_case_insensitive(self, db: Database):
        db.insert_lead(full_name="A B", email="mixed@example.com")
        assert db.find_lead_by_email("MIXED@example.com") is not None
        assert db.find_lead_by_email("nobody@example.com") is None

    def test_delete_cascades_to_children(self, db: Database, sample_lead: int):
        db.save_report(build_report(sample_lead))
        db.delete_lead(sample_lead)
        assert db.conn.execute("SELECT COUNT(*) FROM lead_messages").fetchone()[0] == 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM qualification_reports"
        ).fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM followups").fetchone()[0] == 0

    def test_list_leads_includes_aggregates(self, db: Database, sample_lead: int):
        db.save_report(build_report(sample_lead, score_value=80.0))
        row = db.list_leads()[0]
        assert row["message_count"] == 1
        assert row["report_count"] == 1
        assert row["latest_score"] == 80.0
        assert row["latest_band"] == "READY"

    def test_list_leads_filters_by_status(self, db: Database):
        a = db.insert_lead(full_name="A B", email="a@example.com")
        db.insert_lead(full_name="C D", email="c@example.com")
        db.update_lead_status(a, "qualified")
        assert len(db.list_leads(status="qualified")) == 1


class TestMessages:
    def test_messages_come_back_in_order(self, db: Database):
        lead_id = db.insert_lead(full_name="A B", email="a@example.com")
        db.insert_message(lead_id=lead_id, body="Second", sent_at="2026-02-01T00:00:00+00:00")
        db.insert_message(lead_id=lead_id, body="First", sent_at="2026-01-01T00:00:00+00:00")
        bodies = [m["body"] for m in db.get_messages(lead_id)]
        assert bodies == ["First", "Second"]

    def test_direction_filter(self, db: Database):
        lead_id = db.insert_lead(full_name="A B", email="a@example.com")
        db.insert_message(lead_id=lead_id, body="In", direction="inbound")
        db.insert_message(lead_id=lead_id, body="Out", direction="outbound")
        assert len(db.get_messages(lead_id, direction="inbound")) == 1

    def test_inserting_a_message_touches_the_lead(self, db: Database):
        lead_id = db.insert_lead(full_name="A B", email="a@example.com")
        before = db.get_lead(lead_id)["updated_at"]
        db.conn.execute(
            "UPDATE leads SET updated_at = '2000-01-01T00:00:00+00:00' WHERE lead_id = ?",
            (lead_id,),
        )
        db.conn.commit()
        db.insert_message(lead_id=lead_id, body="Hello")
        assert db.get_lead(lead_id)["updated_at"] != "2000-01-01T00:00:00+00:00"
        assert before  # sanity


class TestReports:
    def test_save_writes_report_fields_and_followups(self, db: Database, sample_lead: int):
        report_id = db.save_report(build_report(sample_lead))
        stored = db.get_report(report_id)
        assert stored["lead_id"] == sample_lead
        assert stored["provider"] == "anthropic"
        assert stored["model"] == "claude-opus-5-5"
        assert stored["is_fallback"] == 0
        assert stored["input_tokens"] == 100
        assert len(stored["fields"]) == 10  # one row per field in the catalogue
        assert len(stored["followups"]) == 2
        assert stored["llm_payload"]["qualification_summary"]

    def test_save_sets_ids_on_the_object(self, db: Database, sample_lead: int):
        report = build_report(sample_lead)
        db.save_report(report)
        assert report.report_id is not None
        assert report.created_at is not None

    def test_get_latest_report_returns_the_newest(self, db: Database, sample_lead: int):
        db.save_report(build_report(sample_lead, score_value=40.0))
        second = db.save_report(build_report(sample_lead, score_value=90.0))
        assert db.get_latest_report(sample_lead)["report_id"] == second

    def test_latest_report_is_none_when_never_run(self, db: Database, sample_lead: int):
        assert db.get_latest_report(sample_lead) is None

    def test_missing_report_raises(self, db: Database):
        with pytest.raises(RecordNotFoundError):
            db.get_report(999)

    def test_score_check_constraint(self, db: Database, sample_lead: int):
        report = build_report(sample_lead)
        report.readiness_score = 150.0
        with pytest.raises(DatabaseError):
            db.save_report(report)

    def test_failed_save_is_rolled_back(self, db: Database, sample_lead: int):
        report = build_report(sample_lead)
        report.readiness_band = "NOT_A_BAND"
        with pytest.raises(DatabaseError):
            db.save_report(report)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM qualification_reports"
        ).fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM followups").fetchone()[0] == 0


class TestFollowupsAndReviews:
    def test_update_followup_status(self, db: Database, sample_lead: int):
        report_id = db.save_report(build_report(sample_lead))
        followup_id = db.get_followups(report_id)[0]["followup_id"]
        db.update_followup_status(followup_id, "asked")
        assert db.get_followups(report_id)[0]["status"] == "asked"

    def test_followups_ordered_by_priority(self, db: Database, sample_lead: int):
        report_id = db.save_report(build_report(sample_lead))
        priorities = [f["priority"] for f in db.get_followups(report_id)]
        assert priorities == sorted(priorities)

    def test_review_marks_report_reviewed_and_moves_lead(
        self, db: Database, sample_lead: int
    ):
        report_id = db.save_report(build_report(sample_lead))
        review_id = db.save_review(
            report_id=report_id,
            reviewer="A. Counsellor",
            decision="approved",
            notes="Looks good.",
            lead_status="qualified",
        )
        assert review_id > 0
        assert db.get_report(report_id)["status"] == "reviewed"
        assert db.get_lead(sample_lead)["status"] == "qualified"

    def test_review_on_missing_report_raises(self, db: Database):
        with pytest.raises(RecordNotFoundError):
            db.save_review(report_id=999, reviewer="X", decision="approved")


class TestAudit:
    def test_actions_are_logged(self, db: Database, sample_lead: int):
        report_id = db.save_report(build_report(sample_lead))
        db.save_review(report_id=report_id, reviewer="A", decision="approved")
        actions = {row["action"] for row in db.get_audit_log()}
        assert {"created"} <= actions
        entities = {row["entity_type"] for row in db.get_audit_log()}
        assert {"lead", "qualification_report", "counsellor_review"} <= entities

    def test_filter_by_entity(self, db: Database, sample_lead: int):
        rows = db.get_audit_log(entity_type="lead")
        assert rows and all(r["entity_type"] == "lead" for r in rows)


class TestQueries:
    @pytest.fixture
    def populated(self, db: Database):
        for index, score in enumerate([90.0, 60.0, 20.0], start=1):
            lead_id = db.insert_lead(
                full_name=f"Lead {index}", email=f"lead{index}@example.com"
            )
            db.insert_message(lead_id=lead_id, body="I want an MBA in January.")
            db.save_report(build_report(lead_id, score_value=score))
        return db

    def test_readiness_distribution(self, populated: Database):
        rows = {r["band"]: r["lead_count"] for r in populated.query_readiness_distribution()}
        assert rows == {"READY": 1, "NEEDS_INFO": 1, "NOT_READY": 1}

    def test_missing_field_frequency(self, populated: Database):
        rows = populated.query_missing_field_frequency()
        assert rows
        assert all(0 <= r["missing_pct"] <= 100 for r in rows)
        assert all(r["total_count"] == 3 for r in rows)

    def test_leads_needing_info_respects_threshold(self, populated: Database):
        assert len(populated.query_leads_needing_info(threshold=75.0)) == 2
        assert len(populated.query_leads_needing_info(threshold=10.0)) == 0

    def test_provider_usage_aggregates_tokens(self, populated: Database):
        rows = populated.query_provider_usage()
        assert len(rows) == 1
        assert rows[0]["run_count"] == 3
        assert rows[0]["input_tokens"] == 300

    def test_pipeline_overview(self, populated: Database):
        overview = populated.query_pipeline_overview()
        assert overview["total_leads"] == 3
        assert overview["total_reports"] == 3
        assert overview["awaiting_review"] == 3


class TestConnectionHandling:
    def test_context_manager_closes(self, tmp_path):
        path = tmp_path / "ctx.db"
        with Database(path) as database:
            database.initialise()
            database.insert_lead(full_name="A B", email="a@example.com")
        with pytest.raises(sqlite3.ProgrammingError):
            database.conn.execute("SELECT 1")

    def test_data_survives_reopening(self, tmp_path):
        path = tmp_path / "persist.db"
        with Database(path) as first:
            first.initialise()
            first.insert_lead(full_name="A B", email="a@example.com")
        with Database(path) as second:
            assert len(second.list_leads()) == 1
