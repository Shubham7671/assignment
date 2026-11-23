"""End-to-end tests for the workflow, seeding and reporting layers."""

from __future__ import annotations

import pytest

from leadqual.db import Database
from leadqual.exceptions import PipelineError, RecordNotFoundError, ValidationError
from leadqual.llm.client import QualificationLLM
from leadqual.pipeline import (
    pipeline_preview,
    qualify_lead,
    qualify_many,
    record_review,
    similar_past_leads,
)
from leadqual.reporting import (
    export_all,
    field_coverage_matrix,
    leads_needing_info,
    missing_field_report,
    overview_metrics,
    provider_usage,
    readiness_distribution,
    review_summary,
)
from leadqual.seed import load_sample_file, seed_database
from tests.conftest import MockAnthropicClient


@pytest.fixture
def llm(live_settings, mock_client) -> QualificationLLM:
    """An LLM client backed by the mock - no network, deterministic output."""
    return QualificationLLM(live_settings, client=mock_client)


class TestQualifyLead:
    def test_runs_every_step_and_persists(self, db: Database, sample_lead: int,
                                          live_settings, llm):
        report = qualify_lead(db, sample_lead, settings=live_settings, llm=llm)

        assert report.report_id is not None
        assert report.lead_id == sample_lead
        assert report.summary
        assert 0 <= report.readiness_score <= 100
        assert report.readiness_band in {"READY", "NEEDS_INFO", "NOT_READY"}
        assert report.followups
        assert report.metadata and report.metadata.provider == "anthropic"
        assert report.nlp.fields and report.score.per_field

        stored = db.get_report(report.report_id)
        assert stored["summary"] == report.summary
        assert len(stored["fields"]) == 10
        assert stored["followups"]

    def test_lead_status_follows_the_assessment(self, db: Database, sparse_lead: int,
                                                live_settings, llm):
        qualify_lead(db, sparse_lead, settings=live_settings, llm=llm)
        assert db.get_lead(sparse_lead)["status"] == "needs_info"

    def test_preview_mode_writes_nothing(self, db: Database, sample_lead: int,
                                         live_settings, llm):
        report = qualify_lead(
            db, sample_lead, settings=live_settings, llm=llm, persist=False
        )
        assert report.report_id is None
        assert report.created_at is not None
        assert db.list_reports() == []

    def test_lead_without_messages_is_rejected(self, db: Database, live_settings, llm):
        lead_id = db.insert_lead(full_name="Silent Lead", email="silent@example.com")
        with pytest.raises(PipelineError, match="no inbound messages"):
            qualify_lead(db, lead_id, settings=live_settings, llm=llm)

    def test_outbound_only_history_is_rejected(self, db: Database, live_settings, llm):
        lead_id = db.insert_lead(full_name="Outbound Only", email="out@example.com")
        db.insert_message(lead_id=lead_id, body="We called you.", direction="outbound")
        with pytest.raises(PipelineError):
            qualify_lead(db, lead_id, settings=live_settings, llm=llm)

    def test_unknown_lead_raises(self, db: Database, live_settings, llm):
        with pytest.raises(RecordNotFoundError):
            qualify_lead(db, 9999, settings=live_settings, llm=llm)

    @pytest.mark.parametrize("bad", [0, -3, "abc", None])
    def test_bad_lead_id_raises(self, db: Database, live_settings, llm, bad):
        with pytest.raises(ValidationError):
            qualify_lead(db, bad, settings=live_settings, llm=llm)

    def test_offline_settings_mark_the_report_as_fallback(self, db: Database,
                                                          sample_lead: int, settings):
        report = qualify_lead(db, sample_lead, settings=settings)
        assert report.metadata.is_fallback is True
        assert db.get_report(report.report_id)["is_fallback"] == 1


class TestFollowupMerging:
    def test_model_questions_are_kept(self, db: Database, sample_lead: int,
                                      live_settings, llm):
        report = qualify_lead(db, sample_lead, settings=live_settings, llm=llm)
        questions = [f.question for f in report.followups]
        assert "What is the best way and time to reach you for a short call?" in questions

    def test_uncovered_required_gaps_are_topped_up(self, db: Database, sparse_lead: int,
                                                   live_settings, llm):
        """The mock only ever returns a contact-preference question, so the
        required gaps it ignores must be filled from the baseline."""
        report = qualify_lead(db, sparse_lead, settings=live_settings, llm=llm)
        covered = {f.field_name for f in report.followups}
        assert report.score.missing_required
        assert covered & set(report.score.missing_required)

    def test_followups_are_capped_and_ordered(self, db: Database, sparse_lead: int,
                                              live_settings, llm):
        report = qualify_lead(db, sparse_lead, settings=live_settings, llm=llm)
        assert len(report.followups) <= 6
        priorities = [f.priority for f in report.followups]
        assert priorities == sorted(priorities)

    def test_duplicate_questions_are_dropped(self, db: Database, sample_lead: int,
                                             live_settings):
        duplicate = (
            '{"qualification_summary": "'
            + "A" * 60
            + '", "suggested_followups": ['
            '{"question": "What is your budget?", "priority": 1, "rationale": "x"},'
            '{"question": "What is your budget?", "priority": 2, "rationale": "y"}],'
            '"recommended_next_step": "Call the enquirer soon."}'
        )
        llm = QualificationLLM(
            live_settings, client=MockAnthropicClient(response_text=duplicate)
        )
        report = qualify_lead(db, sample_lead, settings=live_settings, llm=llm)
        questions = [f.question.lower() for f in report.followups]
        assert questions.count("what is your budget?") == 1


class TestQualifyMany:
    def test_processes_every_lead_with_messages(self, db: Database, sample_lead: int,
                                                sparse_lead: int, settings):
        reports, errors = qualify_many(db, settings=settings)
        assert len(reports) == 2
        assert errors == []
        assert {r.lead_id for r in reports} == {sample_lead, sparse_lead}

    def test_skips_leads_without_messages(self, db: Database, sample_lead: int, settings):
        db.insert_lead(full_name="Silent", email="silent@example.com")
        reports, errors = qualify_many(db, settings=settings)
        assert len(reports) == 1
        assert errors == []

    def test_collects_errors_without_aborting(self, db: Database, sample_lead: int,
                                              settings):
        reports, errors = qualify_many(db, [sample_lead, 9999], settings=settings)
        assert len(reports) == 1
        assert len(errors) == 1
        assert "9999" in errors[0]

    def test_can_raise_instead(self, db: Database, sample_lead: int, settings):
        with pytest.raises(RecordNotFoundError):
            qualify_many(db, [9999], settings=settings, skip_errors=False)

    def test_a_provider_outage_does_not_abort_the_batch(
        self, db: Database, sample_lead: int, sparse_lead: int, settings, monkeypatch
    ):
        """With fallback disabled an LLM failure propagates out of
        qualify_lead; qualify_many must record it and carry on, not discard
        the leads it already finished."""
        import leadqual.pipeline as pipeline_module
        from leadqual.exceptions import LLMCallError

        real = pipeline_module.qualify_lead
        seen: list[int] = []

        def flaky(db_, lead_id, **kwargs):
            seen.append(lead_id)
            if len(seen) == 1:
                raise LLMCallError("503 UNAVAILABLE", status_code=503)
            return real(db_, lead_id, **kwargs)

        monkeypatch.setattr(pipeline_module, "qualify_lead", flaky)
        reports, errors = qualify_many(
            db, [sample_lead, sparse_lead], settings=settings
        )

        assert len(seen) == 2, "the batch stopped at the first failure"
        assert len(errors) == 1 and "503" in errors[0]
        assert len(reports) == 1

    def test_outage_still_raises_when_skip_errors_is_off(
        self, db: Database, sample_lead: int, settings, monkeypatch
    ):
        import leadqual.pipeline as pipeline_module
        from leadqual.exceptions import LLMCallError

        def always_fails(db_, lead_id, **kwargs):
            raise LLMCallError("503 UNAVAILABLE", status_code=503)

        monkeypatch.setattr(pipeline_module, "qualify_lead", always_fails)
        with pytest.raises(LLMCallError):
            qualify_many(db, [sample_lead], settings=settings, skip_errors=False)


class TestReview:
    def test_saves_and_closes_the_loop(self, db: Database, sample_lead: int,
                                       live_settings, llm):
        report = qualify_lead(db, sample_lead, settings=live_settings, llm=llm)
        review_id = record_review(
            db, report.report_id, reviewer="A. Counsellor",
            decision="approved", notes="Good to go.", lead_status="qualified",
        )
        assert review_id > 0
        stored = db.get_report(report.report_id)
        assert stored["status"] == "reviewed"
        assert stored["reviews"][0]["decision"] == "approved"
        assert db.get_lead(sample_lead)["status"] == "qualified"

    def test_validates_the_submission(self, db: Database, sample_lead: int,
                                      live_settings, llm):
        report = qualify_lead(db, sample_lead, settings=live_settings, llm=llm)
        with pytest.raises(ValidationError):
            record_review(db, report.report_id, reviewer="X", decision="maybe")
        with pytest.raises(ValidationError):
            record_review(db, report.report_id, reviewer="", decision="approved")


class TestPreviewAndSimilarity:
    def test_preview_returns_the_deterministic_half(self, db: Database, sample_lead: int):
        preview = pipeline_preview(db, sample_lead)
        assert set(preview) == {
            "lead", "messages", "nlp", "score", "ranked_gaps", "baseline_followups"
        }
        assert preview["score"].score > 0
        assert db.list_reports() == []  # nothing written

    def test_similar_leads_excludes_self(self, db: Database, sample_lead: int,
                                         sparse_lead: int):
        matches = similar_past_leads(db, sample_lead)
        assert all(m["lead_id"] != sample_lead for m in matches)
        assert all("text" not in m for m in matches)


class TestSeeding:
    def test_bundled_file_parses(self):
        leads = load_sample_file()
        assert len(leads) >= 8
        assert all("full_name" in lead for lead in leads)

    def test_seeding_inserts_leads_and_messages(self, db: Database):
        result = seed_database(db)
        assert result["leads_inserted"] >= 8
        assert result["messages_inserted"] >= 10
        assert result["errors"] == []
        assert len(db.list_leads(limit=100)) == result["leads_inserted"]

    def test_seeding_twice_skips_duplicates(self, db: Database):
        first = seed_database(db)
        second = seed_database(db)
        assert second["leads_inserted"] == 0
        assert second["leads_skipped"] == first["leads_inserted"]

    def test_sample_data_is_synthetic(self):
        """Every address must use the reserved example.com domain."""
        for lead in load_sample_file():
            assert lead["email"].endswith("@example.com")

    def test_bad_records_are_reported_not_fatal(self, db: Database, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(
            '{"leads": ['
            '{"full_name": "Good Lead", "email": "good@example.com", "messages": []},'
            '{"full_name": "", "email": "not-an-email"}]}',
            encoding="utf-8",
        )
        result = seed_database(db, path=bad_file)
        assert result["leads_inserted"] == 1
        assert len(result["errors"]) == 1

    def test_missing_file_raises(self, db: Database, tmp_path):
        with pytest.raises(ValidationError):
            seed_database(db, path=tmp_path / "nope.json")

    def test_malformed_json_raises(self, db: Database, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValidationError):
            seed_database(db, path=broken)


class TestReporting:
    @pytest.fixture
    def populated(self, db: Database, settings):
        seed_database(db)
        qualify_many(db, settings=settings)
        reports = db.list_reports(limit=5)
        record_review(
            db, reports[0]["report_id"], reviewer="A. Counsellor", decision="approved"
        )
        return db

    def test_overview_metrics(self, populated: Database):
        metrics = overview_metrics(populated)
        assert metrics["total_leads"] >= 8
        assert metrics["total_reports"] >= 8
        assert 0 <= metrics["review_rate_pct"] <= 100

    def test_readiness_distribution_lists_every_band(self, populated: Database):
        frame = readiness_distribution(populated)
        assert set(frame["band"]) == {"READY", "NEEDS_INFO", "NOT_READY"}
        assert frame["lead_count"].sum() >= 8
        assert abs(frame["share_pct"].sum() - 100.0) < 1.0

    def test_missing_field_report(self, populated: Database):
        frame = missing_field_report(populated)
        assert not frame.empty
        assert frame["missing_pct"].between(0, 100).all()
        assert frame.iloc[0]["is_required"]  # required fields sort first

    def test_leads_needing_info(self, populated: Database):
        frame = leads_needing_info(populated, threshold=75.0)
        assert (frame["readiness_score"] < 75).all()

    def test_provider_usage_records_the_fallback(self, populated: Database):
        frame = provider_usage(populated)
        assert not frame.empty
        assert frame["is_fallback"].all()
        assert frame["run_count"].sum() >= 8

    def test_review_summary(self, populated: Database):
        frame = review_summary(populated)
        assert "approved" in set(frame["decision"])

    def test_field_coverage_matrix(self, populated: Database):
        matrix = field_coverage_matrix(populated)
        assert not matrix.empty
        assert matrix.isin([0, 1]).all().all()

    def test_empty_database_produces_empty_frames(self, db: Database):
        for frame in (
            readiness_distribution(db), missing_field_report(db),
            leads_needing_info(db), provider_usage(db), review_summary(db),
        ):
            assert frame is not None  # shape is defined even with no rows

    def test_export_writes_every_view(self, populated: Database, tmp_path):
        written = export_all(populated, tmp_path)
        assert len(written) == 6
        for path in written.values():
            from pathlib import Path

            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
