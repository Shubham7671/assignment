"""Tests for completeness scoring."""

from __future__ import annotations

import pytest

from leadqual.models import FIELD_SPECS, REQUIRED_FIELDS, band_for_score
from leadqual.nlp.extract import analyse_lead
from leadqual.scoring import (
    MISSING_REQUIRED_PENALTY,
    baseline_followups,
    explain_score,
    rank_gaps,
    score_completeness,
)


def msg(body: str) -> dict:
    return {"message_id": 1, "body": body, "direction": "inbound", "channel": "email"}


RICH_ENQUIRY = (
    "I want to move into data science. I am interested in the MSc in Data "
    "Analytics. I can study weekday evenings after 7pm, about 10 hours a week. "
    "I would like to start in the January intake. My budget is around £9,000. "
    "I prefer to study online. I have completed a BSc in Statistics. I am based "
    "in Manchester. Please contact me via WhatsApp. What documents do I need?"
)


class TestWeights:
    def test_weights_sum_to_one_hundred(self):
        assert sum(spec.weight for spec in FIELD_SPECS) == 100

    def test_field_names_are_unique(self):
        names = [spec.name for spec in FIELD_SPECS]
        assert len(names) == len(set(names))

    def test_required_fields_are_declared(self):
        assert set(REQUIRED_FIELDS) <= {spec.name for spec in FIELD_SPECS}
        assert len(REQUIRED_FIELDS) == 5


class TestBands:
    @pytest.mark.parametrize(
        "score,expected",
        [(100, "READY"), (75, "READY"), (74.9, "NEEDS_INFO"), (45, "NEEDS_INFO"),
         (44.9, "NOT_READY"), (0, "NOT_READY")],
    )
    def test_band_boundaries(self, score, expected):
        assert band_for_score(score)[0] == expected


class TestScoring:
    def test_rich_enquiry_scores_ready(self):
        score = score_completeness(analyse_lead([msg(RICH_ENQUIRY)]))
        assert score.band == "READY"
        assert score.score >= 75
        assert score.missing_required == []

    def test_empty_enquiry_scores_zero(self):
        score = score_completeness(analyse_lead([msg("Hi. Thanks.")]))
        assert score.score == 0.0
        assert score.band == "NOT_READY"
        assert len(score.missing_required) == len(REQUIRED_FIELDS)

    def test_score_is_always_in_range(self):
        for text in ["Hi", RICH_ENQUIRY, "fees?", "I want an MBA in January"]:
            score = score_completeness(analyse_lead([msg(text)]))
            assert 0.0 <= score.score <= 100.0

    def test_missing_required_fields_are_penalised(self):
        """Two enquiries with equal weight earned score differently when one
        of them is short a required field."""
        partial = score_completeness(analyse_lead([msg("I want an MBA.")]))
        assert partial.missing_required
        expected_penalty = MISSING_REQUIRED_PENALTY * len(partial.missing_required)
        unpenalised = (partial.earned_weight / partial.total_weight) * 100.0
        assert partial.score == pytest.approx(
            max(0.0, unpenalised - expected_penalty), abs=0.1
        )

    def test_partition_of_missing_fields(self):
        score = score_completeness(analyse_lead([msg("I want an MBA.")]))
        overlap = set(score.missing_required) & set(score.missing_optional)
        assert not overlap

    def test_per_field_credit_never_exceeds_weight(self):
        score = score_completeness(analyse_lead([msg(RICH_ENQUIRY)]))
        for spec in FIELD_SPECS:
            assert score.per_field[spec.name] <= spec.weight


class TestGapsAndFollowups:
    def test_required_gaps_rank_above_optional(self):
        score = score_completeness(analyse_lead([msg("Hi. Thanks.")]))
        gaps = rank_gaps(score)
        required_positions = [gaps.index(n) for n in score.missing_required]
        optional_positions = [gaps.index(n) for n in score.missing_optional]
        assert max(required_positions) < min(optional_positions)

    def test_baseline_followups_target_the_biggest_gaps(self):
        score = score_completeness(analyse_lead([msg("Hi. Thanks.")]))
        followups = baseline_followups(score, limit=3)
        assert len(followups) == 3
        assert [f.priority for f in followups] == [1, 2, 3]
        assert all(f.field_name in score.missing_required for f in followups)
        assert all(f.question.endswith("?") for f in followups)

    def test_complete_enquiry_yields_no_baseline_questions(self):
        score = score_completeness(analyse_lead([msg(RICH_ENQUIRY)]))
        if not score.missing_required and not score.missing_optional:
            assert baseline_followups(score) == []

    def test_explain_score_covers_every_field(self):
        score = score_completeness(analyse_lead([msg(RICH_ENQUIRY)]))
        rows = explain_score(score)
        assert len(rows) == len(FIELD_SPECS)
        assert {r["status"] for r in rows} <= {"present", "partial", "missing"}
        # required fields are listed first
        assert rows[0]["required"] is True
