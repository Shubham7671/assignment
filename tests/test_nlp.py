"""Tests for the NLP layer: preprocessing, extraction, intent, fairness."""

from __future__ import annotations

import pytest

from leadqual.fairness import REDACTION_TOKEN, find_protected_mentions, redact
from leadqual.nlp.extract import analyse_lead, extract_fields
from leadqual.nlp.preprocess import (
    content_tokens,
    extract_questions,
    normalise_text,
    sentence_split,
    sentiment,
    tokenize,
    top_keywords,
)
from leadqual.nlp.similarity import IntentClassifier, find_similar_leads


def msg(body: str, message_id: int = 1, direction: str = "inbound") -> dict:
    """Build a message dict of the shape the extractor expects."""
    return {
        "message_id": message_id,
        "body": body,
        "direction": direction,
        "channel": "email",
        "sent_at": "2026-10-01T10:00:00+00:00",
    }


class TestPreprocess:
    def test_normalise_collapses_whitespace_and_smart_quotes(self):
        assert normalise_text("  I’m   keen\n\nto  start ") == "I'm keen to start"

    def test_normalise_handles_empty_input(self):
        assert normalise_text("") == ""
        assert normalise_text(None) == ""

    def test_normalise_can_strip_contacts(self):
        cleaned = normalise_text(
            "Email me at a@b.com or see https://example.com", strip_contacts=True
        )
        assert "a@b.com" not in cleaned
        assert "https://example.com" not in cleaned

    def test_sentence_split(self):
        sentences = sentence_split("I want an MBA. When does it start? Thanks.")
        assert len(sentences) == 3
        assert sentences[1] == "When does it start?"

    def test_tokenize_keeps_hyphenated_domain_terms(self):
        tokens = tokenize("I want part-time online study")
        assert "part-time" in tokens

    def test_content_tokens_drop_stopwords(self):
        tokens = content_tokens("I would like to study the data analytics course")
        assert "the" not in tokens
        assert "to" not in tokens
        assert "data" in tokens and "analytics" in tokens

    def test_top_keywords_ranks_by_frequency(self):
        keywords = top_keywords("analytics analytics analytics nursing", limit=2)
        assert keywords[0] == "analytics"

    def test_sentiment_detects_polarity(self):
        positive, positive_score = sentiment("I am excited and really keen to start!")
        negative, negative_score = sentiment(
            "I am worried, confused and disappointed about the cost."
        )
        assert positive == "positive" and positive_score > 0
        assert negative == "negative" and negative_score < 0

    def test_sentiment_of_empty_text_is_neutral(self):
        assert sentiment("") == ("neutral", 0.0)


class TestQuestions:
    def test_finds_punctuated_and_unpunctuated_questions(self):
        questions = extract_questions(
            "Hello. What are the fees? How long is the course"
        )
        assert any("fees" in q for q in questions)
        assert any("How long" in q for q in questions)

    def test_deduplicates(self):
        assert len(extract_questions("What are the fees? What are the fees?")) == 1

    def test_returns_empty_when_no_questions(self):
        assert extract_questions("I would like to enrol next month.") == []


class TestFieldExtraction:
    @pytest.fixture
    def rich_fields(self):
        return extract_fields(
            [
                msg(
                    "I want to move into data science. I am interested in the MSc "
                    "in Data Analytics. I can study weekday evenings after 7pm, "
                    "about 10 hours a week. I would like to start in the January "
                    "intake. My budget is around £9,000 and I need an instalment "
                    "plan. I prefer to study online. I have completed a BSc in "
                    "Statistics. I am based in Manchester. Please contact me via "
                    "WhatsApp. What documents do I need?"
                )
            ]
        )

    @pytest.mark.parametrize(
        "field",
        [
            "study_goal", "programme_interest", "preferred_schedule",
            "start_timeline", "budget_indication", "study_mode",
            "current_background", "location", "contact_preference",
        ],
    )
    def test_detects_each_field_in_a_rich_enquiry(self, rich_fields, field):
        assert rich_fields[field].is_present, f"{field} should have been found"
        assert rich_fields[field].confidence > 0.5

    def test_records_evidence(self, rich_fields):
        goal = rich_fields["study_goal"]
        assert goal.evidence_message_id == 1
        assert goal.evidence_text

    def test_sparse_enquiry_finds_nothing(self):
        fields = extract_fields([msg("Hi, please send info. Thanks.")])
        assert all(not f.is_present for f in fields.values())

    def test_outbound_messages_are_ignored(self):
        fields = extract_fields(
            [msg("Our fees are £9,000 and we start in January.", direction="outbound")]
        )
        assert not fields["budget_indication"].is_present
        assert not fields["start_timeline"].is_present

    def test_empty_history_is_safe(self):
        fields = extract_fields([])
        assert all(not f.is_present for f in fields.values())


class TestAnalyseLead:
    def test_populates_the_whole_result(self):
        result = analyse_lead(
            [msg("I want an MBA. How much are the fees? I can study at weekends.")]
        )
        assert result.message_count == 1
        assert result.token_count > 0
        assert result.questions
        assert result.keywords
        assert result.intent in {
            "course_information", "fees_and_funding", "admission_requirements",
            "scheduling_and_mode", "enrolment_and_dates", "career_outcomes",
            "general_enquiry", "unknown",
        }

    def test_open_questions_field_tracks_questions(self):
        with_question = analyse_lead([msg("I want an MBA. What are the fees?")])
        without = analyse_lead([msg("I want an MBA starting in January.")])
        assert with_question.fields["open_questions"].is_present
        assert not without.fields["open_questions"].is_present

    def test_present_and_missing_partition_all_fields(self):
        result = analyse_lead([msg("I want an MBA.")])
        assert set(result.present_fields()) | set(result.missing_fields()) == set(
            result.fields
        )
        assert not set(result.present_fields()) & set(result.missing_fields())


class TestFairness:
    @pytest.mark.parametrize(
        "text,category",
        [
            ("I am a 42 year old applicant", "age"),
            ("I am a married woman", "gender"),
            ("I am a single mother of two", "marital_family"),
            ("I observe Ramadan", "religion"),
            ("I have dyslexia", "disability_health"),
        ],
    )
    def test_detects_protected_mentions(self, text, category):
        assert category in find_protected_mentions(text)

    def test_ordinary_text_trips_nothing(self):
        assert find_protected_mentions("I want to study data analytics online") == []

    def test_redaction_removes_the_span(self):
        redacted = redact("I am a 42 year old married woman from Valencia")
        assert "42 year old" not in redacted
        assert "married" not in redacted
        assert REDACTION_TOKEN in redacted
        assert "Valencia" in redacted  # location is not protected

    def test_extracted_values_are_redacted(self):
        fields = extract_fields(
            [msg("I am a 38 year old and I want to become a nurse.")]
        )
        goal = fields["study_goal"]
        assert goal.is_present
        assert "38" not in (goal.value or "")

    def test_keywords_are_redacted(self):
        result = analyse_lead(
            [msg("I am a 42 year old married woman who wants to study nursing.")]
        )
        assert "married" not in result.keywords
        assert "woman" not in result.keywords
        assert result.protected_attribute_hits


class TestIntentClassifier:
    @pytest.fixture(scope="class")
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("How much are the tuition fees and is there a scholarship?", "fees_and_funding"),
            ("What are the entry requirements for the masters degree?", "admission_requirements"),
            ("Are the classes in the evening or at the weekend?", "scheduling_and_mode"),
            ("When does the next intake start and how do I enrol?", "enrolment_and_dates"),
        ],
    )
    def test_classifies_clear_examples(self, classifier, text, expected):
        label, confidence = classifier.predict(text)
        assert label == expected
        assert confidence > 0

    def test_short_text_is_unknown(self, classifier):
        assert classifier.predict("hi")[0] == "unknown"

    def test_detailed_prediction_exposes_runner_up(self, classifier):
        prediction = classifier.predict_detailed(
            "What are the fees and when is the next intake?"
        )
        assert prediction.runner_up is not None
        assert prediction.confidence >= prediction.runner_up_confidence

    def test_backend_is_reported(self, classifier):
        assert classifier.backend in {"tfidf-centroid", "keyword-overlap"}


class TestSimilarity:
    def test_ranks_the_closer_document_first(self):
        candidates = [
            {"lead_id": 1, "text": "I want to study data analytics online part time"},
            {"lead_id": 2, "text": "I am asking about nursing placements on campus"},
        ]
        matches = find_similar_leads(
            "Looking for an online part-time data analytics course", candidates, top_n=2
        )
        assert matches
        assert matches[0]["lead_id"] == 1
        assert matches[0]["similarity"] >= matches[-1]["similarity"]

    def test_empty_inputs_return_nothing(self):
        assert find_similar_leads("", [{"lead_id": 1, "text": "x"}]) == []
        assert find_similar_leads("anything", []) == []
