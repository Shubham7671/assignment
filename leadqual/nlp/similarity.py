"""TF-IDF intent classification and nearest-neighbour lead search.

Two scikit-learn uses, both genuinely useful to a counsellor:

* :class:`IntentClassifier` labels what an enquiry is mainly *about*
  (course information, fees, admission requirements, scheduling, ...) using
  TF-IDF vectors and a nearest-centroid cosine decision.
* :func:`find_similar_leads` retrieves past enquiries whose wording resembles
  the current one, so a counsellor can reuse what worked before.

Both degrade gracefully: if scikit-learn is missing, the classifier falls
back to keyword overlap rather than failing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from .preprocess import content_tokens, normalise_text

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


#: Labelled training phrases. Synthetic, written for this project.
#: Each label needs several examples so the centroid is not dominated by one.
INTENT_TRAINING_DATA: tuple[tuple[str, str], ...] = (
    # ---- course_information
    ("course_information", "Can you tell me more about the data science programme and what it covers?"),
    ("course_information", "What modules are included in the MBA and how long does it take?"),
    ("course_information", "I would like details of the syllabus and the subjects taught."),
    ("course_information", "Which specialisations are available on the masters degree?"),
    ("course_information", "How many credits is the diploma and is it accredited?"),
    ("course_information", "Please send me the course brochure and curriculum outline."),
    # ---- fees_and_funding
    ("fees_and_funding", "How much are the tuition fees for the full programme?"),
    ("fees_and_funding", "Do you offer scholarships or any discount on the course fee?"),
    ("fees_and_funding", "Is there an EMI or instalment payment plan available?"),
    ("fees_and_funding", "My budget is around 2 lakh, what can you offer within that?"),
    ("fees_and_funding", "Can my employer sponsor the course and pay the invoice directly?"),
    ("fees_and_funding", "Is a student loan accepted and what is the total cost?"),
    # ---- admission_requirements
    ("admission_requirements", "What are the entry requirements and do I qualify with my diploma?"),
    ("admission_requirements", "Do I need work experience to be eligible for the masters?"),
    ("admission_requirements", "Is IELTS required for admission and what score do you accept?"),
    ("admission_requirements", "I have a bachelors degree, am I eligible to apply?"),
    ("admission_requirements", "What documents do I need to submit with my application?"),
    ("admission_requirements", "Will you accept my qualification from another country?"),
    # ---- scheduling_and_mode
    ("scheduling_and_mode", "Are classes held in the evening or at weekends?"),
    ("scheduling_and_mode", "Can I study online part time while I keep working?"),
    ("scheduling_and_mode", "How many hours a week do I need to commit to the course?"),
    ("scheduling_and_mode", "Is the programme on campus or is there a hybrid option?"),
    ("scheduling_and_mode", "I can only attend after 6pm on weekdays, is that possible?"),
    ("scheduling_and_mode", "Are the live sessions recorded if I cannot join the timing?"),
    # ---- enrolment_and_dates
    ("enrolment_and_dates", "When does the next intake start and how do I enrol?"),
    ("enrolment_and_dates", "What is the application deadline for the January cohort?"),
    ("enrolment_and_dates", "I would like to register for the course starting next month."),
    ("enrolment_and_dates", "Please book me onto the September batch, I am ready to join."),
    ("enrolment_and_dates", "How long does the admission process take from application?"),
    ("enrolment_and_dates", "Is there still a seat available in the upcoming semester?"),
    # ---- career_outcomes
    ("career_outcomes", "What jobs can I get after completing this qualification?"),
    ("career_outcomes", "Do you provide placement support and interview preparation?"),
    ("career_outcomes", "I want to change career into analytics, will this course help?"),
    ("career_outcomes", "What salary can graduates of this programme expect?"),
    ("career_outcomes", "Does the certificate help me get a promotion at my current job?"),
    ("career_outcomes", "Are there internships or industry projects included?"),
    # ---- general_enquiry
    ("general_enquiry", "Hi, I saw your advertisement and wanted to know more."),
    ("general_enquiry", "Hello, please can someone get in touch with me."),
    ("general_enquiry", "I filled in the form on your website, following up."),
    ("general_enquiry", "Good morning, I am interested in studying with you."),
    ("general_enquiry", "Can you call me back when you have a moment?"),
    ("general_enquiry", "Just enquiring about your institution, thanks."),
)

INTENT_LABELS: tuple[str, ...] = tuple(dict.fromkeys(label for label, _ in INTENT_TRAINING_DATA))

#: Human-readable descriptions, shown in the UI and used in the LLM prompt.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "course_information": "Wants detail about course content, structure or accreditation",
    "fees_and_funding": "Focused on cost, payment options, scholarships or sponsorship",
    "admission_requirements": "Checking eligibility, entry criteria or documents",
    "scheduling_and_mode": "Concerned with timings, workload and online/campus delivery",
    "enrolment_and_dates": "Ready to act - asking about intakes, deadlines or registering",
    "career_outcomes": "Interested in jobs, progression and what the course leads to",
    "general_enquiry": "Opening contact without a specific question yet",
    "unknown": "Not enough text to classify reliably",
}


@dataclass
class IntentPrediction:
    """A label with its score and the runner-up, for transparency in the UI."""

    label: str
    confidence: float
    runner_up: str | None = None
    runner_up_confidence: float = 0.0


class IntentClassifier:
    """Nearest-centroid intent classifier over TF-IDF vectors.

    The classifier is fitted on :data:`INTENT_TRAINING_DATA`. Each label's
    examples are averaged into a centroid vector; prediction is the cosine
    nearest centroid. This keeps the model small, deterministic and
    explainable - appropriate for a decision-support tool a counsellor must
    be able to question.
    """

    #: Below this cosine score the prediction is reported as ``unknown``.
    MIN_CONFIDENCE = 0.12

    def __init__(self, training_data: Sequence[tuple[str, str]] | None = None) -> None:
        self.training_data = tuple(training_data or INTENT_TRAINING_DATA)
        self.labels: list[str] = list(dict.fromkeys(l for l, _ in self.training_data))
        self._vectorizer: Any | None = None
        self._centroids: Any | None = None
        self._fitted = False
        self._keyword_index: dict[str, set[str]] = {}
        self.fit()

    # ------------------------------------------------------------------
    def fit(self) -> "IntentClassifier":
        """Fit the vectoriser and compute one centroid per label."""
        texts = [text for _, text in self.training_data]
        label_list = [label for label, _ in self.training_data]

        # Keyword index powers the no-sklearn fallback.
        self._keyword_index = {}
        for label, text in self.training_data:
            self._keyword_index.setdefault(label, set()).update(content_tokens(text))

        if not _SKLEARN_AVAILABLE:
            self._fitted = False
            return self

        try:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            matrix = self._vectorizer.fit_transform(texts)
            centroids = []
            for label in self.labels:
                rows = [i for i, l in enumerate(label_list) if l == label]
                centroids.append(np.asarray(matrix[rows].mean(axis=0)).ravel())
            self._centroids = np.vstack(centroids)
            self._fitted = True
        except Exception:  # pragma: no cover - defensive
            self._fitted = False
        return self

    # ------------------------------------------------------------------
    def predict_detailed(self, text: str) -> IntentPrediction:
        """Classify ``text`` and return the label plus the runner-up."""
        cleaned = normalise_text(text)
        if not cleaned or len(cleaned.split()) < 3:
            return IntentPrediction("unknown", 0.0)

        if self._fitted and self._vectorizer is not None and self._centroids is not None:
            try:
                vector = self._vectorizer.transform([cleaned])
                scores = cosine_similarity(vector, self._centroids).ravel()
                order = np.argsort(scores)[::-1]
                best_index = int(order[0])
                best_score = float(scores[best_index])
                if best_score < self.MIN_CONFIDENCE:
                    return IntentPrediction("unknown", round(best_score, 3))
                runner_index = int(order[1]) if len(order) > 1 else None
                return IntentPrediction(
                    label=self.labels[best_index],
                    confidence=round(best_score, 3),
                    runner_up=self.labels[runner_index] if runner_index is not None else None,
                    runner_up_confidence=(
                        round(float(scores[runner_index]), 3) if runner_index is not None else 0.0
                    ),
                )
            except Exception:  # pragma: no cover
                pass

        return self._predict_by_keywords(cleaned)

    def predict(self, text: str) -> tuple[str, float]:
        """Classify ``text``; returns ``(label, confidence)``."""
        prediction = self.predict_detailed(text)
        return prediction.label, prediction.confidence

    # ------------------------------------------------------------------
    def _predict_by_keywords(self, text: str) -> IntentPrediction:
        """Jaccard-style fallback used when scikit-learn is unavailable."""
        tokens = set(content_tokens(text))
        if not tokens:
            return IntentPrediction("unknown", 0.0)
        scored: list[tuple[str, float]] = []
        for label, vocabulary in self._keyword_index.items():
            overlap = len(tokens & vocabulary)
            if overlap:
                scored.append((label, overlap / len(tokens | vocabulary)))
        if not scored:
            return IntentPrediction("unknown", 0.0)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_label, best_score = scored[0]
        if best_score < 0.02:
            return IntentPrediction("unknown", round(best_score, 3))
        runner = scored[1] if len(scored) > 1 else None
        return IntentPrediction(
            label=best_label,
            confidence=round(best_score, 3),
            runner_up=runner[0] if runner else None,
            runner_up_confidence=round(runner[1], 3) if runner else 0.0,
        )

    @property
    def backend(self) -> str:
        """Which implementation is actually in use."""
        return "tfidf-centroid" if self._fitted else "keyword-overlap"


@lru_cache(maxsize=1)
def get_default_classifier() -> IntentClassifier:
    """A process-wide classifier fitted on the bundled training data."""
    return IntentClassifier()


# --------------------------------------------------------------------------
# Similar-lead retrieval
# --------------------------------------------------------------------------
def find_similar_leads(
    query_text: str,
    candidates: Sequence[dict[str, Any]],
    *,
    top_n: int = 3,
    min_score: float = 0.05,
) -> list[dict[str, Any]]:
    """Rank past enquiries by TF-IDF cosine similarity to ``query_text``.

    Args:
        query_text: The current lead's combined message text.
        candidates: Dicts with at least ``lead_id`` and ``text`` keys.
        top_n: How many neighbours to return.
        min_score: Discard matches scoring below this.

    Returns:
        The best matches, each a copy of the candidate dict with a
        ``similarity`` key added, highest first.
    """
    cleaned_query = normalise_text(query_text)
    usable = [c for c in candidates if normalise_text(str(c.get("text") or ""))]
    if not cleaned_query or not usable:
        return []

    corpus = [cleaned_query] + [str(c["text"]) for c in usable]

    if _SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
            )
            matrix = vectorizer.fit_transform(corpus)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        except Exception:  # pragma: no cover
            scores = None
    else:
        scores = None

    if scores is None:  # keyword-overlap fallback
        query_tokens = set(content_tokens(cleaned_query))
        scores = []
        for candidate in usable:
            tokens = set(content_tokens(str(candidate["text"])))
            union = query_tokens | tokens
            scores.append(len(query_tokens & tokens) / len(union) if union else 0.0)

    ranked = [
        {**candidate, "similarity": round(float(score), 3)}
        for candidate, score in zip(usable, scores)
        if float(score) >= min_score
    ]
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[:top_n]


def similarity_backend_status() -> dict[str, Any]:
    """Report which similarity backend is active - shown in the UI footer."""
    classifier = get_default_classifier()
    return {
        "sklearn_installed": _SKLEARN_AVAILABLE,
        "intent_backend": classifier.backend,
        "intent_labels": list(classifier.labels),
        "training_examples": len(classifier.training_data),
    }
