"""Rule-based entity extraction into the qualification field catalogue.

The extractor reads a lead's inbound messages sentence by sentence and
decides, for each field in :data:`leadqual.models.FIELD_SPECS`, whether the
enquirer has supplied it - and if so, with what evidence.

Each rule is a compiled regular expression paired with a confidence weight.
A field's final confidence rises when several independent rules agree, which
keeps a single incidental keyword from being treated as a firm answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..fairness import find_protected_mentions, redact
from ..models import FIELD_SPECS, ExtractedField, NLPResult
from .preprocess import (
    extract_questions,
    normalise_text,
    sentence_split,
    sentiment,
    tokenize,
    top_keywords,
)

MAX_VALUE_LEN = 220
_FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class ExtractionRule:
    """One pattern that signals the presence of a qualification field.

    Attributes:
        field_name: Which field this rule supports.
        pattern: Compiled regex applied to a single sentence.
        confidence: Confidence contributed by a match (0-1).
        capture: Optional named group holding the value to store. When
            omitted the whole matching sentence is stored as evidence.
    """

    field_name: str
    pattern: re.Pattern[str]
    confidence: float
    capture: str | None = None


def _rule(field_name: str, pattern: str, confidence: float, capture: str | None = None) -> ExtractionRule:
    return ExtractionRule(field_name, re.compile(pattern, _FLAGS), confidence, capture)


# --------------------------------------------------------------------------
# Shared fragments
# --------------------------------------------------------------------------
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DAYS = r"mon(?:day)?s?|tue(?:s|sday)?s?|wed(?:nesday)?s?|thu(?:r|rs|rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?"
_CURRENCY = r"(?:[₹$£€]|\b(?:inr|usd|gbp|eur|aud|cad|rs\.?|aed)\b)"
_QUALIFICATIONS = (
    r"mba|m\.?sc|b\.?sc|b\.?tech|m\.?tech|b\.?a\b|m\.?a\b|ph\.?d|bba|bca|mca|"
    r"diploma|certificate|certification|bootcamp|degree|masters?|bachelors?|"
    r"foundation|short course|pg\s*dip"
)


# --------------------------------------------------------------------------
# The rule base
# --------------------------------------------------------------------------
RULES: tuple[ExtractionRule, ...] = (
    # ---------------------------------------------------------- study_goal --
    _rule("study_goal", r"\b(?:my\s+)?(?:goal|aim|objective|ambition)\s+is\s+(?P<v>.{5,160})", 0.90, "v"),
    _rule("study_goal", r"\bi\s+(?:want|would like|hope|plan|intend)\s+to\s+(?P<v>.{5,160})", 0.85, "v"),
    _rule("study_goal", r"\bso\s+(?:that\s+)?i\s+can\s+(?P<v>.{5,160})", 0.75, "v"),
    _rule("study_goal", r"\b(?:looking|hoping)\s+to\s+(?P<v>(?:move|switch|transition|change|get|become|work|start|build|grow|progress|advance).{3,150})", 0.85, "v"),
    _rule("study_goal", r"\b(?:career\s+(?:change|switch|progression|growth)|change\s+careers?|switch\s+careers?)\b", 0.80),
    _rule("study_goal", r"\b(?:up\s*-?\s*skill|re\s*-?\s*skill|upskilling|reskilling)\b", 0.75),
    _rule("study_goal", r"\b(?:get|secure|qualify for)\s+(?:a\s+)?(?:promotion|pay\s*rise|raise|better\s+(?:job|role))\b", 0.80),
    _rule("study_goal", r"\bbecome\s+a\s+(?P<v>[a-z\s]{3,60})", 0.80, "v"),
    _rule("study_goal", r"\b(?:move|migrate|relocate)\s+(?:to|abroad|overseas)\b", 0.70),

    # -------------------------------------------------- programme_interest --
    _rule("programme_interest", rf"\b(?:interested\s+in|enquir(?:ing|y)\s+about|asking\s+about|looking\s+(?:at|for)|applying\s+for|apply\s+for)\s+(?:the\s+|your\s+|a\s+|an\s+)?(?P<v>[^.?!]{{3,120}})", 0.85, "v"),
    _rule("programme_interest", rf"\b(?P<v>(?:{_QUALIFICATIONS})\s+(?:in|of)\s+[a-z\s&]{{3,60}})", 0.90, "v"),
    _rule("programme_interest", rf"\b(?:{_QUALIFICATIONS})\b", 0.65),
    _rule("programme_interest", r"\b(?:data\s+(?:science|analytics)|machine\s+learning|artificial\s+intelligence|cyber\s*security|software\s+(?:engineering|development)|digital\s+marketing|project\s+management|business\s+analytics|human\s+resources?|supply\s+chain|graphic\s+design|ux\s*/?\s*ui|cloud\s+computing|accounting|finance|nursing|psychology)\b", 0.80),
    _rule("programme_interest", r"\b(?:course|programme|program)\s+(?:on|in|about)\s+(?P<v>[a-z\s&]{3,60})", 0.85, "v"),

    # -------------------------------------------------- preferred_schedule --
    _rule("preferred_schedule", rf"\b(?:{_DAYS})\b(?:\s*(?:and|or|to|,|-|/)\s*(?:{_DAYS})\b)*", 0.80),
    _rule("preferred_schedule", r"\b(?:week\s*ends?|week\s*days?|weeknights?)\b", 0.80),
    _rule("preferred_schedule", r"\b(?:evenings?|mornings?|afternoons?|nights?)\b", 0.75),
    _rule("preferred_schedule", r"\b(?:after|before|from|around|past)\s+(?P<v>\d{1,2}\s*(?::\d{2})?\s*(?:am|pm|hrs|o'?clock))", 0.85, "v"),
    _rule("preferred_schedule", r"\b(?P<v>\d{1,2}\s*(?:-|to|until|till)\s*\d{1,2}\s*(?:am|pm))", 0.85, "v"),
    _rule("preferred_schedule", r"\b(?P<v>\d{1,2}(?:\.\d)?\s*(?:\+\s*)?hours?\s+(?:a|per|each)\s+(?:week|day))", 0.90, "v"),
    _rule("preferred_schedule", r"\b(?:flexible|free)\s+(?:timing|timings|hours|schedule|slots?)\b", 0.65),
    _rule("preferred_schedule", r"\b(?:available|free)\s+(?:on|during|in the)\s+(?P<v>[a-z\s]{3,40})", 0.70, "v"),

    # ------------------------------------------------------ start_timeline --
    _rule("start_timeline", rf"\b(?:start(?:ing)?|begin(?:ning)?|join(?:ing)?|commence|enrol(?:l|ling)?)\s+(?:in|from|by|on|around)?\s*(?P<v>(?:{_MONTHS})\b[^.?!]{{0,20}})", 0.90, "v"),
    _rule("start_timeline", rf"\b(?P<v>(?:{_MONTHS})\s+(?:intake|cohort|batch|semester|term|session|20\d\d))", 0.90, "v"),
    _rule("start_timeline", r"\b(?:next|upcoming|this|the)\s+(?P<v>(?:intake|cohort|batch|semester|term|academic\s+year))", 0.80, "v"),
    _rule("start_timeline", r"\b(?:asap|as\s+soon\s+as\s+possible|immediately|right\s+away|straight\s+away)\b", 0.80),
    _rule("start_timeline", r"\b(?:within|in)\s+(?:the\s+)?(?:next\s+)?(?P<v>\d{1,2}\s*(?:weeks?|months?))", 0.85, "v"),
    _rule("start_timeline", r"\b(?:next|this)\s+(?P<v>(?:month|year|spring|summer|autumn|fall|winter))\b", 0.75, "v"),
    _rule("start_timeline", r"\b(?:spring|summer|autumn|fall|winter)\s+(?:20\d\d|intake|term)\b", 0.85),

    # --------------------------------------------------- budget_indication --
    _rule("budget_indication", rf"(?P<v>{_CURRENCY}\s*\d[\d,.]*\s*(?:k|lakh|lac|lakhs|crore|m|million)?)", 0.90, "v"),
    _rule("budget_indication", r"\b(?P<v>\d[\d,.]*\s*(?:k|lakh|lac|lakhs|crore)\b)", 0.80, "v"),
    _rule("budget_indication", r"\b(?:budget|price\s+range|fee\s+range)\s*(?:is|of|around|about|up\s+to|under|below)?\s*(?P<v>[^.?!]{2,60})", 0.85, "v"),
    _rule("budget_indication", r"\b(?:afford|affordable|affordability|within\s+my\s+means|too\s+expensive|cost\s+effective|cheap(?:er)?)\b", 0.70),
    _rule("budget_indication", r"\b(?:emi|instal?ment\s+plan|instal?ments|payment\s+plan|pay\s+monthly|pay\s+in\s+parts)\b", 0.85),
    _rule("budget_indication", r"\b(?:scholarship|bursary|grant|funding|sponsor(?:ed|ship)?|student\s+loan|employer\s+(?:paid|funded|sponsor))\b", 0.80),
    _rule("budget_indication", r"\b(?:fees?|tuition|cost)\s+(?:is|are|of|around|about|approximately)\s+(?P<v>[^.?!]{2,50})", 0.85, "v"),
    _rule("budget_indication", r"\b(?:self[\s-]?fund(?:ed|ing)?|out\s+of\s+pocket)\b", 0.75),

    # ---------------------------------------------------------- study_mode --
    _rule("study_mode", r"\b(?P<v>online|remote(?:ly)?|virtual|distance\s+learning|e-?learning)\b", 0.85, "v"),
    _rule("study_mode", r"\b(?P<v>on[\s-]?campus|in[\s-]?person|face[\s-]?to[\s-]?face|classroom[\s-]?based)\b", 0.85, "v"),
    _rule("study_mode", r"\b(?P<v>hybrid|blended)\b", 0.85, "v"),
    _rule("study_mode", r"\b(?P<v>part[\s-]?time|full[\s-]?time|self[\s-]?paced)\b", 0.80, "v"),

    # --------------------------------------------------- current_background --
    _rule("current_background", r"\bi\s+(?:have\s+)?(?:completed|finished|hold|did|studied|graduated\s+(?:with|in|from))\s+(?P<v>[^.?!]{3,120})", 0.90, "v"),
    _rule("current_background", r"\bi\s+(?:am|'m)\s+(?:currently\s+)?(?:working|employed)\s+(?:as|at|in|with)\s+(?P<v>[^.?!]{3,100})", 0.90, "v"),
    _rule("current_background", r"\b(?:i\s+work\s+as|my\s+(?:current\s+)?(?:role|job|position)\s+is)\s+(?P<v>[^.?!]{3,100})", 0.90, "v"),
    _rule("current_background", r"\b(?P<v>\d{1,2}\+?\s*years?\s+(?:of\s+)?experience)", 0.85, "v"),
    _rule("current_background", r"\bmy\s+background\s+is\s+(?:in\s+)?(?P<v>[^.?!]{3,100})", 0.90, "v"),
    _rule("current_background", rf"\b(?:i\s+have\s+an?|i\s+hold\s+an?)\s+(?P<v>(?:{_QUALIFICATIONS})[^.?!]{{0,60}})", 0.90, "v"),
    _rule("current_background", r"\b(?:fresher|no\s+prior\s+experience|career\s+break|recent\s+graduate|final\s+year)\b", 0.75),

    # ------------------------------------------------------------ location --
    _rule("location", r"\b(?:based|living|located|residing)\s+in\s+(?P<v>[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})", 0.90, "v"),
    _rule("location", r"\bi\s+(?:live|stay)\s+in\s+(?P<v>[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})", 0.90, "v"),
    _rule("location", r"\b(?:i'?m|i\s+am)\s+from\s+(?P<v>[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})", 0.85, "v"),
    _rule("location", r"\b(?:relocat(?:e|ing)|moving)\s+to\s+(?P<v>[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})", 0.80, "v"),
    _rule("location", r"\b(?P<v>(?:GMT|UTC|IST|EST|PST|CST|BST|CET|AEST|SGT)(?:\s*[+-]\s*\d{1,2})?)\b", 0.85, "v"),
    _rule("location", r"\b(?:my\s+)?time\s*zone\s+is\s+(?P<v>[^.?!]{2,40})", 0.90, "v"),

    # -------------------------------------------------- contact_preference --
    _rule("contact_preference", r"\b(?:call|phone|ring)\s+me\s+(?P<v>[^.?!]{2,60})", 0.90, "v"),
    _rule("contact_preference", r"\b(?:please\s+)?(?:contact|reach|text|message)\s+me\s+(?:on|via|by|at|through)\s+(?P<v>[^.?!]{2,60})", 0.90, "v"),
    _rule("contact_preference", r"\b(?:prefer|prefers?|preferred)\s+(?:to\s+be\s+)?(?:contacted\s+)?(?:by|via|on)\s+(?P<v>email|whats\s*app|phone|call|text|sms|chat)", 0.90, "v"),
    _rule("contact_preference", r"\b(?:whats\s*app|wa)\s+(?:me|is\s+(?:best|fine|easiest|preferred))\b", 0.85),
    _rule("contact_preference", r"\b(?:best\s+(?:time|way)\s+to\s+(?:call|reach|contact)\s+me)\s+(?:is\s+)?(?P<v>[^.?!]{2,60})", 0.90, "v"),
    _rule("contact_preference", r"\b(?:do\s*n[o']t|please\s+do\s+not|don't)\s+call\b", 0.85),
    _rule("contact_preference", r"\b(?:email|e-mail)\s+(?:me\s+)?(?:is\s+)?(?:best|fine|preferred|easier)\b", 0.85),
)

#: Grouped for quick iteration.
RULES_BY_FIELD: dict[str, list[ExtractionRule]] = {}
for _r in RULES:
    RULES_BY_FIELD.setdefault(_r.field_name, []).append(_r)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _clean_value(raw: str) -> str:
    """Tidy a captured value: collapse space, trim trailing conjunctions."""
    value = normalise_text(raw)
    value = re.sub(r"[\s,;:.\-]+$", "", value)
    value = re.sub(r"\s+(?:and|but|so|because|as|which|that)$", "", value, flags=_FLAGS)
    if len(value) > MAX_VALUE_LEN:
        value = value[:MAX_VALUE_LEN].rsplit(" ", 1)[0] + "..."
    return value.strip()


def _iter_sentences(
    messages: Sequence[dict[str, Any]], *, inbound_only: bool = True
) -> list[tuple[int | None, str]]:
    """Flatten messages into ``(message_id, sentence)`` pairs."""
    pairs: list[tuple[int | None, str]] = []
    for message in messages:
        if inbound_only and str(message.get("direction", "inbound")).lower() != "inbound":
            continue
        body = message.get("body") or ""
        message_id = message.get("message_id")
        for sentence in sentence_split(body):
            pairs.append((message_id, sentence))
    return pairs


def _combined_text(messages: Iterable[dict[str, Any]], *, inbound_only: bool = True) -> str:
    """Join message bodies into one string."""
    parts = [
        str(m.get("body") or "")
        for m in messages
        if not inbound_only or str(m.get("direction", "inbound")).lower() == "inbound"
    ]
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Field extraction
# --------------------------------------------------------------------------
def extract_fields(messages: Sequence[dict[str, Any]]) -> dict[str, ExtractedField]:
    """Resolve every qualification field from a lead's message history.

    Args:
        messages: Message dicts as returned by
            :meth:`leadqual.db.Database.get_messages`. Only inbound messages
            are considered - the enquirer's own words are the evidence.

    Returns:
        A dict keyed by field name. Every field in the catalogue is present;
        fields with no evidence come back with ``is_present=False``.
    """
    sentences = _iter_sentences(messages)
    results: dict[str, ExtractedField] = {}

    for spec in FIELD_SPECS:
        if spec.name == "open_questions":
            continue  # handled separately, from the question extractor

        best_confidence = 0.0
        best_value: str | None = None
        best_message_id: int | None = None
        best_sentence: str | None = None
        distinct_rules = 0

        for rule in RULES_BY_FIELD.get(spec.name, []):
            rule_matched = False
            for message_id, sentence in sentences:
                match = rule.pattern.search(sentence)
                if not match:
                    continue
                rule_matched = True
                if rule.capture:
                    captured = match.groupdict().get(rule.capture) or match.group(0)
                else:
                    captured = match.group(0)
                value = _clean_value(captured)
                if not value:
                    continue
                if rule.confidence > best_confidence:
                    best_confidence = rule.confidence
                    best_value = value
                    best_message_id = message_id
                    best_sentence = sentence
            if rule_matched:
                distinct_rules += 1

        if best_value is None:
            results[spec.name] = ExtractedField(
                field_name=spec.name, is_present=False, confidence=0.0, method="rule"
            )
            continue

        # Corroboration bonus: independent rules agreeing raises confidence.
        confidence = min(0.98, best_confidence + 0.04 * max(0, distinct_rules - 1))
        results[spec.name] = ExtractedField(
            field_name=spec.name,
            is_present=True,
            value=redact(best_value),
            confidence=round(confidence, 3),
            evidence_message_id=best_message_id,
            evidence_text=redact(best_sentence) if best_sentence else None,
            method="rule",
        )

    return results


# --------------------------------------------------------------------------
# Full analysis
# --------------------------------------------------------------------------
def analyse_lead(
    messages: Sequence[dict[str, Any]], *, classifier: Any | None = None
) -> NLPResult:
    """Run the full NLP pass over a lead's history.

    Steps: sentence segmentation -> rule-based field extraction -> question
    extraction -> TF-IDF intent classification -> sentiment -> keywords.

    Args:
        messages: Message dicts for one lead.
        classifier: An optional pre-fitted
            :class:`leadqual.nlp.similarity.IntentClassifier`. One is built
            lazily from the bundled training examples when omitted.

    Returns:
        A populated :class:`~leadqual.models.NLPResult`.
    """
    inbound = [
        m for m in messages
        if str(m.get("direction", "inbound")).lower() == "inbound"
    ]
    text = _combined_text(messages)

    fields = extract_fields(messages)
    questions = extract_questions(text)

    # open_questions is derived rather than pattern-matched.
    fields["open_questions"] = ExtractedField(
        field_name="open_questions",
        is_present=bool(questions),
        value="; ".join(redact(q) for q in questions[:5]) if questions else None,
        confidence=0.95 if questions else 0.0,
        evidence_message_id=inbound[0].get("message_id") if inbound and questions else None,
        evidence_text=redact(questions[0]) if questions else None,
        method="rule",
    )

    # Intent classification (TF-IDF + cosine similarity).
    if classifier is None:
        from .similarity import get_default_classifier

        classifier = get_default_classifier()
    try:
        intent, intent_confidence = classifier.predict(text)
    except Exception:  # pragma: no cover - never let NLP extras break a run
        intent, intent_confidence = "unknown", 0.0

    sentiment_label, sentiment_score = sentiment(text)

    return NLPResult(
        fields=fields,
        questions=questions,
        intent=intent,
        intent_confidence=round(float(intent_confidence), 3),
        sentiment=sentiment_label,
        sentiment_score=sentiment_score,
        # Keywords are taken from the redacted text: they are surfaced in the
        # UI and placed in the LLM prompt, so protected-attribute words must
        # not survive here either.
        keywords=top_keywords(redact(text), limit=12),
        token_count=len(tokenize(text)),
        message_count=len(inbound),
        protected_attribute_hits=find_protected_mentions(text),
    )
