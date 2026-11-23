"""Non-discrimination guardrails.

The brief requires a qualification summary that does **not** make
discriminatory decisions. This module is how that requirement is enforced in
code rather than left to the prompt alone:

1. :func:`find_protected_mentions` detects protected-attribute language in a
   lead's own words.
2. :func:`redact` removes those spans before any text is sent to the LLM.
3. The scoring layer never awards or deducts points for them, and
   :data:`FAIRNESS_RULES` is injected into the system prompt.

Detecting a mention is not a judgement about the person; it only marks text
that must be excluded from an automated assessment and left to a human.
"""

from __future__ import annotations

import re

#: Protected / sensitive characteristics, grouped for auditability.
#: Patterns are deliberately narrow to limit false positives on ordinary words.
PROTECTED_PATTERNS: dict[str, tuple[str, ...]] = {
    "age": (
        r"\b\d{1,2}\s*(?:years?\s*old|yrs?\s*old|y/?o)\b",
        r"\bi\s+am\s+\d{1,2}\b",
        r"\bage[d]?\s*[:\-]?\s*\d{1,2}\b",
        r"\b(?:elderly|pensioner|retired|teenager|minor)\b",
    ),
    "gender": (
        r"\b(?:male|female|man|woman|transgender|non[\s\-]?binary)\b",
        r"\b(?:he/him|she/her|they/them)\b",
    ),
    "marital_family": (
        r"\b(?:married|unmarried|divorced|widow(?:ed|er)?|single\s+(?:mother|father|parent))\b",
        r"\b(?:pregnan(?:t|cy)|maternity|paternity)\b",
        r"\bi\s+have\s+\d+\s+(?:kids|children)\b",
    ),
    "religion": (
        r"\b(?:muslim|christian|hindu|sikh|jewish|buddhist|atheist|catholic)\b",
        r"\b(?:ramadan|sabbath|church|mosque|temple|synagogue)\b",
    ),
    "ethnicity_nationality": (
        r"\b(?:caste|race|ethnicity|ethnic\s+background)\b",
        r"\b(?:asylum\s+seeker|refugee|immigrant\s+status)\b",
    ),
    "disability_health": (
        r"\b(?:disabled|disability|wheelchair|dyslexi[ac]|autis(?:m|tic)|adhd)\b",
        r"\b(?:chronic\s+illness|mental\s+health|depression|anxiety\s+disorder)\b",
    ),
}

#: Compiled once at import time.
_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    category: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for category, patterns in PROTECTED_PATTERNS.items()
}

REDACTION_TOKEN = "[redacted-personal-detail]"

#: Injected verbatim into the LLM system prompt.
FAIRNESS_RULES = """\
FAIRNESS CONSTRAINTS (non-negotiable):
- Assess ONLY whether the enquiry contains enough information for a useful
  consultation. Never assess the person's suitability, ability, or likelihood
  to enrol or pay.
- Never infer or comment on age, gender, marital or family status, religion,
  ethnicity, nationality, immigration status, disability or health.
- Never recommend deprioritising, declining or filtering out an enquirer.
  Every enquirer is entitled to a consultation; your output only affects what
  the counsellor asks first.
- Do not treat a smaller stated budget, a non-traditional background or an
  unfamiliar qualification as a negative signal. Missing information is a gap
  to ask about, not a mark against anyone.
- If the text mentions a protected characteristic, ignore it entirely rather
  than repeating it in the summary."""


def find_protected_mentions(text: str) -> list[str]:
    """Return the protected-attribute categories mentioned in ``text``.

    Args:
        text: Raw lead text.

    Returns:
        Sorted list of category names (e.g. ``["age", "religion"]``).
        Empty when nothing is detected.
    """
    if not text:
        return []
    hits = {
        category
        for category, patterns in _COMPILED.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    return sorted(hits)


def redact(text: str) -> str:
    """Replace protected-attribute spans with :data:`REDACTION_TOKEN`.

    Applied to every message before it is placed in an LLM prompt, so the
    model is not given the opportunity to reason about these details.

    Args:
        text: Raw lead text.

    Returns:
        The text with sensitive spans replaced.
    """
    if not text:
        return text
    redacted = text
    for patterns in _COMPILED.values():
        for pattern in patterns:
            redacted = pattern.sub(REDACTION_TOKEN, redacted)
    return redacted


def redaction_summary(text: str) -> dict[str, object]:
    """Describe what redaction would do to ``text`` (used by the UI/audit log)."""
    categories = find_protected_mentions(text)
    redacted = redact(text)
    return {
        "categories": categories,
        "redacted_text": redacted,
        "redaction_count": redacted.count(REDACTION_TOKEN),
        "changed": redacted != text,
    }
