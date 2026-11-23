"""Prompt engineering for the qualification summary.

Design notes
------------
* **Stable system prompt.** Role, task, context boundaries, fairness rules,
  the field catalogue and a worked example are all static, so the prompt
  prefix is identical on every call and benefits from prompt caching.
* **Volatile content last.** Only the user turn changes per lead.
* **Context boundaries.** The model is told exactly what it may use (the
  lead's own words plus the deterministic analysis) and what it must not do
  (invent facts, judge the person, decide who gets a consultation).
* **Structured output.** The response shape is enforced by
  :data:`leadqual.llm.schema.QUALIFICATION_SCHEMA`; the prompt describes the
  same shape in words so the model's reasoning matches the constraint.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ..fairness import FAIRNESS_RULES, redact
from ..models import FIELD_SPECS, NLPResult, ScoreResult
from ..nlp.similarity import INTENT_DESCRIPTIONS
from .schema import QUALIFICATION_SCHEMA

MAX_MESSAGES_IN_PROMPT = 12
MAX_MESSAGE_CHARS = 1500


def _field_catalogue() -> str:
    """The qualification fields, rendered for the system prompt."""
    lines = []
    for spec in FIELD_SPECS:
        tag = "REQUIRED" if spec.required else "optional"
        lines.append(f"- {spec.name} ({tag}, weight {spec.weight}): {spec.description}")
    return "\n".join(lines)


#: A single worked example. One is enough to fix the register and shape
#: without crowding out the actual enquiry.
WORKED_EXAMPLE = """\
EXAMPLE

Input analysis (abbreviated):
  Readiness score: 52.0 (NEEDS_INFO)
  Present: programme_interest ("MSc in data analytics"), study_goal ("move into
           a data role"), study_mode ("online")
  Missing (required): preferred_schedule, start_timeline, budget_indication
  Enquirer's words: "Hi, I'm interested in your MSc in data analytics. I'd like
           to move into a data role eventually. Does it run online?"

Correct output:
{
  "qualification_summary": "This enquirer has named the programme (MSc in data analytics), given a clear goal of moving into a data role, and asked specifically about online delivery. What is not on file is when they want to start, how much time they can commit each week, and any indication of budget or funding route. A short qualifying exchange on those three points would make the consultation far more productive.",
  "information_present": ["Programme: MSc in data analytics", "Goal: move into a data role", "Asked whether the course runs online"],
  "information_missing": ["Preferred study schedule", "Intended start date or intake", "Budget or funding route"],
  "suggested_followups": [
    {"question": "Which intake were you hoping to join - is there a date you are working towards?", "field_name": "start_timeline", "priority": 1, "rationale": "Start date drives which cohort and deadlines apply."},
    {"question": "Roughly how many hours a week could you set aside for study, and on which days?", "field_name": "preferred_schedule", "priority": 2, "rationale": "Confirms the part-time online format is workable before recommending it."},
    {"question": "Is there a fee range or payment arrangement you would like us to work within?", "field_name": "budget_indication", "priority": 3, "rationale": "Lets the counsellor bring the right funding and instalment options to the call."},
    {"question": "Could you tell me about your most recent qualification or role?", "field_name": "current_background", "priority": 4, "rationale": "Needed to check entry requirements before the consultation."}
  ],
  "recommended_next_step": "Send a short qualifying email covering start date, weekly availability and funding, then book the consultation once at least the start date is confirmed.",
  "confidence_note": "The goal is stated loosely ('eventually'), so the timeline may be longer than it appears."
}

Note what the example does NOT do: it does not guess the enquirer's budget,
does not speculate about their circumstances, and does not suggest that any
enquiry is or is not worth the counsellor's time."""


SYSTEM_PROMPT_TEMPLATE = """\
You are a qualification assistant inside an education provider's CRM. You
support admissions counsellors who run one-to-one consultations with people
enquiring about courses.

YOUR TASK
Given one enquirer's own messages plus a deterministic analysis produced by
the CRM, write (a) a short qualification summary for the counsellor and
(b) the follow-up questions that would best prepare them for a consultation.

WHAT YOU ARE ASSESSING
You assess the COMPLETENESS OF THE INFORMATION ON FILE - nothing else. The
counsellor needs to know what the enquiry already tells them and which gaps
to close first. You are not deciding whether anyone deserves a consultation.

THE QUALIFICATION FIELDS
{field_catalogue}

{fairness_rules}

CONTEXT BOUNDARIES
- Use ONLY the enquirer's messages and the CRM analysis given in the user
  turn. Do not use outside knowledge about institutions, fees or courses.
- Never invent a detail the enquirer did not provide. If something is
  unclear, name it as a gap or mention it in confidence_note.
- Some text may contain "{redaction_token}". That marks a personal detail
  the CRM removed on purpose. Ignore it completely; never ask about it and
  never refer to it.
- The CRM's readiness score and missing-field list are computed by code and
  are authoritative. You may add nuance (for example, noting that a field was
  answered vaguely) but do not contradict them outright.
- Address the counsellor, not the enquirer. Write the follow-up questions in
  words the counsellor could say to the enquirer directly.

STYLE
- Plain professional English, no marketing language, no bullet-point padding.
- The summary is 2-4 sentences of continuous prose.
- Questions are open, single-barrelled, and polite. One question per entry.
- Order follow-ups by what unblocks the consultation soonest: priority 1 is
  the question to ask first.

OUTPUT
Return a single JSON object matching the provided schema, with keys:
qualification_summary, information_present, information_missing,
suggested_followups (each with question, field_name, priority, rationale),
recommended_next_step, and optionally confidence_note. Output nothing else.

{worked_example}"""


def build_system_prompt() -> str:
    """Build the static system prompt.

    The result is byte-identical across calls, which keeps the cached prefix
    valid and makes runs reproducible.
    """
    from ..fairness import REDACTION_TOKEN

    return SYSTEM_PROMPT_TEMPLATE.format(
        field_catalogue=_field_catalogue(),
        fairness_rules=FAIRNESS_RULES,
        redaction_token=REDACTION_TOKEN,
        worked_example=WORKED_EXAMPLE,
    )


def _format_messages(messages: Sequence[dict[str, Any]]) -> str:
    """Render the enquirer's message history, redacted and trimmed."""
    inbound = [
        m for m in messages
        if str(m.get("direction", "inbound")).lower() == "inbound"
    ]
    if not inbound:
        return "(no inbound messages on file)"

    # Keep the most recent messages when the history is long.
    selected = inbound[-MAX_MESSAGES_IN_PROMPT:]
    omitted = len(inbound) - len(selected)

    lines: list[str] = []
    if omitted:
        lines.append(f"[{omitted} earlier message(s) omitted for length]")
    for index, message in enumerate(selected, start=1):
        body = redact(str(message.get("body") or "").strip())
        if len(body) > MAX_MESSAGE_CHARS:
            body = body[:MAX_MESSAGE_CHARS].rsplit(" ", 1)[0] + " [...]"
        channel = message.get("channel", "unknown")
        sent_at = message.get("sent_at") or message.get("created_at") or "unknown date"
        lines.append(f"[{index}] ({channel}, {sent_at})\n{body}")
    return "\n\n".join(lines)


def _format_analysis(nlp: NLPResult, score: ScoreResult) -> str:
    """Render the deterministic CRM analysis for the user turn."""
    present_lines = []
    missing_lines = []
    for spec in FIELD_SPECS:
        field = nlp.fields.get(spec.name)
        tag = "required" if spec.required else "optional"
        if field and field.is_present:
            value = field.value or "(detected, no value captured)"
            present_lines.append(
                f"- {spec.name} [{tag}, confidence {field.confidence:.2f}]: {value}"
            )
        else:
            missing_lines.append(f"- {spec.name} [{tag}, weight {spec.weight}]: {spec.description}")

    intent_description = INTENT_DESCRIPTIONS.get(nlp.intent, "")
    questions = "\n".join(f"- {redact(q)}" for q in nlp.questions[:6]) or "- (none detected)"

    sections = [
        f"READINESS SCORE: {score.score} / 100  ->  {score.band} ({score.band_meaning})",
        f"DETECTED INTENT: {nlp.intent} ({nlp.intent_confidence:.2f}) - {intent_description}",
        f"TONE: {nlp.sentiment} ({nlp.sentiment_score:+.2f})",
        f"KEYWORDS: {', '.join(nlp.keywords) if nlp.keywords else '(none)'}",
        "",
        "FIELDS FOUND BY THE CRM:",
        "\n".join(present_lines) if present_lines else "- (none)",
        "",
        "FIELDS MISSING:",
        "\n".join(missing_lines) if missing_lines else "- (none - the enquiry is complete)",
        "",
        "QUESTIONS THE ENQUIRER ASKED:",
        questions,
    ]

    if nlp.protected_attribute_hits:
        sections.extend(
            [
                "",
                "NOTE: the enquirer mentioned personal details in these categories: "
                + ", ".join(nlp.protected_attribute_hits)
                + ". They have been redacted from the text above and must play no part "
                "in your assessment.",
            ]
        )

    return "\n".join(sections)


def build_user_prompt(
    lead: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    nlp: NLPResult,
    score: ScoreResult,
) -> str:
    """Build the per-lead user turn.

    Args:
        lead: The lead record.
        messages: The lead's message history.
        nlp: Deterministic extraction results.
        score: Completeness scoring results.

    Returns:
        The user-turn text. The enquirer's name is deliberately **not**
        included - it is irrelevant to completeness and keeping it out
        removes one more route to biased output.
    """
    programme = lead.get("programme_interest") or "not recorded"
    source = lead.get("source") or "unknown"

    return f"""\
ENQUIRY TO ASSESS

CRM metadata:
- Enquiry source: {source}
- Programme recorded on the CRM record: {programme}
- Messages on file: {nlp.message_count}

CRM ANALYSIS (computed by code - authoritative):
{_format_analysis(nlp, score)}

THE ENQUIRER'S OWN MESSAGES:
{_format_messages(messages)}

Produce the qualification summary and follow-up questions for the counsellor
now, as a single JSON object matching the schema."""


def prompt_debug_bundle(
    lead: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    nlp: NLPResult,
    score: ScoreResult,
) -> dict[str, Any]:
    """Return the exact prompts that would be sent - used by the UI's
    "show the prompt" panel and by tests that assert on prompt content."""
    system = build_system_prompt()
    user = build_user_prompt(lead, messages, nlp, score)
    return {
        "system": system,
        "user": user,
        "system_chars": len(system),
        "user_chars": len(user),
        "schema": json.dumps(QUALIFICATION_SCHEMA, indent=2),
    }
