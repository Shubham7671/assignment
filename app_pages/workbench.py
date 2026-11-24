"""Lead workbench — one enquiry, end to end.

This page walks the assignment's development flow in order, one numbered card
per step:

1. Read the lead's history.
2. NLP extraction and completeness scoring (deterministic, offline, nothing
   written to the database).
3. LLM summary and follow-up questions (the only step that can call an API).
4. Counsellor review (the only step that changes the lead's status).

The split matters: step 2 can be re-run freely and costs nothing, so the
counsellor can see what the model would be told *before* anyone pays for a
call.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from leadqual.exceptions import LeadQualError
from leadqual.llm.prompts import prompt_debug_bundle
from leadqual.models import FIELD_SPECS_BY_NAME
from leadqual.nlp.similarity import INTENT_DESCRIPTIONS
from leadqual.pipeline import pipeline_preview, qualify_lead, record_review, similar_past_leads
from leadqual.scoring import explain_score
from ui import components, state, theme

db = state.current_db()
settings = state.get_settings()

components.page_header(
    "Lead workbench",
    icon=":material/precision_manufacturing:",
    subtitle=(
        "Read history → NLP extraction → completeness scoring → "
        "LLM summary and follow-up questions → counsellor review"
    ),
)

leads = db.list_leads(limit=500)
if not leads:
    components.empty_state(
        "No leads on file",
        icon=":material/person_off:",
        hint="Load the synthetic sample data from Settings & diagnostics.",
    )
    st.stop()

# --------------------------------------------------------------------------
# Lead picker
# --------------------------------------------------------------------------
leads_by_id = {int(row["lead_id"]): row for row in leads}


def _lead_option(lead_id: int) -> str:
    row = leads_by_id[lead_id]
    band = row.get("latest_band")
    suffix = theme.BAND_GLYPHS.get(band, "·") if band else "·"
    return f"{suffix}  #{row['lead_id']} · {row['full_name']} · {row['message_count']} msg"


with st.container(border=True):
    picker, summary_cell = st.columns([3, 2], gap="medium", vertical_alignment="center")
    with picker:
        lead_id = st.selectbox(
            "Choose a lead",
            list(leads_by_id),
            format_func=_lead_option,
            key="workbench_lead",
            help="Leads are listed newest first; the glyph is the latest readiness band.",
        )
    lead = db.get_lead(lead_id)
    messages = db.get_messages(lead_id)
    latest = db.get_latest_report(lead_id)
    with summary_cell:
        if latest:
            components.band_badge(
                latest["readiness_band"],
                float(latest["readiness_score"]),
                help=f"Report #{latest['report_id']}, created {latest['created_at']}.",
            )
            st.caption(f"Latest report #{latest['report_id']} · {latest['created_at']}")
        else:
            st.badge("Not yet qualified", icon=":material/hourglass_empty:", color="gray")
            st.caption("No qualification report on file for this lead.")

# --------------------------------------------------------------------------
# Step 1 · Lead history
# --------------------------------------------------------------------------
st.header("1 · Lead history", icon=":material/history:")

with st.container(border=True):
    facts, counts = st.columns([3, 1], gap="medium")
    with facts:
        st.markdown(f"**{lead['full_name']}**  \n`{lead['email']}`")
        st.markdown(
            f"Reference `{lead['external_ref'] or '—'}` · source *{lead['source']}* · "
            f"status *{lead['status']}*  \n"
            f"Programme on record: *{lead['programme_interest'] or 'not recorded'}*"
        )
    with counts:
        inbound = sum(1 for message in messages if message["direction"] == "inbound")
        # A neutral delta: "inbound" is context, not a change over time, so the
        # arrow and the good/bad colouring are turned off.
        st.metric(
            "Messages",
            len(messages),
            delta=inbound,
            delta_arrow="off",
            delta_color="gray",
            delta_description="inbound",
            border=True,
        )

with st.container(border=True, height=440 if len(messages) > 3 else "content"):
    if not messages:
        components.empty_state(
            "No messages yet",
            icon=":material/chat_bubble:",
            hint="Add the enquirer's first message below to give the NLP layer something to read.",
        )
    for message in messages:
        inbound = message["direction"] == "inbound"
        with st.chat_message("user" if inbound else "assistant"):
            st.caption(
                f"{'Enquirer' if inbound else 'Us'} · {message['channel']} · "
                f"{message['sent_at']}"
            )
            st.markdown(message["body"])

with st.expander("Add another message to this lead", icon=":material/add_comment:"):
    with st.form(f"add_message_{lead_id}", border=False):
        body = st.text_area("Message body", height=110, placeholder="Paste the enquirer's reply…")
        channel_cell, direction_cell = st.columns(2)
        channel = channel_cell.selectbox(
            "Channel", ["email", "web_form", "phone", "whatsapp", "chat", "sms"]
        )
        direction = direction_cell.selectbox("Direction", ["inbound", "outbound"])
        if st.form_submit_button("Add message", icon=":material/send:", type="primary"):
            try:
                from leadqual.validation import validate_message_payload

                payload = validate_message_payload(
                    {"body": body, "channel": channel, "direction": direction},
                    lead_id=lead_id,
                )
                db.insert_message(**payload)
                st.toast("Message added.", icon=":material/check_circle:")
                st.rerun()
            except LeadQualError as exc:
                st.error(str(exc), icon=":material/error:")

# --------------------------------------------------------------------------
# Steps 2 & 3 · NLP extraction and completeness scoring
# --------------------------------------------------------------------------
st.header("2 · NLP extraction and completeness scoring", icon=":material/psychology:")
st.caption("Deterministic and offline — no API call, nothing written to the database.")

if st.button(
    "Run NLP analysis",
    icon=":material/play_arrow:",
    type="secondary",
    key=f"analyse_{lead_id}",
):
    try:
        st.session_state[f"preview_{lead_id}"] = pipeline_preview(db, lead_id)
    except LeadQualError as exc:
        st.error(str(exc), icon=":material/error:")

preview = st.session_state.get(f"preview_{lead_id}")
if preview:
    nlp, score = preview["nlp"], preview["score"]

    components.kpi_wall(
        [
            {
                "label": "Readiness",
                "value": f"{score.score:.0f}",
                "delta": round(score.score - 75, 1),
                "delta_description": "vs. the READY threshold",
                "icon": ":material/speed:",
            },
            # The remaining three are descriptive, not directional: arrows and
            # good/bad colouring are turned off so a "negative" tone does not
            # read as a worse lead.
            {
                "label": "Intent",
                "value": nlp.intent.replace("_", " "),
                "delta": f"{nlp.intent_confidence:.2f}",
                "delta_arrow": "off",
                "delta_color": "gray",
                "delta_description": "classifier confidence",
                "icon": ":material/target:",
                "help": INTENT_DESCRIPTIONS.get(nlp.intent, "—"),
            },
            {
                "label": "Tone",
                "value": nlp.sentiment,
                "delta": f"{nlp.sentiment_score:+.2f}",
                "delta_arrow": "off",
                "delta_color": "gray",
                "delta_description": "VADER compound score",
                "icon": ":material/sentiment_satisfied:",
            },
            {
                "label": "Tokens analysed",
                "value": nlp.token_count,
                "delta": nlp.message_count,
                "delta_arrow": "off",
                "delta_color": "gray",
                "delta_description": "inbound message(s)",
                "icon": ":material/match_word:",
            },
        ],
        columns=4,
    )

    with st.container(horizontal=True, gap="small"):
        components.band_badge(score.band, score.score)
        st.badge(
            INTENT_DESCRIPTIONS.get(nlp.intent, "—"),
            icon=":material/info:",
            color="gray",
        )

    if nlp.protected_attribute_hits:
        st.warning(
            "This enquiry mentions personal details in these categories: **"
            + ", ".join(nlp.protected_attribute_hits)
            + "**. They are excluded from the score and redacted before any text "
            "reaches the language model. Handle them as a human judgement, not an "
            "automated one.",
            icon=":material/shield_person:",
            title="Protected attributes detected",
        )

    with components.card(
        "Qualification fields",
        icon=":material/fact_check:",
        caption="What the NLP layer could and could not find, and what each field is worth.",
    ):
        extraction = pd.DataFrame(
            [
                {
                    "Field": row["label"],
                    "Required": bool(row["required"]),
                    "Weight": row["weight"],
                    "Status": row["status"],
                    "Confidence": round(
                        nlp.fields[row["field"]].confidence
                        if nlp.fields.get(row["field"])
                        else 0.0,
                        2,
                    ),
                    "Extracted value": (
                        nlp.fields[row["field"]].value or ""
                        if nlp.fields.get(row["field"])
                        else ""
                    ),
                }
                for row in explain_score(score)
            ]
        )
        st.dataframe(
            extraction,
            hide_index=True,
            column_config={
                "Required": st.column_config.CheckboxColumn(
                    "Required", help="A consultation cannot run without this field."
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight", help="Contribution to the readiness score.", width="small"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Extracted value": st.column_config.TextColumn(
                    "Extracted value", width="large"
                ),
            },
            alt="Qualification fields with extraction status and confidence",
        )

    detail_left, detail_right = st.columns(2, gap="medium")
    with detail_left:
        with components.card(
            "Questions the enquirer asked",
            icon=":material/help:",
            height="stretch",
        ):
            if nlp.questions:
                for question in nlp.questions:
                    st.markdown(f"- {question}")
            else:
                st.caption("None detected in the inbound messages.")
    with detail_right:
        with components.card("Keywords", icon=":material/label:", height="stretch"):
            if nlp.keywords:
                st.markdown(" ".join(f":gray-badge[{word}]" for word in nlp.keywords))
                st.caption(
                    "Protected-attribute terms are redacted from this list before "
                    "it is shown or sent anywhere."
                )
            else:
                st.caption("Not enough text to rank keywords.")

    with st.expander(
        "Similar past enquiries (TF-IDF cosine similarity)", icon=":material/compare_arrows:"
    ):
        matches = similar_past_leads(db, lead_id, top_n=3)
        if matches:
            st.dataframe(
                pd.DataFrame(matches),
                hide_index=True,
                alt="Past enquiries most similar to this one",
            )
        else:
            st.caption("No comparable enquiries on file yet.")

    if state.show_technical():
        with st.expander(
            "The exact prompt that would be sent to the model", icon=":material/code:"
        ):
            bundle = prompt_debug_bundle(preview["lead"], preview["messages"], nlp, score)
            st.caption(
                f"System prompt: {bundle['system_chars']} characters (cached) · "
                f"User turn: {bundle['user_chars']} characters"
            )
            system_tab, user_tab, schema_tab = st.tabs(
                ["System prompt", "User turn", "Response schema"]
            )
            system_tab.code(bundle["system"], language="markdown", height=280)
            user_tab.code(bundle["user"], language="markdown", height=280)
            schema_tab.code(bundle["schema"], language="json", height=280)

# --------------------------------------------------------------------------
# Step 3 · LLM summary
# --------------------------------------------------------------------------
st.header("3 · Qualification summary and follow-up questions", icon=":material/auto_awesome:")

provider = state.provider_status()
if provider["live_calls_enabled"]:
    st.caption(
        f"Will make a live API call: **{provider['provider']}** / `{provider['model']}`"
    )
else:
    st.caption(f"Offline generator will be used — {provider['reason']}")

if st.button(
    "Generate summary and save report",
    icon=":material/auto_awesome:",
    type="primary",
    key=f"qualify_{lead_id}",
):
    with st.spinner("Analysing the enquiry and generating the summary…"):
        try:
            st.session_state[f"report_{lead_id}"] = qualify_lead(db, lead_id, settings=settings)
            st.session_state.pop(f"preview_{lead_id}", None)
        except LeadQualError as exc:
            st.error(str(exc), icon=":material/error:")

report = st.session_state.get(f"report_{lead_id}")
if report:
    with st.container(border=True):
        header_left, header_right = st.columns([3, 1], vertical_alignment="center")
        with header_left:
            st.subheader("Summary for the counsellor", icon=":material/summarize:")
        with header_right:
            components.band_badge(report.readiness_band, report.readiness_score)

        meta = report.metadata
        if meta:
            if meta.is_fallback:
                st.warning(
                    f"Generated by the **offline rule-based generator**, not a "
                    f"language model. Reason: {meta.note}",
                    icon=":material/cloud_off:",
                )
            elif state.show_technical():
                st.success(
                    f"**{meta.provider} / {meta.model}** · {meta.input_tokens} in / "
                    f"{meta.output_tokens} out tokens · {meta.latency_ms} ms · "
                    f"response `{meta.response_id}`",
                    icon=":material/cloud_done:",
                )
            else:
                # Provenance is always shown — a counsellor should know whether a
                # human-sounding summary came from a model — but the token and
                # request detail stays behind the technical toggle.
                st.success(
                    f"Generated by **{meta.provider} / {meta.model}**",
                    icon=":material/cloud_done:",
                )

        st.write(report.summary)

        if report.recommended_next_step:
            st.info(report.recommended_next_step, icon=":material/arrow_forward:",
                    title="Recommended next step")

    with components.card(
        "Suggested follow-up questions",
        icon=":material/contact_support:",
        caption="Ordered by how much each answer would move the readiness score.",
    ):
        for followup in report.followups:
            label = (
                FIELD_SPECS_BY_NAME[followup.field_name].label
                if followup.field_name in FIELD_SPECS_BY_NAME
                else "general"
            )
            with st.container(border=True):
                st.markdown(f"**{followup.priority}. {followup.question}**")
                st.caption(f":material/label: Fills *{label}* — {followup.rationale}")

    if state.show_technical():
        with st.expander("Raw validated model payload (audit)", icon=":material/data_object:"):
            st.json(report.llm_payload)

    # ---- Step 4 · counsellor review --------------------------------------
    st.header("4 · Counsellor review", icon=":material/how_to_reg:")
    with st.form(f"review_{report.report_id}", border=True):
        st.caption(
            "Saving a review marks the report reviewed and, optionally, moves the "
            "lead to a new status. This is the only step that changes the lead."
        )
        reviewer_cell, decision_cell, status_cell = st.columns(3)
        reviewer = reviewer_cell.text_input("Your name", value="A. Counsellor")
        decision = decision_cell.selectbox(
            "Decision", ["approved", "edited", "rejected", "escalated"]
        )
        lead_status = status_cell.selectbox(
            "Move the lead to",
            ["(leave unchanged)", "new", "in_progress", "qualified", "needs_info", "closed"],
        )
        notes = st.text_area("Notes", height=100, placeholder="What did you change, and why?")
        if st.form_submit_button("Save review", icon=":material/save:", type="primary"):
            try:
                review_id = record_review(
                    db,
                    report.report_id,
                    reviewer=reviewer,
                    decision=decision,
                    notes=notes or None,
                    lead_status=None if lead_status.startswith("(") else lead_status,
                )
                st.success(
                    f"Review #{review_id} saved. Report #{report.report_id} is now "
                    "marked reviewed.",
                    icon=":material/check_circle:",
                )
            except LeadQualError as exc:
                st.error(str(exc), icon=":material/error:")
