"""Qualification reports — everything already produced.

Browsing and auditing happen here; generating happens in the Lead workbench.
The list is filterable because the useful questions are "which reports came
out of the offline fallback?" and "what is still awaiting review?", and both
are filters rather than separate screens.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from leadqual.exceptions import LeadQualError
from leadqual.pipeline import record_review
from leadqual.reporting import report_to_frame
from ui import components, state, theme

db = state.current_db()

components.page_header(
    "Qualification reports",
    icon=":material/description:",
    subtitle="Every report produced so far, with its fields, follow-up questions and reviews.",
)

rows = db.list_reports(limit=200)
if not rows:
    components.empty_state(
        "No reports yet",
        icon=":material/description:",
        hint="Generate the first one from the Lead workbench.",
    )
    st.page_link(
        "app_pages/workbench.py",
        label="Open the lead workbench",
        icon=":material/precision_manufacturing:",
    )
    st.stop()

frame = pd.DataFrame(rows)
frame["is_fallback"] = frame["is_fallback"].astype(bool)

# --------------------------------------------------------------------------
# Filters + list
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("All reports", icon=":material/table_rows:")
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        bands = st.pills(
            "Readiness band",
            list(theme.BAND_ORDER),
            selection_mode="multi",
            format_func=theme.band_title,
            key="reports_band_filter",
        )
        review_state = st.segmented_control(
            "Review state",
            ["All", "Awaiting review", "Reviewed"],
            default="All",
            key="reports_review_filter",
        )
        fallback_only = st.toggle(
            "Offline fallback only",
            key="reports_fallback_filter",
            help="Reports the offline rule-based generator produced, not a language model.",
        )

    filtered = frame
    if bands:
        filtered = filtered[filtered["readiness_band"].isin(bands)]
    if review_state == "Awaiting review":
        filtered = filtered[filtered["status"] == "awaiting_review"]
    elif review_state == "Reviewed":
        filtered = filtered[filtered["status"] != "awaiting_review"]
    if fallback_only:
        filtered = filtered[filtered["is_fallback"]]

    st.caption(f"Showing {len(filtered)} of {len(frame)} report(s).")

    if filtered.empty:
        components.empty_state(
            "No reports match these filters",
            icon=":material/filter_alt_off:",
            hint="Clear a filter above to widen the list.",
        )
    else:
        display = filtered[
            ["report_id", "lead_id", "full_name", "readiness_score", "readiness_band",
             "status", "provider", "model", "is_fallback", "followup_count", "created_at"]
        ].copy()
        display["readiness_band"] = display["readiness_band"].map(components.band_cell)
        st.dataframe(
            display,
            hide_index=True,
            column_config={
                "report_id": st.column_config.NumberColumn("Report", format="#%d", width="small"),
                "lead_id": st.column_config.NumberColumn("Lead", format="#%d", width="small"),
                "full_name": st.column_config.TextColumn("Name", width="medium"),
                "readiness_score": components.score_column(),
                "readiness_band": components.band_column(),
                "status": st.column_config.TextColumn("Status"),
                "provider": st.column_config.TextColumn("Provider"),
                "model": st.column_config.TextColumn("Model", width="medium"),
                "is_fallback": st.column_config.CheckboxColumn(
                    "Offline fallback",
                    help="True when no language model was reachable and the "
                         "rule-based generator produced the summary.",
                ),
                "followup_count": st.column_config.NumberColumn("Follow-ups"),
                "created_at": st.column_config.DatetimeColumn(
                    "Created", format="DD MMM YYYY HH:mm"
                ),
            },
            alt="All qualification reports matching the current filters",
        )

# --------------------------------------------------------------------------
# One report in detail
# --------------------------------------------------------------------------
options = [int(value) for value in (filtered["report_id"] if not filtered.empty else frame["report_id"])]
lookup = {int(row["report_id"]): row for row in rows}

report_id = st.selectbox(
    "Open a report",
    options,
    format_func=lambda rid: (
        f"#{rid} · {lookup[rid]['full_name']} · {theme.band_title(lookup[rid]['readiness_band'])}"
    ),
    key="reports_selected",
)
report = db.get_report(int(report_id))

with st.container(border=True):
    header_left, header_right = st.columns([3, 1], vertical_alignment="center")
    with header_left:
        st.subheader(
            f"Report #{report['report_id']} — {report['full_name']}",
            icon=":material/article:",
        )
        st.caption(
            f"Created {report['created_at']} · status *{report['status']}* · "
            f"provider `{report['provider']}` / `{report['model']}`"
            + ("  ·  offline fallback" if report["is_fallback"] else "")
        )
    with header_right:
        components.band_badge(report["readiness_band"], float(report["readiness_score"]))

    st.write(report["summary"])
    if report.get("recommended_next_step"):
        st.info(
            report["recommended_next_step"],
            icon=":material/arrow_forward:",
            title="Recommended next step",
        )

fields_tab, followups_tab, reviews_tab = st.tabs(
    ["Qualification fields", "Follow-up questions", "Counsellor reviews"]
)

with fields_tab:
    st.dataframe(
        report_to_frame(report),
        hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn("Weight", width="small"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Value": st.column_config.TextColumn("Value", width="large"),
        },
        alt="Qualification fields recorded on this report",
    )

with followups_tab:
    if not report["followups"]:
        st.caption("No follow-up questions were suggested — nothing material was missing.")
    for followup in report["followups"]:
        with st.container(border=True):
            question, status_cell, action_cell = st.columns(
                [6, 2, 2], vertical_alignment="center"
            )
            question.markdown(f"**{followup['priority']}.** {followup['question']}")
            status_cell.badge(
                followup["status"],
                icon=":material/flag:",
                color="green" if followup["status"] != "suggested" else "gray",
            )
            if action_cell.button(
                "Mark asked",
                icon=":material/check:",
                key=f"ask_{followup['followup_id']}",
                width="stretch",
                disabled=followup["status"] != "suggested",
            ):
                db.update_followup_status(followup["followup_id"], "asked")
                st.rerun()

with reviews_tab:
    if report["reviews"]:
        st.dataframe(
            pd.DataFrame(report["reviews"]),
            hide_index=True,
            alt="Counsellor reviews recorded against this report",
        )
    else:
        st.caption("Not yet reviewed.")
        with st.form(f"review_existing_{report['report_id']}", border=True):
            reviewer_cell, decision_cell = st.columns(2)
            reviewer = reviewer_cell.text_input("Your name", value="A. Counsellor")
            decision = decision_cell.selectbox(
                "Decision", ["approved", "edited", "rejected", "escalated"]
            )
            notes = st.text_area("Notes", height=90)
            if st.form_submit_button("Save review", icon=":material/save:", type="primary"):
                try:
                    record_review(
                        db,
                        int(report["report_id"]),
                        reviewer=reviewer,
                        decision=decision,
                        notes=notes or None,
                    )
                    st.toast("Review saved.", icon=":material/check_circle:")
                    st.rerun()
                except LeadQualError as exc:
                    st.error(str(exc), icon=":material/error:")

if state.show_technical():
    with st.expander(
        "Stored analysis and raw model payload (audit)", icon=":material/data_object:"
    ):
        st.json(
            {
                "score": report.get("score"),
                "llm_payload": report.get("llm_payload"),
                "nlp": report.get("nlp"),
            }
        )
