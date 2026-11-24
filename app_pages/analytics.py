"""Analytics and audit — how the numbers were produced.

This page deliberately ignores the "Show technical details" toggle: it is the
place where the full picture is always on screen, so a marker or a developer
has one page to go to rather than a toggle to find.

Everything here is a SQL aggregate from :mod:`leadqual.db`, shaped with pandas
in :mod:`leadqual.reporting`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from leadqual.reporting import field_coverage_matrix, provider_usage, review_summary
from ui import charts, components, state

db = state.current_db()

components.page_header(
    "Analytics and audit",
    icon=":material/query_stats:",
    subtitle=(
        "SQL aggregates over the SQLite database, shaped with pandas. Every "
        "report can be traced back to a real API call or an offline fallback."
    ),
)

usage = provider_usage(db)
reviews = review_summary(db)

# --------------------------------------------------------------------------
# Spend and throughput
# --------------------------------------------------------------------------
if usage.empty:
    st.info(
        "Nothing has been generated yet, so there is no provider usage to audit.",
        icon=":material/info:",
        title="No runs recorded",
    )
else:
    total_runs = int(usage["run_count"].sum())
    live_runs = int(usage.loc[~usage["is_fallback"], "run_count"].sum())
    total_tokens = int(usage["total_tokens"].sum())
    latency = usage["avg_latency_ms"].dropna()
    components.kpi_wall(
        [
            {
                "label": "Generation runs",
                "value": total_runs,
                "icon": ":material/play_circle:",
                "help": "Every summary generated, live or offline.",
            },
            {
                "label": "Live API calls",
                "value": live_runs,
                "delta": f"{(100 * live_runs / total_runs):.0f}%",
                "delta_arrow": "off",
                "delta_color": "gray",
                "delta_description": "of all runs",
                "icon": ":material/cloud_done:",
            },
            {
                "label": "Tokens used",
                "value": f"{total_tokens:,}",
                "icon": ":material/toll:",
                "help": "Input plus output tokens across every live call.",
            },
            {
                "label": "Average latency",
                "value": f"{latency.mean():.0f} ms" if not latency.empty else "—",
                "icon": ":material/timer:",
                "help": "Mean round-trip time of the provider calls on record.",
            },
        ],
        columns=4,
    )

usage_card, decisions_card = st.columns(2, gap="medium")

with usage_card:
    with components.card(
        "Provider and model usage",
        icon=":material/memory:",
        caption=(
            "Every report records its provider, model, token usage and latency, "
            "so a demonstration is always traceable."
        ),
        height="stretch",
    ):
        if usage.empty:
            components.empty_state("No runs recorded yet", icon=":material/memory:")
        else:
            st.dataframe(
                usage,
                hide_index=True,
                column_config={
                    "is_fallback": st.column_config.CheckboxColumn("Offline fallback"),
                    "run_count": st.column_config.NumberColumn("Runs"),
                    "input_tokens": st.column_config.NumberColumn("In"),
                    "output_tokens": st.column_config.NumberColumn("Out"),
                    "total_tokens": st.column_config.NumberColumn("Total tokens"),
                    "avg_latency_ms": st.column_config.NumberColumn(
                        "Latency", format="%d ms"
                    ),
                    "share_of_runs_pct": st.column_config.ProgressColumn(
                        "Share of runs", min_value=0, max_value=100, format="%.0f%%"
                    ),
                },
                alt="Token usage and latency per provider and model",
            )

with decisions_card:
    with components.card(
        "Counsellor decisions",
        icon=":material/how_to_reg:",
        caption="What humans did with the assistant's output.",
        height="stretch",
    ):
        if reviews.empty:
            components.empty_state("No reviews recorded yet", icon=":material/how_to_reg:")
        else:
            st.dataframe(
                reviews,
                hide_index=True,
                column_config={
                    "decision": st.column_config.TextColumn("Decision"),
                    "review_count": st.column_config.NumberColumn("Reviews"),
                    "avg_score_reviewed": st.column_config.NumberColumn(
                        "Average score reviewed", format="%.0f"
                    ),
                    "distinct_reviewers": st.column_config.NumberColumn("Reviewers"),
                },
                alt="Counsellor decisions and the average score they reviewed",
            )

# --------------------------------------------------------------------------
# Field coverage across the pipeline
# --------------------------------------------------------------------------
with components.card(
    "Field coverage across leads",
    icon=":material/grid_on:",
    caption=(
        "One row per lead, one column per qualification field. A pale column is "
        "a field the enquiry process never captures; a pale row is a lead the "
        "counsellor has almost nothing on."
    ),
):
    coverage = field_coverage_matrix(db, limit=30)
    if coverage.empty:
        components.empty_state(
            "No assessed leads yet",
            icon=":material/grid_on:",
            hint="Coverage appears once reports exist.",
        )
    else:
        long_form = (
            coverage.reset_index()
            .melt(id_vars="lead", var_name="field", value_name="present")
        )
        st.altair_chart(
            charts.coverage_heatmap(long_form),
            alt="Grid of which qualification fields are on file for each lead",
        )

# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------
with components.card(
    "Audit log",
    icon=":material/receipt_long:",
    caption="Every write the application made, in the order it made them.",
):
    entity = st.segmented_control(
        "Entity type",
        ["All", "lead", "qualification_report", "counsellor_review", "database"],
        default="All",
        key="analytics_audit_entity",
    )
    audit = db.get_audit_log(
        entity_type=None if entity in (None, "All") else entity, limit=200
    )
    if audit:
        st.dataframe(
            pd.DataFrame(audit),
            hide_index=True,
            column_config={
                "created_at": st.column_config.DatetimeColumn(
                    "When", format="DD MMM YYYY HH:mm:ss"
                ),
                "details": st.column_config.TextColumn("Details", width="large"),
            },
            alt="Audit log entries, newest first",
        )
    else:
        components.empty_state(
            "Nothing logged for this entity type",
            icon=":material/receipt_long:",
        )
