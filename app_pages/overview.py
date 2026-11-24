"""Pipeline overview — the dashboard.

Four questions, in the order a head of admissions would ask them:

1. How big is the pipeline and how ready is it? (KPI wall)
2. How is readiness distributed, and what do our enquiries fail to capture?
3. Where do scores actually cluster, and is the workload changing?
4. Who needs a counsellor's attention right now? (work queue)

Every figure measures what is *on file*, never the enquirer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from leadqual.reporting import (
    leads_needing_info,
    missing_field_report,
    overview_metrics,
    readiness_distribution,
)
from ui import charts, components, state

db = state.current_db()

components.page_header(
    "Pipeline overview",
    icon=":material/space_dashboard:",
    subtitle=(
        "Information completeness across the enquiry pipeline. Every figure "
        "measures what is on file, never the enquirer."
    ),
)

metrics = overview_metrics(db)
reports = pd.DataFrame(db.list_reports(limit=500))

if not metrics.get("total_reports"):
    st.info(
        "Load the sample data from **Settings & diagnostics**, then run a lead "
        "through the **Lead workbench** to populate this dashboard.",
        icon=":material/rocket_launch:",
        title="No qualification reports yet",
    )

# --------------------------------------------------------------------------
# 1 · KPI wall
# --------------------------------------------------------------------------
# The sparkline is the readiness score of the last few reports in the order
# they were produced — a trend, not a ranking, so oldest first.
score_trend: list[float] = []
if not reports.empty:
    score_trend = (
        reports.sort_values("report_id").tail(12)["readiness_score"].astype(float).tolist()
    )

average_score = float(metrics.get("avg_score") or 0.0)

components.kpi_wall(
    [
        {
            "label": "Leads",
            "value": metrics.get("total_leads", 0),
            "icon": ":material/group:",
            "help": "Enquirers on file, whether or not they have been assessed.",
        },
        {
            "label": "Messages",
            "value": metrics.get("total_messages", 0),
            "icon": ":material/forum:",
            "help": "Inbound and outbound messages across every lead.",
        },
        {
            "label": "Reports",
            "value": metrics.get("total_reports", 0),
            "icon": ":material/description:",
            "help": "Qualification reports produced so far.",
        },
        {
            "label": "Average readiness",
            "value": f"{average_score:.0f}",
            "delta": round(average_score - 75, 1),
            "delta_description": "vs. the READY threshold of 75",
            "icon": ":material/speed:",
            "chart_data": score_trend or None,
            "chart_type": "line",
            "help": "Mean readiness score across every report, out of 100.",
        },
        {
            "label": "Awaiting review",
            "value": metrics.get("awaiting_review", 0),
            "icon": ":material/pending_actions:",
            "help": "Reports no counsellor has signed off yet.",
        },
    ],
    columns=5,
)

# --------------------------------------------------------------------------
# 2 · Readiness distribution and the gaps behind it
# --------------------------------------------------------------------------
distribution = readiness_distribution(db)
missing = missing_field_report(db)

left, right = st.columns(2, gap="medium")

with left:
    with components.card(
        "Readiness of the pipeline",
        icon=":material/donut_small:",
        caption="How many leads sit in each readiness band.",
        height="stretch",
    ):
        view = st.segmented_control(
            "View",
            ["Chart", "Table"],
            default="Chart",
            key="overview_readiness_view",
            label_visibility="collapsed",
        )
        if not distribution["lead_count"].sum():
            components.empty_state(
                "No scored leads yet",
                icon=":material/query_stats:",
                hint="Qualify a lead in the workbench to see the distribution.",
            )
        elif view == "Table":
            st.dataframe(
                distribution.rename(
                    columns={
                        "band": "Band",
                        "lead_count": "Leads",
                        "avg_score": "Average",
                        "min_score": "Lowest",
                        "max_score": "Highest",
                        "share_pct": "Share %",
                    }
                ),
                hide_index=True,
                alt="Lead counts and score range per readiness band",
            )
        else:
            st.altair_chart(
                charts.readiness_chart(distribution),
                alt="Lead counts per readiness band",
            )

with right:
    with components.card(
        "What our enquiries fail to capture",
        icon=":material/rule:",
        caption="Share of assessed enquiries with no value for each field.",
        height="stretch",
    ):
        view = st.segmented_control(
            "View",
            ["Chart", "Table"],
            default="Chart",
            key="overview_missing_view",
            label_visibility="collapsed",
        )
        if missing.empty:
            components.empty_state(
                "No scored leads yet",
                icon=":material/rule:",
                hint="Field coverage appears once at least one lead is assessed.",
            )
        elif view == "Table":
            st.dataframe(
                missing.rename(
                    columns={
                        "field_label": "Field",
                        "is_required": "Required",
                        "weight": "Weight",
                        "missing_count": "Missing",
                        "total_count": "Assessed",
                        "missing_pct": "Missing %",
                        "score_impact": "Score impact",
                    }
                ).drop(columns=["field_name"]),
                hide_index=True,
                alt="How often each qualification field is missing",
            )
        else:
            st.altair_chart(
                charts.missing_fields_chart(missing),
                alt="Share of enquiries missing each qualification field",
            )
            worst = missing.iloc[0]
            st.caption(
                f"**{worst['field_label']}** is missing from "
                f"{worst['missing_pct']:.0f}% of assessed enquiries — the clearest "
                "candidate for a change to the enquiry form or first-reply template. "
                "Solid bars are fields a consultation cannot run without."
            )

# --------------------------------------------------------------------------
# 3 · Score shape and workload over time
# --------------------------------------------------------------------------
if not reports.empty:
    shape_left, shape_right = st.columns(2, gap="medium")

    with shape_left:
        with components.card(
            "Where scores cluster",
            icon=":material/bar_chart:",
            caption="Readiness scores in ten-point bins, coloured by band.",
            height="stretch",
        ):
            st.altair_chart(
                charts.score_distribution_chart(reports),
                alt="Distribution of readiness scores in ten-point bins",
            )
            near_ready = reports[
                (reports["readiness_score"] >= 60) & (reports["readiness_score"] < 75)
            ]
            if not near_ready.empty:
                st.caption(
                    f"{len(near_ready)} report(s) score 60–74: close enough that one "
                    "answered follow-up question would usually move them to READY."
                )

    with shape_right:
        with components.card(
            "Assessment activity",
            icon=":material/timeline:",
            caption="Reports produced per day, with the day's average score on hover.",
            height="stretch",
        ):
            activity = (
                reports.assign(day=pd.to_datetime(reports["created_at"]).dt.normalize())
                .groupby("day", as_index=False)
                .agg(reports=("report_id", "count"), avg_score=("readiness_score", "mean"))
                .sort_values("day")
            )
            if len(activity) < 2:
                components.empty_state(
                    "Not enough history yet",
                    icon=":material/timeline:",
                    hint="A trend needs reports from at least two different days.",
                )
            else:
                st.altair_chart(
                    charts.activity_chart(activity),
                    alt="Qualification reports produced per day",
                )

# --------------------------------------------------------------------------
# 4 · Counsellor work queue
# --------------------------------------------------------------------------
with components.card(
    "Counsellor work queue",
    icon=":material/assignment_late:",
    caption="Leads whose latest assessment falls below the READY threshold.",
):
    queue = leads_needing_info(db)
    if queue.empty:
        st.success(
            "Every assessed lead is READY — nothing is waiting on more information.",
            icon=":material/check_circle:",
        )
    else:
        display = queue[
            [
                "lead_id", "full_name", "readiness_score", "readiness_band",
                "open_followups", "missing_required",
            ]
        ].copy()
        display["readiness_band"] = display["readiness_band"].map(components.band_cell)
        st.dataframe(
            display,
            hide_index=True,
            column_config={
                "lead_id": st.column_config.NumberColumn("Lead", format="#%d", width="small"),
                "full_name": st.column_config.TextColumn("Name", width="medium"),
                "readiness_score": components.score_column(),
                "readiness_band": components.band_column(),
                "open_followups": st.column_config.NumberColumn(
                    "Open questions",
                    help="Suggested follow-up questions nobody has asked yet.",
                ),
                "missing_required": st.column_config.TextColumn(
                    "Missing required fields", width="large"
                ),
            },
            alt="Leads below the READY threshold, worst first",
        )
        st.page_link(
            "app_pages/workbench.py",
            label="Open the lead workbench",
            icon=":material/precision_manufacturing:",
        )
