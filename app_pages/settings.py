"""Settings and diagnostics.

Three different audiences land here, so the page is three tabs rather than one
long scroll: **Data** for whoever needs sample leads, **Environment** for
whoever is wiring up an API key, and **Fairness** for whoever is marking or
reviewing how the assistant is constrained.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from leadqual.exceptions import LeadQualError
from leadqual.llm.client import QualificationLLM
from leadqual.nlp.preprocess import nlp_backend_status
from leadqual.nlp.similarity import similarity_backend_status
from leadqual.seed import seed_database
from ui import components, state

db = state.current_db()
settings = state.get_settings()

components.page_header(
    "Settings and diagnostics",
    icon=":material/tune:",
    subtitle="Sample data, resolved configuration, backend status and the fairness controls.",
)

data_tab, environment_tab, fairness_tab = st.tabs(["Data", "Environment", "Fairness"])

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
with data_tab:
    with components.card(
        "Sample data",
        icon=":material/dataset:",
        caption=(
            "The bundled dataset is entirely synthetic: invented names and "
            "`example.com` addresses only."
        ),
    ):
        with st.container(horizontal=True, gap="small"):
            load = st.button(
                "Load sample leads", icon=":material/download:", type="primary"
            )
            reset = st.button(
                "Reset database and reload",
                icon=":material/restart_alt:",
                type="secondary",
                help="Drops every lead, report and review, then re-seeds the sample data.",
            )

        if load:
            try:
                result = seed_database(db)
                st.success(
                    f"Inserted {result['leads_inserted']} lead(s) and "
                    f"{result['messages_inserted']} message(s); skipped "
                    f"{result['leads_skipped']} already present.",
                    icon=":material/check_circle:",
                )
                for error in result["errors"]:
                    st.warning(error, icon=":material/warning:")
            except LeadQualError as exc:
                st.error(str(exc), icon=":material/error:")

        if reset:
            try:
                db.reset()
                result = seed_database(db)
                st.success(
                    f"Database reset. Inserted {result['leads_inserted']} lead(s) "
                    f"and {result['messages_inserted']} message(s).",
                    icon=":material/check_circle:",
                )
                st.cache_resource.clear()
            except LeadQualError as exc:
                st.error(str(exc), icon=":material/error:")

    with components.card(
        "Database", icon=":material/database:", caption=f"File: `{db.path}`"
    ):
        overview = db.query_pipeline_overview()
        components.kpi_wall(
            [
                {"label": "Leads", "value": overview.get("total_leads", 0),
                 "icon": ":material/group:"},
                {"label": "Messages", "value": overview.get("total_messages", 0),
                 "icon": ":material/forum:"},
                {"label": "Reports", "value": overview.get("total_reports", 0),
                 "icon": ":material/description:"},
                {"label": "Reviews", "value": overview.get("total_reviews", 0),
                 "icon": ":material/how_to_reg:"},
                {"label": "Open follow-ups", "value": overview.get("open_followups", 0),
                 "icon": ":material/contact_support:"},
            ],
            columns=5,
        )
        with st.expander("Raw overview query result", icon=":material/data_object:"):
            st.json(overview)

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
with environment_tab:
    status = QualificationLLM(settings).status()

    with components.card("LLM provider", icon=":material/smart_toy:"):
        if status["live_calls_enabled"]:
            st.success(status["reason"], icon=":material/cloud_done:")
        else:
            st.warning(status["reason"], icon=":material/cloud_off:")

        st.markdown("**All supported providers**")
        st.dataframe(
            pd.DataFrame(QualificationLLM.all_provider_status(settings)),
            hide_index=True,
            alt="Configuration status of every supported LLM provider",
        )
        st.caption(
            "To enable live calls: copy `.env.example` to `.env` and set "
            "`ANTHROPIC_API_KEY` (paid) or `GEMINI_API_KEY` (free — get one at "
            "aistudio.google.com/apikey), then restart the app. With no "
            "`LEADQUAL_PROVIDER` set, whichever key is present is used."
        )
        with st.expander("Resolved provider status", icon=":material/data_object:"):
            st.json(status)

    config_card, backends_card = st.columns(2, gap="medium")
    with config_card:
        with components.card(
            "Configuration",
            icon=":material/settings:",
            caption="Resolved from environment variables and Streamlit secrets. Keys are redacted.",
            height="stretch",
        ):
            st.json(settings.describe())
    with backends_card:
        with components.card(
            "NLP backends",
            icon=":material/spellcheck:",
            caption="Which optional corpora and libraries are actually available here.",
            height="stretch",
        ):
            st.json({**nlp_backend_status(), **similarity_backend_status()})

# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
with fairness_tab:
    with components.card("Fairness controls", icon=":material/balance:"):
        st.markdown(
            """
The assistant assesses **information completeness only**. Three controls
enforce that in code rather than leaving it to the prompt:

1. **Protected attributes are detected** in the enquirer's own words
   (age, gender, marital or family status, religion, ethnicity or
   nationality, disability or health).
2. **They are redacted** from every piece of text before it reaches the
   language model, and from the keyword list shown in the workbench.
3. **They never affect the score.** The readiness score is a weighted
   completeness measure over the qualification fields; no protected
   attribute appears in that calculation.

The system prompt additionally forbids recommending that any enquirer be
deprioritised, declined or filtered out, and forbids treating a smaller
budget or a non-traditional background as a negative signal.
            """
        )
        st.info(
            "A readiness band describes the *enquiry*, not the person. A "
            "NOT_READY lead is one we have not asked enough questions of yet.",
            icon=":material/info:",
        )
