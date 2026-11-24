"""Streamlit counsellor UI for the Lead Qualification & Readiness Assistant.

Run with::

    streamlit run app.py

This module is the entry point only: it bootstraps shared resources, draws the
sidebar chrome that every page shares, and hands control to ``st.navigation``.
The pages themselves live in ``app_pages/`` and the presentation helpers they
share live in ``ui/``.

Page order follows the assignment's development flow rather than an
alphabetical menu: **Overview** (where does the pipeline stand) → **Lead
workbench** (read history → NLP extraction → completeness scoring → LLM
summary and follow-up questions → counsellor review) → **Reports** (what was
decided) → **Analytics & audit** / **Settings** (how it was produced).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from leadqual import __project__, __version__
from leadqual.exceptions import LeadQualError
from ui import components, state, theme

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="Lead Qualification Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "about": (
            f"**{__project__}** v{__version__}\n\n"
            "Assesses how complete an enquiry is — never the enquirer. "
            "All bundled data is synthetic."
        )
    },
)

# Must run before get_settings(): on Streamlit Community Cloud the API key
# arrives as a secret, not as a .env file.
state.bridge_secrets()
state.ensure_nltk_data()

settings = state.get_settings()

# One-time session defaults, declared in a single place.
if state.DB_PATH_KEY not in st.session_state:
    st.session_state[state.DB_PATH_KEY] = str(settings.db_path)

# The logo is a flat image, so it cannot follow the theme the way a widget
# does — pick the variant that matches the active appearance.
st.logo(
    str(ASSETS / ("logo-dark.svg" if theme.appearance() == "dark" else "logo-light.svg")),
    size="large",
    link=None,
)

# Fail loudly and early if the database cannot be opened: every page needs it.
try:
    state.current_db()
except LeadQualError as exc:
    st.error(f"Could not open the database: {exc}", icon=":material/database:")
    st.caption("Check the path under **Data source** in the sidebar.")
    with st.sidebar.expander("Data source", icon=":material/database:", expanded=True):
        st.text_input("Database file", key=state.DB_PATH_KEY)
    st.stop()

pages = st.navigation(
    {
        "Pipeline": [
            st.Page(
                "app_pages/overview.py",
                title="Overview",
                icon=":material/space_dashboard:",
                default=True,
                help=(
                    "Headline readiness of the whole enquiry pipeline: KPI cards, "
                    "band distribution, which fields enquiries fail to capture, and "
                    "the counsellor work queue."
                ),
            ),
            st.Page(
                "app_pages/workbench.py",
                title="Lead workbench",
                icon=":material/precision_manufacturing:",
                help=(
                    "Work one lead end to end: read its history, run NLP extraction "
                    "and completeness scoring, generate the summary and follow-up "
                    "questions, then record your review."
                ),
            ),
            st.Page(
                "app_pages/reports.py",
                title="Reports",
                icon=":material/description:",
                help=(
                    "Every qualification report already produced, with its fields, "
                    "follow-up questions and counsellor decisions. Browse and filter "
                    "here; generate new ones in the Lead workbench."
                ),
            ),
        ],
        "Operations": [
            st.Page(
                "app_pages/analytics.py",
                title="Analytics & audit",
                icon=":material/query_stats:",
                help=(
                    "SQL aggregates over the database: provider and token usage, "
                    "counsellor decisions, field coverage across leads, and the full "
                    "audit log. Always shows technical detail."
                ),
            ),
            st.Page(
                "app_pages/settings.py",
                title="Settings & diagnostics",
                icon=":material/tune:",
                help=(
                    "Load or reset the synthetic sample data, inspect the resolved "
                    "configuration and LLM provider status, and read how the "
                    "fairness controls are enforced."
                ),
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)

# --------------------------------------------------------------------------
# Sidebar chrome, shared by every page. Written after st.navigation so it
# appears below the nav.
# --------------------------------------------------------------------------
with st.sidebar:
    # Reserved now, filled at the end of the block: the status line reads the
    # "Show technical details" toggle, which is declared further down, and a
    # placeholder lets it reflect the new value on the same render rather than
    # one behind.
    status_slot = st.container()

    st.markdown("**Appearance**")
    components.appearance_switch()

    with st.expander("Data source", icon=":material/database:"):
        st.text_input(
            "Database file",
            key=state.DB_PATH_KEY,
            help="SQLite file. Changing it opens that database on the next rerun.",
        )
        st.caption("Synthetic sample data is seeded automatically into an empty file.")

    st.toggle(
        "Show technical details",
        key=state.TECHNICAL_KEY,
        help=(
            "Provider and model names, token counts, latency, response ids, the raw "
            "model payload and the prompt inspector. Off by default: a counsellor "
            "does not need them, a marker or developer does. Everything is stored "
            "either way and always visible on Analytics & audit."
        ),
    )

    with status_slot:
        provider = state.provider_status()
        if provider["live_calls_enabled"]:
            # The badge reports whether a live call is possible. Which provider
            # and model follows the technical toggle, like every other audit
            # detail — the per-report provenance line always names the model.
            if state.show_technical():
                st.success(
                    f"**{provider['provider']}** · `{provider['model']}`",
                    icon=":material/cloud_done:",
                    title="Live API ready",
                )
            else:
                st.success("Live API connected", icon=":material/cloud_done:")
        else:
            st.warning(
                "No API key configured — the offline rule-based generator will be used.",
                icon=":material/cloud_off:",
                title="Offline generator",
            )

    st.caption(
        f"{__project__} · v{__version__}  \n"
        "Synthetic sample data only. The assistant assesses information "
        "completeness, never the enquirer."
    )

pages.run()
