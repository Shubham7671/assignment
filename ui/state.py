"""Resources and session state shared by every page.

Pages run in their own namespace under ``st.navigation``, so a helper defined
in ``app.py`` is invisible to them. Anything more than one page needs lives
here: cached resources via ``@st.cache_resource`` (one per process, shared by
all sessions) and per-user choices via ``st.session_state``.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from leadqual.config import Settings, load_settings
from leadqual.db import Database, open_database
from leadqual.exceptions import LeadQualError
from leadqual.llm.client import QualificationLLM
from leadqual.seed import seed_database

#: Settings that may arrive through Streamlit secrets instead of a ``.env``.
_SECRET_KEYS = (
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LEADQUAL_PROVIDER", "LEADQUAL_MODEL",
    "LEADQUAL_GEMINI_MODEL", "LEADQUAL_MAX_TOKENS", "LEADQUAL_OFFLINE",
    "LEADQUAL_ALLOW_FALLBACK", "LEADQUAL_GEMINI_RETRIES", "LEADQUAL_DB_PATH",
)

#: Session-state key holding the database file the user is pointed at.
DB_PATH_KEY = "db_path"

#: Session-state key behind the "Show technical details" toggle.
TECHNICAL_KEY = "show_technical"


# --------------------------------------------------------------------------
# Bootstrapping
# --------------------------------------------------------------------------
def bridge_secrets() -> None:
    """Copy Streamlit secrets into the environment.

    Locally the API key comes from a git-ignored ``.env``. On Streamlit
    Community Cloud there is no ``.env`` — the key is typed into the app's
    Secrets panel. Copying those into ``os.environ`` lets
    :func:`leadqual.config.load_settings` stay a plain environment-variable
    reader with no Streamlit dependency inside the ``leadqual`` package.

    A real environment variable always wins, so this never overrides ``.env``.
    """
    try:
        secrets = st.secrets
    except Exception:
        return  # no secrets.toml configured — normal when running locally
    for key in _SECRET_KEYS:
        try:
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
        except Exception:
            continue


@st.cache_resource(show_spinner="Preparing language resources…")
def ensure_nltk_data() -> dict[str, bool]:
    """Download the NLTK corpora if they are missing.

    A fresh Community Cloud container has none of them. The NLP layer falls
    back to regex and a built-in stopword list without them, but sentence
    segmentation and sentiment are better with them, so fetch them once per
    container. Failures are non-fatal.
    """
    results: dict[str, bool] = {}
    try:
        import nltk
    except ImportError:
        return results
    for package in ("punkt", "punkt_tab", "stopwords", "vader_lexicon"):
        try:
            results[package] = bool(nltk.download(package, quiet=True))
        except Exception:
            results[package] = False
    return results


@st.cache_resource(show_spinner=False)
def get_database(path: str) -> Database:
    """Open (once per process, per path) the SQLite database.

    On a fresh deployment the database file does not exist — it is a generated
    artefact and is git-ignored — so an empty one is seeded with the synthetic
    sample data. Without this, a deployed app would open on a blank dashboard.
    """
    database = open_database(path)
    try:
        if (database.query_pipeline_overview().get("total_leads") or 0) == 0:
            seed_database(database)
    except LeadQualError:
        pass  # a blank app beats a crash; Settings can seed manually
    return database


# --------------------------------------------------------------------------
# Accessors used by the pages
# --------------------------------------------------------------------------
def get_settings() -> Settings:
    """Environment-derived settings, read once per session.

    Re-read whenever :func:`invalidate_settings` has been called, so that
    saving a key on the Settings page takes effect without a restart.
    """
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    return st.session_state.settings


def invalidate_settings() -> None:
    """Force the next :func:`get_settings` call to re-read the environment."""
    st.session_state.pop("settings", None)


def current_db() -> Database:
    """The database the sidebar is pointed at.

    Raises:
        LeadQualError: if the file cannot be opened. Pages let this bubble up
            to ``app.py``, which has already validated the path once per run.
    """
    return get_database(st.session_state[DB_PATH_KEY])


def provider_status() -> dict[str, Any]:
    """Whether a live API call is possible, and with which provider/model."""
    return QualificationLLM(get_settings()).status()


def show_technical() -> bool:
    """Whether to show audit and debug panels (tokens, raw JSON, prompts).

    A counsellor needs none of it; a marker and a developer do. The data is
    stored and queryable either way — this only controls how much of it is on
    screen. Analytics & audit always shows the full picture.
    """
    return bool(st.session_state.get(TECHNICAL_KEY, False))
