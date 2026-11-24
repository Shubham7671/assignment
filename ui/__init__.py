"""Presentation layer for the Streamlit counsellor UI.

Nothing in here knows how a lead is scored or how the database is queried —
that all lives in :mod:`leadqual`. This package only decides how the result is
*shown*: theme-aware colour tokens (:mod:`ui.theme`), reusable layout pieces
(:mod:`ui.components`), Altair chart builders (:mod:`ui.charts`) and the
cached resources every page shares (:mod:`ui.state`).

Keeping it separate from ``leadqual`` means the package stays importable from
the CLI and the tests without pulling in Streamlit.
"""

__all__ = ["charts", "components", "state", "theme"]
