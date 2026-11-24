"""Reusable UI pieces.

Every helper here is built from native Streamlit elements so it inherits the
light and dark palettes from ``.streamlit/config.toml`` automatically. The one
place that reaches for raw HTML is :func:`_apply_appearance`, which has to talk
to the browser's local storage — see the note there.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import streamlit as st

from ui import theme


# --------------------------------------------------------------------------
# Page chrome
# --------------------------------------------------------------------------
def page_header(title: str, *, icon: str, subtitle: str | None = None) -> None:
    """The title block every page opens with.

    Args:
        title: Page title, in sentence case.
        icon: A Material Symbols token, e.g. ``":material/dashboard:"``.
        subtitle: One line explaining what the page is for.
    """
    st.title(title, icon=icon)
    if subtitle:
        st.caption(subtitle)


@contextmanager
def card(
    title: str | None = None,
    *,
    icon: str | None = None,
    caption: str | None = None,
    height: Any = "content",
) -> Iterator[Any]:
    """A bordered panel with an optional heading.

    Cards are how this dashboard groups things: one question per card, its
    heading stated as that question, and the answer inside.
    """
    container = st.container(border=True, height=height)
    with container:
        if title:
            st.subheader(title, icon=icon)
        if caption:
            st.caption(caption)
        yield container


def empty_state(message: str, *, icon: str = ":material/inbox:", hint: str | None = None) -> None:
    """What a card shows when there is genuinely nothing to show.

    A blank space reads as a bug; a sentence saying *why* it is blank, and
    what to do about it, does not.
    """
    with st.container(horizontal_alignment="center", gap=None):
        st.markdown(f"### {icon}", text_alignment="center")
        st.markdown(f"**{message}**", text_alignment="center")
        if hint:
            st.caption(hint, text_alignment="center")


# --------------------------------------------------------------------------
# Readiness band presentation
# --------------------------------------------------------------------------
def band_badge(band: str, score: float | None = None, *, help: str | None = None) -> None:
    """A readiness badge carrying colour, icon and text together.

    ``st.badge`` uses the theme's semantic colours, so the badge is legible in
    both appearances without any colour of ours.
    """
    label = theme.band_title(band)
    if score is not None:
        label = f"{label} · {score:.0f}/100"
    st.badge(
        label,
        icon=theme.band_icon(band),
        color=theme.band_badge_color(band),
        help=help,
    )


def band_chip(band: str) -> str:
    """Inline Markdown badge for a band, for use inside a sentence."""
    colour = theme.band_badge_color(band)
    return f":{colour}-badge[{theme.band_title(band)}]"


def band_cell(band: str) -> list[str]:
    """Wrap a band name for :func:`band_column`.

    ``MultiselectColumn`` renders list values as coloured chips, so a
    single-band cell is a one-item list.
    """
    return [theme.band_title(band)]


def band_column(label: str = "Readiness") -> Any:
    """A dataframe column that shows readiness bands as coloured chips.

    The colours are mapped to ``options`` explicitly rather than left on
    ``"auto"``: band colour is a status encoding, so it must not shift when
    the set of bands present in a given table changes.
    """
    return st.column_config.MultiselectColumn(
        label,
        options=[theme.band_title(band) for band in theme.BAND_ORDER],
        color=[theme.band_badge_color(band) for band in theme.BAND_ORDER],
        help="Readiness band of the lead's most recent assessment.",
    )


def score_column(label: str = "Score") -> Any:
    """A 0–100 readiness score shown as a progress bar."""
    return st.column_config.ProgressColumn(
        label,
        min_value=0,
        max_value=100,
        format="%d",
        help="Weighted completeness of the qualification fields, out of 100.",
    )


# --------------------------------------------------------------------------
# KPI wall
# --------------------------------------------------------------------------
def kpi_wall(metrics: Sequence[dict[str, Any]], *, columns: int = 5) -> None:
    """A responsive row of bordered metric cards.

    ``st.grid`` rather than ``st.columns``: it drops to fewer columns as the
    viewport narrows instead of stacking every card at once, and
    ``height="stretch"`` keeps cards with and without a delta the same height.

    Args:
        metrics: Dicts of keyword arguments for ``st.metric`` — at least
            ``label`` and ``value``.
        columns: Maximum number of cards per row.
    """
    with st.grid(columns, gap="small"):
        for metric in metrics:
            st.metric(border=True, height="stretch", **metric)


# --------------------------------------------------------------------------
# Appearance switch
# --------------------------------------------------------------------------
#: Streamlit's frontend remembers the chosen appearance in local storage under
#: ``stActiveTheme-<pathname>-v<schema>``, as the JSON string "Light", "Dark"
#: or "System". It reads schema 2 today; writing the next couple of versions
#: too means an upgrade that bumps the schema does not silently break this.
_APPEARANCE_SCRIPT = """
<script>
(function () {
  try {
    const path = window.location.pathname;
    const value = JSON.stringify("__CHOICE__");
    for (const version of [2, 3, 4]) {
      window.localStorage.setItem(`stActiveTheme-${path}-v${version}`, value);
    }
  } catch (error) {
    /* Private browsing or a blocked storage API: fall back to the built-in
       Settings menu rather than throwing. */
  }
  window.location.reload();
})();
</script>
"""


def _apply_appearance(choice: str) -> None:
    """Persist an appearance choice in the browser and reload.

    Streamlit resolves the theme in the browser, from local storage, before
    the Python script runs, and theme options are explicitly not settable from
    a script (``st.set_option`` rejects them). So an *in-app* switch has to
    write the same local-storage entry the built-in Settings → Appearance
    control writes, then reload so the frontend picks it up.

    This is the one deliberately non-native thing in the UI. If a future
    Streamlit release changes that storage schema, the switch simply stops
    having an effect: the built-in Settings → Appearance control keeps
    working either way, and nothing here can break the app.
    """
    if choice not in {"Light", "Dark"}:  # never interpolate an unchecked value
        raise ValueError(f"unsupported appearance: {choice!r}")
    st.html(
        _APPEARANCE_SCRIPT.replace("__CHOICE__", choice),
        unsafe_allow_javascript=True,
    )


def appearance_switch() -> None:
    """Light/dark switch for the sidebar.

    Buttons rather than a stateful widget on purpose: each click is a one-shot
    action that reloads the page, so there is no widget value left behind to
    re-trigger a second reload.
    """
    current = theme.appearance()
    note = (
        "Switching reloads the page, which starts a new session — anything you "
        "have generated but not saved is cleared. The same control is in "
        "Settings → Appearance (top-right menu), which switches without a reload."
    )
    with st.container(horizontal=True, gap="small"):
        to_light = st.button(
            "Light",
            icon=":material/light_mode:",
            key="appearance_light",
            type="primary" if current == "light" else "secondary",
            width="stretch",
            help=note,
        )
        to_dark = st.button(
            "Dark",
            icon=":material/dark_mode:",
            key="appearance_dark",
            type="primary" if current == "dark" else "secondary",
            width="stretch",
            help=note,
        )
    if to_light and current != "light":
        _apply_appearance("Light")
    elif to_dark and current != "dark":
        _apply_appearance("Dark")
