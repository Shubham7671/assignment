"""Theme-aware visual tokens.

Almost all colour in this app is configuration, not code: ``.streamlit/
config.toml`` defines a light and a dark palette and every widget, dataframe
and Altair chart picks the right one automatically.

Two things cannot be expressed in ``config.toml`` and therefore live here:

* **Readiness band colour.** READY / NEEDS_INFO / NOT_READY is a *status*
  encoding, not a categorical series, so it must not come from
  ``chartCategoricalColors`` — a chart that regrouped its data would silently
  swap the meaning of the colours. The three hexes are pinned here, once per
  appearance, and chosen to clear 3:1 against their own surface.
* **Altair axis furniture.** Gridlines, tick labels and axis titles are drawn
  by Altair, which has no access to the Streamlit theme tokens.

Everything reads :data:`streamlit.context.theme`, so a mode switch repaints
the whole app on the next rerun. Colour is never the only signal: every band
also carries a glyph and its text label.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

Appearance = Literal["light", "dark"]

#: Worst-last ordering, used for chart axes and legends alike.
BAND_ORDER: tuple[str, ...] = ("READY", "NEEDS_INFO", "NOT_READY")

#: Redundant non-colour encoding for each band.
BAND_GLYPHS: dict[str, str] = {
    "READY": "●",
    "NEEDS_INFO": "◐",
    "NOT_READY": "○",
}

#: Material Symbols shown next to a band in headings, badges and alerts.
BAND_ICONS: dict[str, str] = {
    "READY": ":material/check_circle:",
    "NEEDS_INFO": ":material/pending:",
    "NOT_READY": ":material/error:",
}

#: Streamlit semantic colour names — these follow the theme on their own.
BAND_BADGE_COLORS: dict[str, str] = {
    "READY": "green",
    "NEEDS_INFO": "orange",
    "NOT_READY": "red",
}

#: Status hexes for Altair, per appearance. Kept in step with the
#: ``greenColor`` / ``yellowColor`` / ``redColor`` entries in config.toml.
_BAND_HEX: dict[str, dict[str, str]] = {
    "light": {
        "READY": "#0E8A3E",
        "NEEDS_INFO": "#B4690E",
        "NOT_READY": "#C62828",
    },
    "dark": {
        "READY": "#3DD68C",
        "NEEDS_INFO": "#F5B342",
        "NOT_READY": "#FF6B6B",
    },
}

#: Chart furniture: text, muted text, gridlines and the single accent hue used
#: by one-series magnitude charts. Mirrors config.toml's surfaces.
_INK: dict[str, dict[str, str]] = {
    "light": {
        "text": "#15141A",
        "muted": "#6B6A64",
        "grid": "#E4E3DC",
        "accent": "#4338CA",
        "accent_soft": "#C2C8F7",
    },
    "dark": {
        "text": "#E8E8F2",
        "muted": "#9A99AD",
        "grid": "#2A2A3E",
        "accent": "#818CF8",
        "accent_soft": "#3C3B85",
    },
}


def appearance() -> Appearance:
    """The appearance the browser is currently rendering.

    ``st.context.theme.type`` is derived from the resolved background colour.
    It can briefly be wrong on the very first run of a session and right after
    the user flips the appearance, until the next rerun — so this never drives
    anything but colour, and light is the safe default.
    """
    try:
        value = st.context.theme.type
    except Exception:  # pragma: no cover - older runtimes / bare script runs
        return "light"
    return "dark" if value == "dark" else "light"


def band_colours() -> dict[str, str]:
    """Readiness band → hex, for the active appearance."""
    return dict(_BAND_HEX[appearance()])


def band_range() -> list[str]:
    """Band hexes in :data:`BAND_ORDER`, ready for an Altair colour scale."""
    colours = band_colours()
    return [colours[band] for band in BAND_ORDER]


def ink() -> dict[str, str]:
    """Chart furniture colours for the active appearance."""
    return dict(_INK[appearance()])


def band_title(band: str) -> str:
    """Human-readable band name, e.g. ``NEEDS_INFO`` → ``Needs info``."""
    return band.replace("_", " ").capitalize()


def band_label(band: str) -> str:
    """Glyph + text, so an axis label never relies on colour alone."""
    return f"{BAND_GLYPHS.get(band, '·')} {band_title(band)}"


def band_icon(band: str) -> str:
    """Material Symbols token for a band, with a neutral fallback."""
    return BAND_ICONS.get(band, ":material/help:")


def band_badge_color(band: str) -> str:
    """Streamlit semantic colour name for a band badge."""
    return BAND_BADGE_COLORS.get(band, "gray")
