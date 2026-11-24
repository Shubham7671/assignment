"""Altair chart builders for the dashboard.

Design rules these charts follow, and the reason for each:

* **One engine per app.** Everything is Altair, so tooltips, hover and
  selection gestures behave the same on every page.
* **Status colour is never categorical colour.** Readiness bands come from
  :func:`ui.theme.band_range`; a chart that is merely showing magnitude uses a
  single accent hue and no legend.
* **Colour is never the only signal.** Band axes are labelled with a glyph and
  the band name, and counts are direct-labelled at the end of the bar.
* **No hard-coded surface.** Charts are drawn on a transparent background and
  take their text, gridline and accent colours from
  :func:`ui.theme.ink`, so the same code renders correctly in light and dark.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from ui import theme


def _axis(*, title: str | None = None, grid: bool = False, **kwargs) -> alt.Axis:
    """An axis styled for the active appearance.

    Domains and ticks are off everywhere: the gridlines already locate a
    value, and the extra chrome only competes with the data.
    """
    palette = theme.ink()
    return alt.Axis(
        title=title,
        grid=grid,
        gridColor=palette["grid"],
        gridDash=[2, 3],
        labelColor=palette["muted"],
        labelFontSize=12,
        titleColor=palette["muted"],
        titleFontSize=12,
        titleFontWeight="normal",
        domain=False,
        ticks=False,
        **kwargs,
    )


def _category_axis(*, label_angle: float = 0, label_font_size: int = 13) -> alt.Axis:
    """Axis for the identifying (nominal) dimension — a touch more prominent."""
    palette = theme.ink()
    return alt.Axis(
        title=None,
        labelAngle=label_angle,
        labelColor=palette["text"],
        labelFontSize=label_font_size,
        labelLimit=240,
        domain=False,
        ticks=False,
    )


def _frame(chart: alt.Chart, *, height: int) -> alt.Chart:
    """Transparent background so the card's surface shows through."""
    return chart.properties(height=height, background="transparent")


# --------------------------------------------------------------------------
# Pipeline readiness
# --------------------------------------------------------------------------
def readiness_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Horizontal status bars: how many leads sit in each readiness band.

    Identity is carried by the axis label, so colour is redundant and there is
    no legend. Counts are direct-labelled at the bar end rather than left to
    be read off the axis.
    """
    data = frame.copy()
    data["label"] = data["band"].map(theme.band_label)
    palette = theme.ink()

    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N",
            sort=[theme.band_label(band) for band in theme.BAND_ORDER],
            title=None,
            axis=_category_axis(),
            scale=alt.Scale(paddingInner=0.4),
        ),
        x=alt.X("lead_count:Q", title="Leads", axis=_axis(title="Leads", grid=True, tickCount=5)),
        tooltip=[
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("lead_count:Q", title="Leads"),
            alt.Tooltip("share_pct:Q", title="Share %", format=".1f"),
            alt.Tooltip("avg_score:Q", title="Average score", format=".0f"),
        ],
    )

    bars = base.mark_bar(cornerRadiusEnd=5, height=24).encode(
        color=alt.Color(
            "band:N",
            scale=alt.Scale(domain=list(theme.BAND_ORDER), range=theme.band_range()),
            legend=None,
        )
    )
    labels = base.mark_text(align="left", dx=8, color=palette["text"], fontSize=12).encode(
        text=alt.Text("lead_count:Q")
    )
    return _frame(bars + labels, height=170)


def missing_fields_chart(frame: pd.DataFrame, *, top_n: int = 10) -> alt.LayerChart:
    """Single-series magnitude chart: how often each field is missing.

    One series, one hue, no legend — the axis title names the measure. Bars
    for fields the pipeline cannot run without are drawn at full strength;
    optional fields are muted, so the eye lands on what actually blocks a
    consultation.
    """
    data = frame.head(top_n).copy()
    palette = theme.ink()

    base = alt.Chart(data).encode(
        y=alt.Y("field_label:N", sort="-x", title=None, axis=_category_axis(),
                scale=alt.Scale(paddingInner=0.35)),
        x=alt.X(
            "missing_pct:Q",
            scale=alt.Scale(domain=[0, 100]),
            axis=_axis(title="% of assessed enquiries missing the field", grid=True, tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("field_label:N", title="Field"),
            alt.Tooltip("missing_pct:Q", title="Missing %", format=".0f"),
            alt.Tooltip("missing_count:Q", title="Leads missing it"),
            alt.Tooltip("total_count:Q", title="Leads assessed"),
            alt.Tooltip("is_required:N", title="Required"),
        ],
    )

    bars = base.mark_bar(cornerRadiusEnd=5, height=18).encode(
        color=alt.Color(
            "is_required:N",
            scale=alt.Scale(
                domain=[True, False],
                range=[palette["accent"], palette["accent_soft"]],
            ),
            legend=None,
        )
    )
    labels = base.mark_text(align="left", dx=8, color=palette["text"], fontSize=11).encode(
        text=alt.Text("missing_pct:Q", format=".0f")
    )
    return _frame(bars + labels, height=max(200, 30 * len(data)))


# --------------------------------------------------------------------------
# Report-level views
# --------------------------------------------------------------------------
def score_distribution_chart(frame: pd.DataFrame) -> alt.Chart:
    """Where readiness scores actually cluster, in ten-point bins.

    The band distribution says how many leads are READY; this says whether the
    NEEDS_INFO group is one short question away from ready or nowhere near —
    a different operational answer each time.
    """
    data = frame.copy()
    return _frame(
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "readiness_score:Q",
                bin=alt.Bin(step=10, extent=[0, 100]),
                axis=_axis(title="Readiness score", grid=False, tickCount=10),
            ),
            y=alt.Y("count():Q", axis=_axis(title="Reports", grid=True, tickCount=4)),
            color=alt.Color(
                "readiness_band:N",
                scale=alt.Scale(domain=list(theme.BAND_ORDER), range=theme.band_range()),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelColor=theme.ink()["muted"],
                    labelFontSize=12,
                    symbolType="square",
                ),
            ),
            tooltip=[
                alt.Tooltip("readiness_band:N", title="Band"),
                alt.Tooltip("count():Q", title="Reports"),
            ],
        ),
        height=220,
    )


def activity_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Assessment volume over time, with the running average score on top.

    Two measures on one time axis: how much work went through (area) and
    whether the enquiries getting through were any more complete (line).
    """
    data = frame.copy()
    palette = theme.ink()

    base = alt.Chart(data).encode(
        x=alt.X("day:T", axis=_axis(title=None, grid=False, format="%d %b"))
    )
    area = base.mark_area(
        line={"color": palette["accent"], "strokeWidth": 2},
        color=palette["accent"],
        opacity=0.18,
        interpolate="monotone",
    ).encode(
        y=alt.Y("reports:Q", axis=_axis(title="Reports", grid=True, tickCount=4)),
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %b %Y"),
            alt.Tooltip("reports:Q", title="Reports"),
            alt.Tooltip("avg_score:Q", title="Average score", format=".0f"),
        ],
    )
    points = base.mark_circle(size=48, color=palette["accent"]).encode(
        y=alt.Y("reports:Q"),
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %b %Y"),
            alt.Tooltip("reports:Q", title="Reports"),
            alt.Tooltip("avg_score:Q", title="Average score", format=".0f"),
        ],
    )
    return _frame(area + points, height=220)


def coverage_heatmap(frame: pd.DataFrame) -> alt.Chart:
    """Lead x field grid of what the enquiry captured.

    A heatmap, not a table, because the question is about *patterns*: a dark
    column is a field the enquiry form never collects, a dark row is one lead
    the counsellor has almost nothing on.
    """
    palette = theme.ink()
    return _frame(
        alt.Chart(frame)
        .mark_rect(cornerRadius=3, stroke="transparent", strokeWidth=1)
        .encode(
            x=alt.X("field:N", title=None, axis=_category_axis(label_angle=-40, label_font_size=11)),
            y=alt.Y("lead:N", title=None, axis=_category_axis(label_font_size=11)),
            color=alt.Color(
                "present:Q",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=[palette["accent_soft"], palette["accent"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("lead:N", title="Lead"),
                alt.Tooltip("field:N", title="Field"),
                alt.Tooltip("present:N", title="On file"),
            ],
        ),
        height=max(220, 22 * frame["lead"].nunique()),
    )
