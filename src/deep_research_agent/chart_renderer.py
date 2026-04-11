"""Deterministic chart rendering from a typed ChartParams schema.

This module is the structured-mode rendering path for the chart generator.
An LLM (the section writer planner) fills `ChartParams` via structured
output and this module renders the corresponding matplotlib figure. No
further LLM involvement — no code generation, no sandbox, no retries.

It also exposes `chart_params_to_prose` which the code-mode adapter uses
to convert the same `ChartParams` to a natural-language description for a
plotly code-generation LLM. That keeps the section writer planner
mode-agnostic: it always produces `ChartParams`, and the backend is chosen
later via `CHART_AGENT_CONFIG["mode"]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend for subprocess-safe rendering

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from src.config import CHART_AGENT_CONFIG, ROOT_DIR  # noqa: E402


CHARTS_DIR = ROOT_DIR / "output" / "charts"


# ---------------------------------------------------------------------------
# Matplotlib style (set once at import time)
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#f0f0f0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    }
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


ChartType = Literal[
    "bar",
    "horizontal_bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "pie",
    "donut",
    "scatter",
]

ValueFormat = Literal[
    "number",
    "percent",
    "currency_usd",
    "millions",
    "billions",
]

SortOrder = Literal["as_is", "ascending", "descending"]


class DataSeries(BaseModel):
    """A single series of numerical values for a chart."""

    name: str = Field(
        description=(
            "Series name shown in the legend. For single-series charts "
            "this can be an empty string — the legend will be suppressed "
            "automatically."
        )
    )
    values: list[float] = Field(
        description=(
            "Numerical values, in the same order as ChartParams.categories. "
            "Must have the same length as categories."
        )
    )


class ChartParams(BaseModel):
    """Complete specification for a single chart.

    Fill every field relevant to the chart being drawn. Optional fields
    can be left as None / empty when not applicable.
    """

    chart_type: ChartType = Field(description="The kind of chart to render.")

    title: str = Field(description="Main chart title.")
    subtitle: Optional[str] = Field(
        default=None,
        description=(
            "Optional one-line subtitle rendered below the title in smaller "
            "font. Use for context like the time period or units."
        ),
    )
    x_label: Optional[str] = Field(default=None, description="X-axis label.")
    y_label: Optional[str] = Field(default=None, description="Y-axis label.")

    categories: list[str] = Field(
        description=(
            "Category labels. For bar/line/scatter these are x-axis tick "
            "labels. For pie/donut these are slice labels. Must be non-empty."
        )
    )
    series: list[DataSeries] = Field(
        description=(
            "One or more data series. pie and donut must have exactly one "
            "series; bar/line/scatter/grouped_bar/stacked_bar may have "
            "multiple. Every series must have the same number of values as "
            "categories."
        )
    )

    value_format: ValueFormat = Field(
        default="number",
        description=(
            "How to format numeric values in data labels and axis ticks. "
            "'percent' assumes values are already on the 0-100 scale."
        ),
    )
    show_data_labels: bool = Field(
        default=True,
        description=(
            "Whether to render value labels on bars, points, and pie slices. "
            "Default True — research reports benefit from visible numbers."
        ),
    )

    sort_order: SortOrder = Field(
        default="as_is",
        description=(
            "Category sort order. 'as_is' preserves the given order (use "
            "for time series). 'descending' is common for ranking bar "
            "charts. Ignored for line and scatter."
        ),
    )
    y_axis_min: Optional[float] = Field(
        default=None,
        description=(
            "Optional lower bound for the value axis. Use sparingly — only "
            "when the data range genuinely benefits from cropping."
        ),
    )
    y_axis_max: Optional[float] = Field(
        default=None,
        description="Optional upper bound for the value axis.",
    )

    highlight_category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category name to highlight in a contrasting color. "
            "Must exactly match one of the entries in `categories` if set. "
            "Applies to single-series bar, horizontal_bar, pie and donut."
        ),
    )

    source_note: Optional[str] = Field(
        default=None,
        description=(
            "Optional small footer text rendered below the chart "
            "(e.g. 'Source: Q3 2024 filings'). Keep it short."
        ),
    )

    @field_validator("categories")
    @classmethod
    def _categories_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("categories must be non-empty")
        return v

    @field_validator("series")
    @classmethod
    def _series_nonempty(cls, v: list[DataSeries]) -> list[DataSeries]:
        if not v:
            raise ValueError("series must contain at least one DataSeries")
        return v


@dataclass
class ChartResult:
    """Outcome of a single chart generation attempt."""

    success: bool
    chart_name: str
    image_markdown: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_value(value: float, fmt: ValueFormat) -> str:
    if fmt == "percent":
        return f"{value:.1f}%"
    if fmt == "currency_usd":
        absv = abs(value)
        if absv >= 1_000_000_000:
            return f"${value / 1_000_000_000:.1f}B"
        if absv >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        if absv >= 1_000:
            return f"${value / 1_000:.1f}K"
        return f"${value:.0f}"
    if fmt == "millions":
        return f"{value:.1f}M"
    if fmt == "billions":
        return f"{value:.1f}B"
    # "number"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _sorted_indices(values: list[float], order: SortOrder) -> list[int]:
    if order == "as_is":
        return list(range(len(values)))
    reverse = order == "descending"
    return sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)


def _palette_colors(n: int, highlight_idx: Optional[int] = None) -> list[str]:
    palette: list[str] = CHART_AGENT_CONFIG["palette"]
    colors = [palette[i % len(palette)] for i in range(n)]
    if highlight_idx is not None and 0 <= highlight_idx < n:
        colors[highlight_idx] = CHART_AGENT_CONFIG["highlight_color"]
    return colors


def _highlight_index_in(
    params: ChartParams, cats_in_order: list[str]
) -> Optional[int]:
    if not params.highlight_category:
        return None
    try:
        return cats_in_order.index(params.highlight_category)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Validation beyond Pydantic
# ---------------------------------------------------------------------------


def _validate_params(params: ChartParams) -> None:
    n_cats = len(params.categories)
    for s in params.series:
        if len(s.values) != n_cats:
            raise ValueError(
                f"Series '{s.name}' has {len(s.values)} values but there "
                f"are {n_cats} categories."
            )

    if params.chart_type in ("pie", "donut") and len(params.series) != 1:
        raise ValueError(
            f"{params.chart_type} requires exactly one series, "
            f"got {len(params.series)}."
        )


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


def _compose_title(fig, params: ChartParams) -> None:
    if params.subtitle:
        fig.suptitle(params.title, fontsize=14, fontweight="bold", y=0.98)
        fig.text(
            0.5,
            0.925,
            params.subtitle,
            ha="center",
            fontsize=10,
            color="#666666",
        )
    else:
        fig.suptitle(params.title, fontsize=14, fontweight="bold", y=0.97)


def _compose_source_note(fig, params: ChartParams) -> None:
    if params.source_note:
        fig.text(
            0.5,
            0.01,
            params.source_note,
            ha="center",
            fontsize=8,
            color="#888888",
            style="italic",
        )


def _apply_y_limits(ax, params: ChartParams) -> None:
    if params.y_axis_min is None and params.y_axis_max is None:
        return
    lo, hi = ax.get_ylim()
    if params.y_axis_min is not None:
        lo = params.y_axis_min
    if params.y_axis_max is not None:
        hi = params.y_axis_max
    ax.set_ylim(lo, hi)


def _set_category_xticks(ax, cats: list[str]) -> None:
    ax.set_xticks(list(range(len(cats))))
    max_len = max((len(c) for c in cats), default=0)
    rotation = 30 if max_len > 8 else 0
    ax.set_xticklabels(
        cats,
        rotation=rotation,
        ha="right" if rotation else "center",
    )


def _value_axis_formatter(fmt: ValueFormat) -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: _format_value(v, fmt))


# ---------------------------------------------------------------------------
# Individual chart renderers
# ---------------------------------------------------------------------------


def _render_bar(ax, params: ChartParams) -> None:
    """Vertical bar, grouped bar, and stacked bar."""
    n_series = len(params.series)
    is_stacked = params.chart_type == "stacked_bar"
    is_grouped = params.chart_type == "grouped_bar" or (
        params.chart_type == "bar" and n_series > 1
    )
    fmt = params.value_format

    # Sort order is based on the total (stacked) or the first series (otherwise)
    if is_stacked and n_series > 1:
        totals = [
            sum(s.values[i] for s in params.series)
            for i in range(len(params.categories))
        ]
        order = _sorted_indices(totals, params.sort_order)
    else:
        order = _sorted_indices(params.series[0].values, params.sort_order)

    cats = [params.categories[i] for i in order]
    x = list(range(len(cats)))

    if is_stacked and n_series > 1:
        bottom = [0.0] * len(cats)
        colors = _palette_colors(n_series)
        for si, series in enumerate(params.series):
            vals = [series.values[i] for i in order]
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=colors[si],
                label=series.name or None,
                width=0.65,
                edgecolor="white",
                linewidth=0.5,
            )
            if params.show_data_labels:
                for xi, v, b in zip(x, vals, bottom):
                    if v != 0:
                        ax.text(
                            xi,
                            b + v / 2,
                            _format_value(v, fmt),
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white",
                            fontweight="bold",
                        )
            bottom = [b + v for b, v in zip(bottom, vals)]
    elif is_grouped and n_series > 1:
        group_width = 0.8
        bar_width = group_width / n_series
        colors = _palette_colors(n_series)
        for si, series in enumerate(params.series):
            offsets = [
                xi - group_width / 2 + bar_width * (si + 0.5) for xi in x
            ]
            vals = [series.values[i] for i in order]
            bars = ax.bar(
                offsets,
                vals,
                width=bar_width,
                color=colors[si],
                label=series.name or None,
            )
            if params.show_data_labels:
                for b, v in zip(bars, vals):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height(),
                        _format_value(v, fmt),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
    else:
        vals = [params.series[0].values[i] for i in order]
        highlight_idx = _highlight_index_in(params, cats)
        colors = _palette_colors(len(vals), highlight_idx=highlight_idx)
        bars = ax.bar(
            x,
            vals,
            color=colors,
            width=0.65,
            edgecolor="white",
            linewidth=0.5,
        )
        if params.show_data_labels:
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    _format_value(v, fmt),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    _set_category_xticks(ax, cats)
    if params.x_label:
        ax.set_xlabel(params.x_label)
    if params.y_label:
        ax.set_ylabel(params.y_label)
    _apply_y_limits(ax, params)
    ax.yaxis.set_major_formatter(_value_axis_formatter(fmt))
    ax.xaxis.grid(False)

    if n_series > 1 and any(s.name for s in params.series):
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))


def _render_horizontal_bar(ax, params: ChartParams) -> None:
    """Single-series horizontal bar (most common research-report use)."""
    order = _sorted_indices(params.series[0].values, params.sort_order)
    cats = [params.categories[i] for i in order]
    vals = [params.series[0].values[i] for i in order]
    fmt = params.value_format

    highlight_idx = _highlight_index_in(params, cats)
    colors = _palette_colors(len(vals), highlight_idx=highlight_idx)

    y = list(range(len(cats)))
    bars = ax.barh(
        y,
        vals,
        color=colors,
        height=0.65,
        edgecolor="white",
        linewidth=0.5,
    )
    if params.show_data_labels:
        for b, v in zip(bars, vals):
            ax.text(
                b.get_width(),
                b.get_y() + b.get_height() / 2,
                f"  {_format_value(v, fmt)}",
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()  # first category on top

    # For horizontal bars, the value axis is x
    if params.x_label:
        ax.set_xlabel(params.x_label)
    if params.y_label:
        ax.set_ylabel(params.y_label)

    # Apply value-axis limits on x instead of y
    if params.y_axis_min is not None or params.y_axis_max is not None:
        lo, hi = ax.get_xlim()
        if params.y_axis_min is not None:
            lo = params.y_axis_min
        if params.y_axis_max is not None:
            hi = params.y_axis_max
        ax.set_xlim(lo, hi)

    ax.xaxis.set_major_formatter(_value_axis_formatter(fmt))
    ax.yaxis.grid(False)


def _render_line(ax, params: ChartParams) -> None:
    cats = params.categories
    x = list(range(len(cats)))
    n_series = len(params.series)
    colors = _palette_colors(n_series)
    fmt = params.value_format

    for si, series in enumerate(params.series):
        ax.plot(
            x,
            series.values,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=colors[si],
            label=series.name or None,
        )
        # Only label points when there are at most 2 series (otherwise clutter)
        if params.show_data_labels and n_series <= 2:
            for xi, v in zip(x, series.values):
                ax.annotate(
                    _format_value(v, fmt),
                    xy=(xi, v),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=colors[si],
                )

    _set_category_xticks(ax, cats)
    if params.x_label:
        ax.set_xlabel(params.x_label)
    if params.y_label:
        ax.set_ylabel(params.y_label)
    _apply_y_limits(ax, params)
    ax.yaxis.set_major_formatter(_value_axis_formatter(fmt))

    if n_series > 1 and any(s.name for s in params.series):
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))


def _render_pie(ax, params: ChartParams, donut: bool) -> None:
    order = _sorted_indices(params.series[0].values, params.sort_order)
    cats = [params.categories[i] for i in order]
    vals = [params.series[0].values[i] for i in order]

    highlight_idx = _highlight_index_in(params, cats)
    colors = _palette_colors(len(vals), highlight_idx=highlight_idx)

    def autopct(pct: float) -> str:
        return f"{pct:.1f}%" if params.show_data_labels else ""

    _wedges, _labels, autotexts = ax.pie(
        vals,
        labels=cats,
        colors=colors,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, color="#333333"),
        pctdistance=0.72 if donut else 0.68,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    if donut:
        ax.add_artist(Circle((0, 0), 0.55, fc="white"))

    ax.axis("equal")
    ax.grid(False)


def _render_scatter(ax, params: ChartParams) -> None:
    cats = params.categories
    # If all category labels parse as floats, use them as numeric x; otherwise
    # lay them out on integer positions with tick labels.
    try:
        x_values = [float(c) for c in cats]
        numeric_x = True
    except ValueError:
        x_values = list(range(len(cats)))
        numeric_x = False

    colors = _palette_colors(len(params.series))
    for si, series in enumerate(params.series):
        ax.scatter(
            x_values,
            series.values,
            s=70,
            color=colors[si],
            edgecolor="white",
            linewidth=1,
            label=series.name or None,
            zorder=3,
        )

    if not numeric_x:
        _set_category_xticks(ax, cats)

    if params.x_label:
        ax.set_xlabel(params.x_label)
    if params.y_label:
        ax.set_ylabel(params.y_label)
    _apply_y_limits(ax, params)
    ax.yaxis.set_major_formatter(_value_axis_formatter(params.value_format))

    if len(params.series) > 1 and any(s.name for s in params.series):
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_chart(chart_name: str, params: ChartParams) -> ChartResult:
    """Render a chart to PNG from a ChartParams specification.

    Args:
        chart_name: snake_case identifier used for the output filename.
        params: Complete chart specification.

    Returns:
        ChartResult. On success, `image_markdown` contains the markdown
        snippet to embed in the section. On failure, `error` describes the
        reason; callers typically drop failed charts silently.
    """
    try:
        _validate_params(params)
    except Exception as e:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error=f"Invalid chart params: {e}",
        )

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = CHARTS_DIR / f"{chart_name}.png"

    is_pie = params.chart_type in ("pie", "donut")
    figsize = (
        tuple(CHART_AGENT_CONFIG["figsize_pie"])
        if is_pie
        else tuple(CHART_AGENT_CONFIG["figsize"])
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=CHART_AGENT_CONFIG["dpi"])

    try:
        if params.chart_type in ("bar", "grouped_bar", "stacked_bar"):
            _render_bar(ax, params)
        elif params.chart_type == "horizontal_bar":
            _render_horizontal_bar(ax, params)
        elif params.chart_type == "line":
            _render_line(ax, params)
        elif params.chart_type == "pie":
            _render_pie(ax, params, donut=False)
        elif params.chart_type == "donut":
            _render_pie(ax, params, donut=True)
        elif params.chart_type == "scatter":
            _render_scatter(ax, params)
        else:  # pragma: no cover — Literal type prevents this
            raise ValueError(f"Unsupported chart_type: {params.chart_type}")

        _compose_title(fig, params)
        _compose_source_note(fig, params)

        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    except Exception as e:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error=f"Matplotlib rendering failed: {str(e)[:200]}",
        )
    finally:
        plt.close(fig)

    return ChartResult(
        success=True,
        chart_name=chart_name,
        image_markdown=f"![{chart_name}](charts/{chart_name}.png)",
    )


# ---------------------------------------------------------------------------
# Prose adapter for code-mode (deterministic, no LLM)
# ---------------------------------------------------------------------------


def chart_params_to_prose(params: ChartParams) -> str:
    """Render ChartParams to natural-language prose for the code-mode agent.

    Used by `chart_generator._generate_via_code_agent` when the chart backend
    is set to `"code"`. This keeps the section writer planner backend-agnostic
    — it always emits ChartParams — while still feeding the legacy plotly
    code-generation LLM a description it can consume.
    """
    readable_type = params.chart_type.replace("_", " ")
    lines: list[str] = [f"Create a {readable_type} chart."]
    lines.append(f'Title: "{params.title}".')
    if params.subtitle:
        lines.append(f'Subtitle: "{params.subtitle}".')
    if params.x_label:
        lines.append(f'X-axis label: "{params.x_label}".')
    if params.y_label:
        lines.append(f'Y-axis label: "{params.y_label}".')

    lines.append(f"Categories (in order): {params.categories}.")
    if len(params.series) == 1:
        s = params.series[0]
        tag = f" (series name: '{s.name}')" if s.name else ""
        lines.append(f"Values{tag}: {s.values}.")
    else:
        lines.append(f"{len(params.series)} data series:")
        for s in params.series:
            lines.append(f"  - {s.name or '(unnamed)'}: {s.values}")

    lines.append(f"Value format: {params.value_format}.")
    lines.append(
        f"Show data labels on points/bars/slices: "
        f"{'yes' if params.show_data_labels else 'no'}."
    )
    if params.sort_order != "as_is":
        lines.append(
            f"Sort categories {params.sort_order} by value before plotting."
        )
    if params.y_axis_min is not None:
        lines.append(f"Set the value-axis minimum to {params.y_axis_min}.")
    if params.y_axis_max is not None:
        lines.append(f"Set the value-axis maximum to {params.y_axis_max}.")
    if params.highlight_category:
        lines.append(
            f"Highlight the '{params.highlight_category}' category with a "
            "contrasting color."
        )
    if params.source_note:
        lines.append(f'Footer / source note: "{params.source_note}".')

    lines.append(
        "Use exactly these values — do not invent, approximate, round, or "
        "omit any data. Aim for a clean, professional look suitable for "
        "static PNG export in a research report."
    )
    return "\n".join(lines)
