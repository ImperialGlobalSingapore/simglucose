"""
Plotting utilities for glucose control analytics.

This module provides functions to visualize blood glucose, carbohydrate intake,
and insulin delivery data, with optional time-in-range visualizations.
"""

import math
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory
import matplotlib as mpl
import matplotlib.font_manager as fm

# Register Calibri fonts
_font_dirs = [
    "/Applications/Microsoft Word.app/Contents/Resources/DFonts/",  # macOS
    Path.home() / ".local/share/fonts/calibri/",  # Linux
]
for _font_dir in _font_dirs:
    _font_dir = Path(_font_dir)
    if _font_dir.exists():
        for font_file in _font_dir.glob("Calibri*.TTF"):
            fm.fontManager.addfont(str(font_file))
        for font_file in _font_dir.glob("Calibri*.ttf"):
            fm.fontManager.addfont(str(font_file))

# Set Calibri font globally
mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.labelweight"] = "normal"


class PatientType(Enum):
    """Patient age groups for diabetes management."""

    ADOLESCENT = "adolescent"
    ADULT = "adult"
    CHILD = "child"


class TIRCategory(Enum):
    """Time in Range category names."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    TARGET = "target"
    LOW = "low"
    VERY_LOW = "very_low"


class TIRStandard(Enum):
    """Available Time in Range standards."""

    BASIC = "basic"
    CLINICAL = "clinical"


class TIRConfig:
    """Configuration for Time in Range standards."""

    THRESHOLDS = {
        TIRStandard.BASIC: {
            TIRCategory.VERY_HIGH: 250,  # > 250
            TIRCategory.HIGH: 180,  # 180 - 250
            TIRCategory.TARGET: 70,  # 70 - 180
            TIRCategory.LOW: 0,  # 0 - 70
        },
        TIRStandard.CLINICAL: {
            TIRCategory.VERY_HIGH: 250,  # > 250
            TIRCategory.HIGH: 180,  # 180 - 250
            TIRCategory.TARGET: 70,  # 70 - 180
            TIRCategory.LOW: 54,  # 54 - 70
            TIRCategory.VERY_LOW: 0,  # 0 - 54
        },
    }

    COLORS = {
        TIRCategory.VERY_HIGH: "#FF6B35",
        TIRCategory.HIGH: "#FFB347",
        TIRCategory.TARGET: "#B5E6CC",
        TIRCategory.LOW: "#DB2020",
        TIRCategory.VERY_LOW: "#8B0000",
    }

    ORDER = {
        TIRStandard.BASIC: [
            TIRCategory.LOW,
            TIRCategory.TARGET,
            TIRCategory.HIGH,
            TIRCategory.VERY_HIGH,
        ],
        TIRStandard.CLINICAL: [
            TIRCategory.VERY_LOW,
            TIRCategory.LOW,
            TIRCategory.TARGET,
            TIRCategory.HIGH,
            TIRCategory.VERY_HIGH,
        ],
    }

    # Refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    # Acceptable ranges from paper (mean +/- SD) as percentages (0-100)
    # children aged 7-15, adult aged 16-70
    ACCEPTABLE_RANGES = {
        PatientType.CHILD: {
            TIRCategory.VERY_HIGH: (9.3 - 6.0, 9.3 + 6.0),
            TIRCategory.HIGH: (21.1 - 6.8, 21.1 + 6.8),
            TIRCategory.TARGET: (67.5 - 11.5, 67.5 + 11.5),
            TIRCategory.LOW: (2.1 - 1.5, 2.1 + 1.5),
        },
        PatientType.ADULT: {
            TIRCategory.VERY_HIGH: (5.6 - 4.9, 5.6 + 4.9),
            TIRCategory.HIGH: (18.2 - 8.4, 18.2 + 8.4),
            TIRCategory.TARGET: (74.5 - 11.9, 74.5 + 11.5),
            TIRCategory.LOW: (1.6 - 2.1, 1.6 + 2.1),
        },
    }

    def __init__(self, standard: TIRStandard = TIRStandard.BASIC):
        self.standard = standard

    def get_thresholds(self):
        return self.THRESHOLDS[self.standard]

    @staticmethod
    def get_color(category: TIRCategory):
        return TIRConfig.COLORS[category]

    def get_order(self):
        return self.ORDER[self.standard]

    @staticmethod
    def get_acceptable_ranges(patient_group: PatientType):
        return TIRConfig.ACCEPTABLE_RANGES.get(
            patient_group, TIRConfig.ACCEPTABLE_RANGES[PatientType.ADULT]
        )

    def calculate_time_in_range(self, BG_values) -> dict:
        """
        Calculate time in range statistics for blood glucose values.

        Args:
            BG_values: List of blood glucose readings in mg/dL

        Returns:
            Dictionary with percentages (0-100) for each range category
        """
        thresholds = self.get_thresholds()
        order = self.get_order()
        time_in_range = {cat: 0 for cat in order}

        for bg in BG_values:
            for category in reversed(order):
                if bg > thresholds[category]:
                    time_in_range[category] += 1
                    break
            else:
                time_in_range[order[0]] += 1

        total = len(BG_values)
        return {k: (v / total) * 100 for k, v in time_in_range.items() if v > 0}

    def get_time_in_range_acceptance(
        self,
        time_in_range,
        patient_group: PatientType,
    ) -> tuple:
        """
        Check if time in range values are within acceptable clinical ranges.

        Args:
            time_in_range: Dict with time in range statistics (as percentages 0-100)
            patient_group: PatientType enum

        Returns:
            Tuple of (category_results, acceptable_count)
        """
        if self.standard != TIRStandard.BASIC:
            raise ValueError(
                f"get_time_in_range_acceptance only supports TIRStandard.BASIC. "
                f"Current standard: {self.standard}"
            )

        acceptable_ranges = self.get_acceptable_ranges(patient_group)
        category_results = {}
        acceptable_count = 0

        for category, (min_val, max_val) in acceptable_ranges.items():
            if category not in time_in_range:
                category_results[category] = None
                continue

            is_acceptable = min_val <= time_in_range[category] <= max_val
            category_results[category] = is_acceptable

            if is_acceptable:
                acceptable_count += 1

        return (category_results, acceptable_count)


def _style_axis(ax, xlabel=None, ylabel=None):
    """Apply common styling: remove top/right borders."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_bg(ax, t, BG):
    """Plot blood glucose data on the given axis."""
    BG_mmol = [bg / 18.0 for bg in BG]

    ax.axhspan(4, 10, color="#00B050", alpha=0.15, zorder=0)
    ax.plot(t, BG_mmol, color="black", linewidth=3, label="Glucose")
    ax.axhline(y=4, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=10, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlim(0, max(t))
    ax.set_ylim(3, 12.5)
    ax.set_yticks([3, 6, 9, 12])
    _style_axis(ax, ylabel="Glucose (mmol/L)")

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y_val, label in [(10, "10 mmol/L"), (4, "4 mmol/L")]:
        ax.text(
            0.50,
            y_val,
            label,
            transform=trans,
            color="gray",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )
    ax.text(
        0.50,
        5,
        "Target glucose range",
        transform=trans,
        color="gray",
        ha="center",
        va="center",
    )


def _plot_cho(ax, t, CHO):
    """Plot carbohydrate intake data."""
    ax.plot(t, CHO)
    _style_axis(ax, ylabel="CHO (g)")


def _plot_insulin(ax, t, insulin):
    """Plot insulin delivery data."""
    ax.plot(t, insulin)
    _style_axis(ax, xlabel="Time (min)", ylabel="Insulin (U/min)")


def _plot_cho_iob(ax, t, CHO, IOB, cho_yticks=None):
    """
    Plot IOB and CHO on dual y-axes.

    Returns:
        list: Combined line handles for legend creation
    """
    iob_color = "#2596BE"
    line1 = ax.plot(t, IOB, color=iob_color, linestyle="-", linewidth=3, label="IOB")
    ax.set_ylabel("IOB (U)", color=iob_color)
    ax.tick_params(axis="y", labelcolor=iob_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(t))

    cho_color = "#E96929"
    ax2 = ax.twinx()
    line2 = ax2.plot(t, CHO, color=cho_color, linestyle="-", linewidth=3, label="CHO")
    ax2.set_ylabel("CHO (g)", color=cho_color)
    ax2.tick_params(axis="y", labelcolor=cho_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    if cho_yticks is not None:
        ax2.set_yticks(cho_yticks)

    return line1 + line2


def _plot_bg_cho_insulin(fig, ax, t, BG, CHO, insulin):
    """Create the main BG/CHO/insulin plots."""
    _plot_bg(ax[0], t, BG)
    _plot_cho(ax[1], t, CHO)
    _plot_insulin(ax[2], t, insulin)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)


def _plot_bg_cho_iob(fig, ax, t, BG, CHO, IOB):
    """Create BG and IOB plots."""
    _plot_bg(ax[0], t, BG)
    lines = _plot_cho_iob(ax[1], t, CHO, IOB)
    labels = [line.get_label() for line in lines]

    bg_handles, bg_labels = ax[0].get_legend_handles_labels()
    ax[0].legend(
        bg_handles,
        bg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )

    ax[1].legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    ax[1].set_xlabel("Time (min)")
    ax[1].xaxis.set_label_coords(0.5, -0.35)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, bottom=0.15)


def _plot_time_in_range_scale(scale_ax, time_in_range, tir_config: TIRConfig):
    """Plot time in range as a pie chart on the given axis."""
    for spine in scale_ax.spines.values():
        spine.set_visible(False)
    scale_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    scale_ax.set_facecolor("none")
    bg_box = FancyBboxPatch(
        (-1.0, -0.5),
        1.95,
        1.1,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor="#ECECEC",
        alpha=0.85,
        edgecolor="none",
        transform=scale_ax.transData,
        zorder=0,
    )
    scale_ax.add_patch(bg_box)

    order = tir_config.get_order()

    sizes = []
    colors = []
    labels = []
    categories = []

    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]
            if percentage > 0:
                sizes.append(percentage)
                colors.append(tir_config.get_color(category))
                labels.append(f"{percentage:.1f}%")
                categories.append(category)

    PIE_RADIUS = 0.5
    wedges, texts = scale_ax.pie(
        sizes,
        colors=colors,
        labels=labels,
        radius=PIE_RADIUS,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(color="black"),
    )

    if sizes:
        max_idx = sizes.index(max(sizes))
        if max_idx < len(texts):
            x, y = texts[max_idx].get_position()
            texts[max_idx].set_position((-x * 1.5, y * 0.4))
            texts[max_idx].set_color("black")

    scale_ax.set_aspect("equal")

    DOT_COLOR = "#404040"
    tar_x_label = -0.6
    tir_x_label = 0.6

    for i, wedge in enumerate(wedges):
        if categories[i] == TIRCategory.HIGH:
            mid_angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
            dot_x = PIE_RADIUS * 0.7 * math.cos(mid_angle)
            dot_y = PIE_RADIUS * 0.7 * math.sin(mid_angle)
            label_x = tar_x_label
            scale_ax.plot(dot_x, dot_y, "o", color=DOT_COLOR, markersize=5, zorder=5)
            scale_ax.plot(
                [dot_x, label_x + 0.02],
                [dot_y, 0],
                color=DOT_COLOR,
                linewidth=1.2,
                zorder=4,
            )
            scale_ax.text(
                label_x,
                0,
                "TAR",
                ha="right",
                va="center",
            )
        else:
            tir_angle = math.radians(25)
            dot_x = PIE_RADIUS * 0.6 * math.cos(tir_angle)
            dot_y = PIE_RADIUS * 0.6 * math.sin(tir_angle)
            label_x = tir_x_label
            scale_ax.plot(dot_x, dot_y, "o", color=DOT_COLOR, markersize=5, zorder=5)
            scale_ax.plot(
                [dot_x, label_x - 0.02],
                [dot_y, 0],
                color=DOT_COLOR,
                linewidth=1.2,
                zorder=4,
            )
            scale_ax.text(
                label_x,
                0,
                "TIR",
                ha="left",
                va="center",
            )


def _add_tir_inset(ax, time_in_range, tir_config):
    """Add a TIR pie chart as an inset on the given axis."""
    scale_ax = ax.inset_axes([0.62, 0.70, 0.35, 0.35], transform=ax.transAxes)
    _plot_time_in_range_scale(scale_ax, time_in_range, tir_config)


def _create_bg_cho_insulin_figure(
    t, BG, CHO, insulin, time_in_range=None, tir_config=None
):
    """Create a BG/CHO/insulin figure, optionally with TIR inset."""
    plot_height = 5
    plot_width = plot_height * 1.2
    other_height = plot_height * 0.5
    if time_in_range:
        fig_height = plot_height + other_height * 2 + 2
    else:
        fig_height = plot_height + other_height + 2

    fig = plt.figure(figsize=(plot_width + 1.5, fig_height))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    _plot_bg_cho_insulin(fig, [ax0, ax1, ax2], t, BG, CHO, insulin)

    if time_in_range and tir_config:
        _add_tir_inset(ax0, time_in_range, tir_config)

    return fig


def _create_bg_cho_iob_figure(t, BG, CHO, IOB, time_in_range=None, tir_config=None):
    """Create a BG/IOB figure, optionally with TIR inset."""
    plot_height = 5
    plot_width = plot_height * 1.2
    iob_height = plot_height * 0.5
    fig_height = plot_height + iob_height + 1.5
    extra_width = 2 if time_in_range else 1.5

    fig = plt.figure(figsize=(plot_width + extra_width, fig_height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    _plot_bg_cho_iob(fig, [ax0, ax1], t, BG, CHO, IOB)

    if time_in_range and tir_config:
        _add_tir_inset(ax0, time_in_range, tir_config)

    return fig


def plot_and_show(t, BG, CHO, insulin, time_in_range=None, tir_config=None):
    """Display BG/CHO/insulin plot, optionally with TIR."""
    _create_bg_cho_insulin_figure(t, BG, CHO, insulin, time_in_range, tir_config)
    plt.show()


def plot_and_save(t, BG, CHO, insulin, file_name, time_in_range=None, tir_config=None):
    """Save BG/CHO/insulin plot to file, optionally with TIR."""
    fig = _create_bg_cho_insulin_figure(t, BG, CHO, insulin, time_in_range, tir_config)
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_bg_cho_iob_and_show(t, BG, CHO, IOB, time_in_range=None, tir_config=None):
    """Display BG/IOB plot, optionally with TIR."""
    _create_bg_cho_iob_figure(t, BG, CHO, IOB, time_in_range, tir_config)
    plt.show()


def plot_bg_cho_iob_and_save(
    t, BG, CHO, IOB, file_name, time_in_range=None, tir_config=None
):
    """Save BG/IOB plot to file, optionally with TIR."""
    fig = _create_bg_cho_iob_figure(t, BG, CHO, IOB, time_in_range, tir_config)
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _create_comparison_figure_with_tir(
    data_left, data_right, tir_config, title_left="", title_right=""
):
    """Create a side-by-side comparison figure with shared legend."""
    plot_height = 5
    plot_width = plot_height * 1.2
    fig_height = 7
    gap_width = 0.6
    total_width = plot_width * 2 + gap_width + 2

    fig = plt.figure(figsize=(total_width, fig_height))
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[plot_width, gap_width, plot_width],
        height_ratios=[0.7, 0.35],
        wspace=0.15,
        hspace=0.35,
    )

    ax_bg_left = fig.add_subplot(gs[0, 0])
    ax_iob_left = fig.add_subplot(gs[1, 0], sharex=ax_bg_left)
    ax_bg_right = fig.add_subplot(gs[0, 2])
    ax_iob_right = fig.add_subplot(gs[1, 2], sharex=ax_bg_right)

    _plot_bg(ax_bg_left, data_left["t"], data_left["BG"])
    lines_left = _plot_cho_iob(
        ax_iob_left,
        data_left["t"],
        data_left["CHO"],
        data_left["IOB"],
        cho_yticks=[0, 25, 50, 75],
    )

    _plot_bg(ax_bg_right, data_right["t"], data_right["BG"])
    _plot_cho_iob(
        ax_iob_right,
        data_right["t"],
        data_right["CHO"],
        data_right["IOB"],
        cho_yticks=[0, 25, 50, 75],
    )

    axes_loc = (0.49, 0.58, 0.65, 0.70)
    # TIR pie chart insets (larger than single-plot insets)
    tir_inset_left = ax_bg_left.inset_axes(axes_loc, transform=ax_bg_left.transAxes)
    _plot_time_in_range_scale(tir_inset_left, data_left["time_in_range"], tir_config)

    tir_inset_right = ax_bg_right.inset_axes(axes_loc, transform=ax_bg_right.transAxes)
    _plot_time_in_range_scale(tir_inset_right, data_right["time_in_range"], tir_config)

    for ax in [ax_bg_left, ax_bg_right, ax_iob_left, ax_iob_right]:
        ax.set_xlabel("Time (min)")

    handles = lines_left
    labels = [str(h.get_label()) for h in handles]
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles),
        frameon=False,
    )

    ax_bg_left.text(
        -0.12,
        1.05,
        "a",
        transform=ax_bg_left.transAxes,
        fontsize=21,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_bg_right.text(
        -0.12,
        1.05,
        "b",
        transform=ax_bg_right.transAxes,
        fontsize=21,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    return fig


def plot_comparison_and_save_with_tir(
    data_left, data_right, file_name, tir_config, title_left="", title_right=""
):
    """Save a side-by-side comparison plot with shared legend."""
    fig = _create_comparison_figure_with_tir(
        data_left, data_right, tir_config, title_left, title_right
    )
    file_path = Path(file_name)
    svg_path = file_path.with_suffix(".svg")
    png_path = file_path.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
