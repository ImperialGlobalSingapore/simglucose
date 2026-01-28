"""
Plotting utilities for glucose control analytics.

This module provides functions to visualize blood glucose, carbohydrate intake,
and insulin delivery data, with optional time-in-range visualizations.
"""

from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib as mpl

# Set Calibri font globally
mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["axes.labelweight"] = "bold"


class PatientType(Enum):
    """
    Enum representing different patient age groups for diabetes management.

    These categories are commonly used in clinical practice and research
    to differentiate treatment parameters and glucose targets.
    """

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

    BASIC = "basic"  # Basic 4-category standard
    CLINICAL = "clinical"  # Clinical 5-category standard (with very_low)


class TIRConfig:
    """Configuration for Time in Range standards."""

    # Define threshold boundaries for each standard (class-level constants)
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
            TIRCategory.TARGET: 70,  # 70 -180
            TIRCategory.LOW: 54,  # 54 -70
            TIRCategory.VERY_LOW: 0,  # 0 - 54
        },
    }

    # Define colors for visualization (class-level constants)
    COLORS = {
        TIRCategory.VERY_HIGH: "#FF6B35",
        TIRCategory.HIGH: "#FFB347",
        TIRCategory.TARGET: "#00B050",
        TIRCategory.LOW: "#DB2020",
        TIRCategory.VERY_LOW: "#8B0000",
    }

    # Define category order (bottom to top for stacked bar) (class-level constants)
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
    # Acceptable ranges from paper (mean±SD) as percentages (0-100)
    # children aged 7-15, adult aged 16-70
    # | Patient Group | Extreme High (>250) level 2 hyperglycemia | High (180-250) level 1 hyperglycemia | Target (70-180) (SD) | Low (<70) hypoglycemia |
    # | ------------- | ----------------------------------------- | ------------------------------------ | -------------------- | ---------------------- |
    # | Children      | 9.3±6.0                                   | 21.1±6.8                             | 67.5±11.5            | 2.1±1.5                |
    # | Adult         | 5.6±4.9                                   | 18.2±8.4                             | 74.5±11.9            | 1.6±2.1                |

    ACCEPTABLE_RANGES = {
        PatientType.CHILD: {
            TIRCategory.VERY_HIGH: (9.3 - 6.0, 9.3 + 6.0),  # mean±SD
            TIRCategory.HIGH: (21.1 - 6.8, 21.1 + 6.8),  # mean±SD
            TIRCategory.TARGET: (67.5 - 11.5, 67.5 + 11.5),  # mean±SD
            TIRCategory.LOW: (2.1 - 1.5, 2.1 + 1.5),  # mean±SD
        },
        PatientType.ADULT: {
            TIRCategory.VERY_HIGH: (5.6 - 4.9, 5.6 + 4.9),  # mean±SD
            TIRCategory.HIGH: (18.2 - 8.4, 18.2 + 8.4),  # mean±SD
            TIRCategory.TARGET: (74.5 - 11.9, 74.5 + 11.5),  # mean±SD
            TIRCategory.LOW: (1.6 - 2.1, 1.6 + 2.1),  # mean±SD
        },
    }

    def __init__(self, standard: TIRStandard = TIRStandard.BASIC):
        """
        Initialize TIRConfig with a specific standard.

        Args:
            standard: The TIR standard to use (defaults to BASIC)
        """
        self.standard = standard

    def get_thresholds(self):
        """Get thresholds for this instance's standard."""
        return self.THRESHOLDS[self.standard]

    @staticmethod
    def get_color(category: TIRCategory):
        """Get color for a given category (static, same for all standards)."""
        return TIRConfig.COLORS[category]

    def get_order(self):
        """Get category order for this instance's standard."""
        return self.ORDER[self.standard]

    @staticmethod
    def get_acceptable_ranges(patient_group: PatientType):
        """Get acceptable ranges for a given patient group (static, only for BASIC standard)."""
        return TIRConfig.ACCEPTABLE_RANGES.get(
            patient_group, TIRConfig.ACCEPTABLE_RANGES[PatientType.ADULT]
        )

    def calculate_time_in_range(self, BG_values) -> dict:
        """
        Calculate time in range statistics for blood glucose values.
        Uses this instance's standard.

        Args:
            BG_values: List of blood glucose readings in mg/dL

        Returns:
            Dictionary with percentages (0-100) for each range category
        """
        thresholds = self.get_thresholds()

        if self.standard == TIRStandard.BASIC:
            time_in_range = {
                TIRCategory.VERY_HIGH: 0,
                TIRCategory.HIGH: 0,
                TIRCategory.TARGET: 0,
                TIRCategory.LOW: 0,
            }

            for bg in BG_values:
                if bg > thresholds[TIRCategory.VERY_HIGH]:
                    time_in_range[TIRCategory.VERY_HIGH] += 1
                elif bg > thresholds[TIRCategory.HIGH]:
                    time_in_range[TIRCategory.HIGH] += 1
                elif bg > thresholds[TIRCategory.TARGET]:
                    time_in_range[TIRCategory.TARGET] += 1
                else:
                    time_in_range[TIRCategory.LOW] += 1

        else:  # TIRStandard.CLINICAL
            time_in_range = {
                TIRCategory.VERY_HIGH: 0,
                TIRCategory.HIGH: 0,
                TIRCategory.TARGET: 0,
                TIRCategory.LOW: 0,
                TIRCategory.VERY_LOW: 0,
            }

            for bg in BG_values:
                if bg > thresholds[TIRCategory.VERY_HIGH]:
                    time_in_range[TIRCategory.VERY_HIGH] += 1
                elif bg > thresholds[TIRCategory.HIGH]:
                    time_in_range[TIRCategory.HIGH] += 1
                elif bg > thresholds[TIRCategory.TARGET]:
                    time_in_range[TIRCategory.TARGET] += 1
                elif bg > thresholds[TIRCategory.LOW]:
                    time_in_range[TIRCategory.LOW] += 1
                else:
                    time_in_range[TIRCategory.VERY_LOW] += 1

        # Convert to percentages (0-100), only include non-zero values
        total = len(BG_values)
        time_in_range = {
            k: (v / total) * 100 for k, v in time_in_range.items() if v > 0
        }

        return time_in_range

    def get_time_in_range_acceptance(
        self,
        time_in_range,
        patient_group: PatientType,
    ) -> tuple:
        """
        Check if time in range values are within acceptable clinical ranges.
        This instance must use BASIC standard.

        Args:
            time_in_range: Dict with time in range statistics (as percentages 0-100)
            patient_group: PatientType enum

        Returns:
            Tuple of (category_results, acceptable_count) where:
            - category_results: Dict[TIRCategory, bool] indicating if each category is acceptable
            - acceptable_count: int count of categories within acceptable ranges

        Raises:
            ValueError: If this instance's standard is not TIRStandard.BASIC
        """
        if self.standard != TIRStandard.BASIC:
            raise ValueError(
                f"is_time_in_range_acceptable only supports TIRStandard.BASIC. "
                f"Current standard: {self.standard}"
            )

        # Get acceptable ranges for the patient group
        acceptable_ranges = self.get_acceptable_ranges(patient_group)

        # Check each category and track results
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
    """
    Apply common styling: remove grid, top/right borders.

    Args:
        ax: Matplotlib axis
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_bg(ax, t, BG, target_BG=None, show_legend=False):
    """
    Plot blood glucose data on the given axis.

    Args:
        ax: Matplotlib axis
        t: Time array
        BG: Blood glucose array (in mg/dL, will be converted to mmol/L)
        target_BG: Target blood glucose value (optional, not used)
        show_legend: Whether to show legend on this axis (default False)
    """
    # Convert mg/dL to mmol/L (divide by 18)
    BG_mmol = [bg / 18.0 for bg in BG]

    ax.plot(t, BG_mmol, color="black", linewidth=2, label="Glucose")

    # Set axis limits (converted to mmol/L)
    ax.set_xlim(0, max(t))
    ax.set_ylim(3.9, 13.9)
    ax.set_yticks([3.9, 6, 8, 10, 12, 13.9])

    _style_axis(ax, ylabel="Glucose (mmol/L)")


def _plot_cho(ax, t, CHO):
    """
    Plot carbohydrate intake data on the given axis.

    Args:
        ax: Matplotlib axis
        t: Time array
        CHO: Carbohydrate array
    """
    ax.plot(t, CHO)
    _style_axis(ax, ylabel="CHO (g)")


def _plot_insulin(ax, t, insulin):
    """
    Plot insulin delivery data on the given axis.

    Args:
        ax: Matplotlib axis
        t: Time array
        insulin: Insulin array
    """
    ax.plot(t, insulin)
    _style_axis(ax, xlabel="Time (min)", ylabel="Insulin (U/min)")


def _plot_cho_iob(ax, t, CHO, IOB, show_legend=False):
    """
    Plot IOB and CHO data on the given axis with CHO on secondary y-axis.

    Args:
        ax: Matplotlib axis (primary, for IOB)
        t: Time array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
        show_legend: Whether to show legend on this axis (default False)

    Returns:
        tuple: (lines, labels) for legend creation
    """
    # Plot IOB on primary y-axis
    iob_color = "#2596BE"
    line1 = ax.plot(t, IOB, color=iob_color, linestyle="-", linewidth=2, label="IOB")
    ax.set_ylabel("IOB (U)", color=iob_color)
    ax.tick_params(axis="y", labelcolor=iob_color)

    # Style primary axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(t))

    # Plot CHO on secondary y-axis
    cho_color = "#E96929"
    ax2 = ax.twinx()
    line2 = ax2.plot(t, CHO, color=cho_color, linestyle="-", linewidth=2, label="CHO")
    ax2.set_ylabel("CHO (g)", color=cho_color)
    ax2.tick_params(axis="y", labelcolor=cho_color)

    # Style secondary axis
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    return lines, labels


def _plot_bg_cho_insulin(fig, ax, t, BG, CHO, insulin, target_BG, fig_title):
    """
    Internal function to create the main BG/CHO/insulin plots.

    Args:
        fig: Matplotlib figure
        ax: List of 3 matplotlib axes
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)
    """
    _plot_bg(ax[0], t, BG, target_BG)
    _plot_cho(ax[1], t, CHO)
    _plot_insulin(ax[2], t, insulin)

    # Collect all legend handles and labels
    handles, labels = ax[0].get_legend_handles_labels()

    # Add legend at the bottom
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)


def _plot_bg_cho_iob(fig, ax, t, BG, CHO, IOB, target_BG, fig_title):
    """
    Internal function to create BG and IOB plots.

    Args:
        fig: Matplotlib figure
        ax: List of 2 matplotlib axes
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)
    """
    _plot_bg(ax[0], t, BG, target_BG)
    cho_iob_lines, cho_iob_labels = _plot_cho_iob(ax[1], t, CHO, IOB)

    # Add glucose legend below glucose subplot
    bg_handles, bg_labels = ax[0].get_legend_handles_labels()
    ax[0].legend(
        bg_handles,
        bg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )

    # Add IOB/CHO legend below its subplot
    ax[1].legend(
        cho_iob_lines,
        cho_iob_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    # Add x label below legend
    ax[1].set_xlabel("Time (min)")
    ax[1].xaxis.set_label_coords(0.5, -0.35)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, bottom=0.15)


def _plot_time_in_range_scale(scale_ax, time_in_range, tir_config: TIRConfig):
    """
    Plot time in range scale bar on the given axis.

    Args:
        scale_ax: Matplotlib axis for the scale bar
        time_in_range: Dict with time in range statistics (using TIRCategory enum as keys)
        tir_config: TIRConfig instance to use for getting thresholds and order
    """
    # Layout parameters (tune these values as needed)
    BAR_WIDTH = 6.0
    BAR_X_POSITION = 1.5
    THRESHOLD_LABEL_OFFSET = 0.1
    TITLE_OFFSET = 0.8
    X_LIMIT_LEFT = -1.0
    X_LIMIT_RIGHT = 4.0
    LABEL_FONTSIZE = 12
    THRESHOLD_FONTSIZE = 12
    TITLE_FONTSIZE = 14

    # Remove all spines and ticks
    for spine in scale_ax.spines.values():
        spine.set_visible(False)
    scale_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Get colors and order from TIRConfig instance
    order = tir_config.get_order()
    thresholds = tir_config.get_thresholds()

    # Create vertical stacked bar
    bottom = 0

    # Draw each segment in specified order
    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]  # Already in percentage (0-100)
            if percentage > 0:  # Only draw if percentage > 0
                color = tir_config.get_color(category)
                scale_ax.bar(
                    BAR_X_POSITION,
                    percentage,
                    bottom=bottom,
                    width=BAR_WIDTH,
                    color=color,
                    edgecolor="white",
                    linewidth=1,
                )

                # Add text label (centered inside bar segment)
                scale_ax.text(
                    BAR_X_POSITION,
                    bottom + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=LABEL_FONTSIZE,
                )

                bottom += percentage

    # Calculate cumulative heights for positioning threshold labels
    cumulative_height = 0
    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]  # Already in percentage (0-100)
            if percentage > 0:
                # Add threshold label at the bottom of this section (convert mg/dL to mmol/L)
                if category != TIRCategory.VERY_LOW:  # Don't show 0 at the bottom
                    threshold_mmol = thresholds[category] / 18.0
                    scale_ax.text(
                        BAR_X_POSITION - BAR_WIDTH / 2 - THRESHOLD_LABEL_OFFSET,
                        cumulative_height,
                        f"{threshold_mmol:.1f}",
                        ha="right",
                        va="center",
                        fontsize=THRESHOLD_FONTSIZE,
                        color="black",
                    )
                cumulative_height += percentage

                # Add top edge threshold for this section
                next_key_index = order.index(category) + 1
                if next_key_index < len(order):
                    next_category = order[next_key_index]
                    # Always show the threshold, regardless of whether next section exists
                    threshold_mmol = thresholds[next_category] / 18.0
                    scale_ax.text(
                        BAR_X_POSITION - BAR_WIDTH / 2 - THRESHOLD_LABEL_OFFSET,
                        cumulative_height,
                        f"{threshold_mmol:.1f}",
                        ha="right",
                        va="center",
                        fontsize=THRESHOLD_FONTSIZE,
                        color="black",
                    )

    # Add rotated "Percentage time in range" text on the right side
    scale_ax.text(
        BAR_X_POSITION + BAR_WIDTH / 2 + TITLE_OFFSET,
        50,
        "Percentage time in range",
        ha="left",
        va="center",
        fontsize=TITLE_FONTSIZE,
        color="black",
        rotation=90,
    )

    # Set axis limits for vertical bar
    scale_ax.set_xlim(X_LIMIT_LEFT, X_LIMIT_RIGHT)
    scale_ax.set_ylim(0, 100)


def _create_bg_cho_insulin_figure(t, BG, CHO, insulin, target_BG, fig_title):
    """
    Create a basic plot figure without time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Glucose subplot ratio 1.2:1 (width:height), CHO and insulin are 0.5 height ratio
    plot_height = 5
    plot_width = plot_height * 1.2  # 1.2:1 ratio (wider than tall)
    other_height = plot_height * 0.5
    fig_height = plot_height + other_height * 2 + 2  # extra for labels/legend

    fig = plt.figure(figsize=(plot_width + 1.5, fig_height))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    axes = [ax0, ax1, ax2]
    _plot_bg_cho_insulin(fig, axes, t, BG, CHO, insulin, target_BG, fig_title)

    return fig


def _create_bg_cho_insulin_figure_with_tir(
    t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config
):
    """
    Create a plot figure with time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Glucose subplot ratio 1.2:1 (width:height), CHO and insulin are 0.5 height ratio
    plot_height = 5
    plot_width = plot_height * 1.2  # 1.2:1 ratio (wider than tall)
    other_height = plot_height * 0.5
    fig_height = plot_height + other_height * 2 + 2  # extra for labels/legend
    tir_width = 1

    fig = plt.figure(figsize=(plot_width + tir_width + 2, fig_height))
    gs = gridspec.GridSpec(
        3,
        2,
        width_ratios=[plot_width, tir_width],
        height_ratios=[1, 0.5, 0.5],
        wspace=0.2,
        hspace=0.3,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    scale_ax = fig.add_subplot(gs[0, 1])  # TIR only next to BG

    axes = [ax0, ax1, ax2]
    _plot_bg_cho_insulin(fig, axes, t, BG, CHO, insulin, target_BG, fig_title)

    # Add time in range scale bar
    _plot_time_in_range_scale(scale_ax, time_in_range, tir_config)

    return fig


def _create_bg_cho_iob_figure(t, BG, CHO, IOB, target_BG, fig_title):
    """
    Create a basic BG/IOB plot figure without time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Glucose subplot ratio 1.2:1 (width:height), IOB is 0.5 height ratio
    plot_height = 5
    plot_width = plot_height * 1.2  # 1.2:1 ratio (wider than tall)
    iob_height = plot_height * 0.5
    fig_height = plot_height + iob_height + 1.5  # extra for labels/legend

    fig = plt.figure(figsize=(plot_width + 1.5, fig_height))  # extra width for y-labels
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    axes = [ax0, ax1]
    _plot_bg_cho_iob(fig, axes, t, BG, CHO, IOB, target_BG, fig_title)

    return fig


def _create_bg_cho_iob_figure_with_tir(
    t, BG, CHO, IOB, target_BG, fig_title, time_in_range, tir_config
):
    """
    Create a BG/IOB plot figure with time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value (not used)
        fig_title: Title for the figure (not used)
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Glucose subplot ratio 1.2:1 (width:height), IOB is 0.5 height ratio
    plot_height = 5
    plot_width = plot_height * 1.2  # 1.2:1 ratio (wider than tall)
    iob_height = plot_height * 0.5
    fig_height = plot_height + iob_height + 1.5  # extra for labels/legend
    tir_width = 0.7  # width for TIR bar

    fig = plt.figure(figsize=(plot_width + tir_width + 2, fig_height))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[plot_width, tir_width],
        height_ratios=[1, 0.5],
        wspace=0.15,
        hspace=0.3,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    scale_ax = fig.add_subplot(gs[0, 1])  # TIR only next to BG

    axes = [ax0, ax1]
    _plot_bg_cho_iob(fig, axes, t, BG, CHO, IOB, target_BG, fig_title)

    # Add time in range scale bar
    _plot_time_in_range_scale(scale_ax, time_in_range, tir_config)

    return fig


def plot_and_show(t, BG, CHO, insulin, target_BG, fig_title):
    """
    Display plot without time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
    """
    _create_bg_cho_insulin_figure(t, BG, CHO, insulin, target_BG, fig_title)
    plt.show()


def plot_and_show_with_tir(
    t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config
):
    """
    Display plot with time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance
    """
    _create_bg_cho_insulin_figure_with_tir(
        t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config
    )
    plt.show()


def plot_and_save(t, BG, CHO, insulin, target_BG, file_name):
    """
    Save plot to file without time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        file_name: Path to save the figure
    """
    fig_title = Path(file_name).stem
    fig = _create_bg_cho_insulin_figure(t, BG, CHO, insulin, target_BG, fig_title)
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_and_save_with_tir(
    t, BG, CHO, insulin, target_BG, file_name, time_in_range, tir_config
):
    """
    Save plot to file with time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        file_name: Path to save the figure
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance
    """
    fig_title = Path(file_name).stem
    fig = _create_bg_cho_insulin_figure_with_tir(
        t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config
    )
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_bg_cho_iob_and_show(t, BG, CHO, IOB, target_BG, fig_title):
    """
    Display BG/IOB plot without time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
    """
    _create_bg_cho_iob_figure(t, BG, CHO, IOB, target_BG, fig_title)
    plt.show()


def plot_bg_cho_iob_and_show_with_tir(
    t, BG, CHO, IOB, target_BG, fig_title, time_in_range, tir_config
):
    """
    Display BG/IOB plot with time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance
    """
    fig = _create_bg_cho_iob_figure_with_tir(
        t, BG, CHO, IOB, target_BG, fig_title, time_in_range, tir_config
    )
    plt.show()


def plot_bg_cho_iob_and_save(t, BG, CHO, IOB, target_BG, file_name):
    """
    Save BG/IOB plot to file without time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value
        file_name: Path to save the figure
    """
    fig_title = Path(file_name).stem
    fig = _create_bg_cho_iob_figure(t, BG, CHO, IOB, target_BG, fig_title)
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_bg_cho_iob_and_save_with_tir(
    t, BG, CHO, IOB, target_BG, file_name, time_in_range, tir_config
):
    """
    Save BG/IOB plot to file with time in range scale bar.

    Args:
        t: Time array
        BG: Blood glucose array
        iob: Insulin on Board array
        target_BG: Target blood glucose value
        file_name: Path to save the figure
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance
    """
    fig_title = Path(file_name).stem
    fig = _create_bg_cho_iob_figure_with_tir(
        t, BG, CHO, IOB, target_BG, fig_title, time_in_range, tir_config
    )
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_bg_no_legend(ax, t, BG):
    """Plot blood glucose without legend."""
    BG_mmol = [bg / 18.0 for bg in BG]
    ax.plot(t, BG_mmol, color="black", linewidth=2, label="Glucose")
    ax.set_xlim(0, max(t))
    ax.set_ylim(3.9, 13.9)
    ax.set_yticks([3.9, 6, 8, 10, 12, 13.9])
    _style_axis(ax, ylabel="Glucose (mmol/L)")


def _plot_cho_iob_no_legend(ax, t, CHO, IOB):
    """Plot IOB and CHO without legend, return lines for shared legend."""
    iob_color = "#2596BE"
    line1 = ax.plot(t, IOB, color=iob_color, linestyle="-", linewidth=2, label="IOB")
    ax.set_ylabel("IOB (U)", color=iob_color)
    ax.tick_params(axis="y", labelcolor=iob_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(t))

    cho_color = "#E96929"
    ax2 = ax.twinx()
    line2 = ax2.plot(t, CHO, color=cho_color, linestyle="-", linewidth=2, label="CHO")
    ax2.set_ylabel("CHO (g)", color=cho_color)
    ax2.tick_params(axis="y", labelcolor=cho_color)
    ax2.set_yticks([0, 25, 50, 75])
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    return line1 + line2


def _create_comparison_figure_with_tir(
    data_left, data_right, tir_config, title_left="", title_right=""
):
    """
    Create a side-by-side comparison figure with shared legend in the middle.

    Args:
        data_left: Dict with keys 't', 'BG', 'CHO', 'IOB', 'time_in_range'
        data_right: Dict with keys 't', 'BG', 'CHO', 'IOB', 'time_in_range'
        tir_config: TIRConfig instance
        title_left: Title for left plot
        title_right: Title for right plot

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot_height = 5
    plot_width = plot_height * 1.2
    iob_height = plot_height * 0.5
    tir_width = 0.7
    legend_height = 0.25

    # Layout rows: [BG plots, BG legend, IOB plots, IOB legend]
    # Layout cols: [left_plot, left_tir, gap, right_plot, right_tir]
    fig_height = plot_height + iob_height + legend_height * 2
    gap_width = 0.6
    total_width = plot_width * 2 + tir_width * 2 + gap_width + 2

    fig = plt.figure(figsize=(total_width, fig_height))
    gs = gridspec.GridSpec(
        4,
        5,
        width_ratios=[plot_width, tir_width, gap_width, plot_width, tir_width],
        height_ratios=[1, legend_height, 0.5, legend_height],
        wspace=0.15,
        hspace=0.05,
    )

    # Left plot axes
    ax_bg_left = fig.add_subplot(gs[0, 0])
    ax_iob_left = fig.add_subplot(gs[2, 0], sharex=ax_bg_left)
    ax_tir_left = fig.add_subplot(gs[0, 1])

    # Right plot axes (col 2 is the gap)
    ax_bg_right = fig.add_subplot(gs[0, 3])
    ax_iob_right = fig.add_subplot(gs[2, 3], sharex=ax_bg_right)
    ax_tir_right = fig.add_subplot(gs[0, 4])

    # Legend axes (spanning middle columns)
    ax_legend_bg = fig.add_subplot(gs[1, :])
    ax_legend_bg.axis("off")
    ax_legend_iob = fig.add_subplot(gs[3, :])
    ax_legend_iob.axis("off")

    # Plot left data
    _plot_bg_no_legend(ax_bg_left, data_left["t"], data_left["BG"])
    lines_left = _plot_cho_iob_no_legend(
        ax_iob_left, data_left["t"], data_left["CHO"], data_left["IOB"]
    )
    _plot_time_in_range_scale(ax_tir_left, data_left["time_in_range"], tir_config)

    # Plot right data
    _plot_bg_no_legend(ax_bg_right, data_right["t"], data_right["BG"])
    lines_right = _plot_cho_iob_no_legend(
        ax_iob_right, data_right["t"], data_right["CHO"], data_right["IOB"]
    )
    _plot_time_in_range_scale(ax_tir_right, data_right["time_in_range"], tir_config)

    # Add x labels
    ax_iob_left.set_xlabel("Time (min)")
    ax_iob_right.set_xlabel("Time (min)")

    # Add subplot labels (a) and (b)
    ax_bg_left.text(
        -0.15,
        1.05,
        "a",
        transform=ax_bg_left.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_bg_right.text(
        -0.15,
        1.05,
        "b",
        transform=ax_bg_right.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Glucose legend below glucose plots with TIR color indicators
    bg_handle = ax_bg_left.get_legend_handles_labels()[0]
    tir_patches = [
        Patch(
            facecolor=tir_config.get_color(TIRCategory.HIGH),
            label="Hyperglycemia (TAR)",
        ),
        Patch(facecolor=tir_config.get_color(TIRCategory.TARGET), label="TIR"),
    ]
    ax_legend_bg.legend(
        bg_handle + tir_patches,
        ["Glucose", "Hyperglycemia (TAR)", "TIR"],
        loc="center",
        frameon=False,
        fontsize=14,
        ncol=3,
    )

    # IOB/CHO legend below IOB plots
    ax_legend_iob.legend(
        lines_left,
        ["IOB", "CHO"],
        loc="center",
        frameon=False,
        fontsize=14,
        ncol=2,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    return fig


def plot_comparison_and_save_with_tir(
    data_left, data_right, file_name, tir_config, title_left="", title_right=""
):
    """
    Save a side-by-side comparison plot with shared legend.

    Args:
        data_left: Dict with keys 't', 'BG', 'CHO', 'IOB', 'time_in_range'
        data_right: Dict with keys 't', 'BG', 'CHO', 'IOB', 'time_in_range'
        file_name: Path to save the figure
        tir_config: TIRConfig instance
        title_left: Title for left plot
        title_right: Title for right plot
    """
    fig = _create_comparison_figure_with_tir(
        data_left, data_right, tir_config, title_left, title_right
    )
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
