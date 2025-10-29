"""
Plotting utilities for glucose control analytics.

This module provides functions to visualize blood glucose, carbohydrate intake,
and insulin delivery data, with optional time-in-range visualizations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .time_in_range import TIRCategory, TIRConfig


def _plot(fig, ax, t, BG, CHO, insulin, target_BG, fig_title):
    """
    Internal function to create the main BG/CHO/insulin plots.

    Args:
        fig: Matplotlib figure
        ax: List of 3 matplotlib axes
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
    """
    ax[0].plot(t, BG)
    ax[0].plot(t, [target_BG] * len(t), "r--", label="Target BG")
    ax[0].plot(t, [70] * len(t), "b--", label="Hypoglycemia")
    ax[0].plot(t, [180] * len(t), "k--", label="Hyperglycemia")
    ax[0].grid()
    ax[0].set_ylabel("BG (mg/dL)")
    ax[1].plot(t, CHO)
    ax[1].grid()
    ax[1].set_ylabel("CHO (g)")
    ax[2].plot(t, insulin)
    ax[2].grid()
    ax[2].set_ylabel("Insulin (U/min)")
    ax[2].set_xlabel("Time (min)")
    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc="lower center", ncol=3)
    fig.suptitle(f"{fig_title}")
    fig.tight_layout()


def _plot_time_in_range_scale(scale_ax, time_in_range, tir_config: TIRConfig):
    """
    Plot time in range scale bar on the given axis.

    Args:
        scale_ax: Matplotlib axis for the scale bar
        time_in_range: Dict with time in range statistics (using TIRCategory enum as keys)
        tir_config: TIRConfig instance to use for getting thresholds and order
    """
    scale_ax.axis("off")

    # Get colors and order from TIRConfig instance
    order = tir_config.get_order()
    thresholds = tir_config.get_thresholds()

    # Create vertical stacked bar
    x_position = 0
    bar_width = 0.8
    bottom = 0

    # Draw each segment in specified order
    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]  # Already in percentage (0-100)
            if percentage > 0:  # Only draw if percentage > 0
                color = tir_config.get_color(category)
                scale_ax.bar(
                    x_position,
                    percentage,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=1,
                )

                # Add text label (centered inside bar segment)
                scale_ax.text(
                    x_position,
                    bottom + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )

                bottom += percentage

    # Calculate cumulative heights for positioning threshold labels
    cumulative_height = 0
    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]  # Already in percentage (0-100)
            if percentage > 0:
                # Add threshold label at the bottom of this section
                if category != TIRCategory.VERY_LOW:  # Don't show 0 at the bottom
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[category]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
                cumulative_height += percentage

                # Add top edge threshold for this section
                next_key_index = order.index(category) + 1
                if next_key_index < len(order):
                    next_category = order[next_key_index]
                    # Always show the threshold, regardless of whether next section exists
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[next_category]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    # Set axis limits for vertical bar
    scale_ax.set_xlim(-0.5, 0.5)
    scale_ax.set_ylim(0, 100)
    scale_ax.set_title("Time in Range", fontsize=12)


def _create_plot_figure(t, BG, CHO, insulin, target_BG, fig_title):
    """
    Create a basic plot figure without time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        fig_title: Title for the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(3, 1, 1)
    ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)
    ax2 = fig.add_subplot(3, 1, 3, sharex=ax0)

    axes = [ax0, ax1, ax2]
    _plot(fig, axes, t, BG, CHO, insulin, target_BG, fig_title)

    return fig


def _create_plot_figure_with_tir(t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config):
    """
    Create a plot figure with time in range scale.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        target_BG: Target blood glucose value
        fig_title: Title for the figure
        time_in_range: Dict with time in range statistics
        tir_config: TIRConfig instance

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure with gridspec layout for TIR scale
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(
        3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1], wspace=0.05
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    scale_ax = fig.add_subplot(gs[:, 1])

    axes = [ax0, ax1, ax2]
    _plot(fig, axes, t, BG, CHO, insulin, target_BG, fig_title)

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
    fig = _create_plot_figure(t, BG, CHO, insulin, target_BG, fig_title)
    plt.show()


def plot_and_show_with_tir(t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config):
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
    fig = _create_plot_figure_with_tir(t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config)
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
    fig = _create_plot_figure(t, BG, CHO, insulin, target_BG, fig_title)
    fig.savefig(f"{file_name}")
    plt.close(fig)


def plot_and_save_with_tir(t, BG, CHO, insulin, target_BG, file_name, time_in_range, tir_config):
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
    fig = _create_plot_figure_with_tir(t, BG, CHO, insulin, target_BG, fig_title, time_in_range, tir_config)
    fig.savefig(f"{file_name}")
    plt.close(fig)
