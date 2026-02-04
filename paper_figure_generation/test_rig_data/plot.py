"""
Plotting utilities for glucose control analytics.

This module provides functions to visualize blood glucose, carbohydrate intake,
and insulin on board data.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

# Set Calibri font globally
mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["axes.labelweight"] = "bold"


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


def _plot_bg(ax, t, BG):
    """
    Plot blood glucose data on the given axis.

    Args:
        ax: Matplotlib axis
        t: Time array
        BG: Blood glucose array (in mg/dL, will be converted to mmol/L)
    """
    # Convert mg/dL to mmol/L (divide by 18)
    BG_mmol = [bg / 18.0 for bg in BG]

    ax.plot(t, BG_mmol, color="black", linewidth=2, label="Glucose")

    # Set axis limits (converted to mmol/L)
    ax.set_xlim(0, max(t))
    ax.set_ylim(3.9, 20)
    ax.set_yticks([4, 6, 8, 10, 12, 14, 16, 18, 20])

    _style_axis(ax, ylabel="Glucose (mmol/L)")


def _plot_cho_iob(ax, t, CHO, IOB):
    """
    Plot IOB and CHO data on the given axis with CHO on secondary y-axis.

    Args:
        ax: Matplotlib axis (primary, for IOB)
        t: Time array
        CHO: Carbohydrate array
        IOB: Insulin on Board array

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


def _plot_bg_cho_iob(fig, ax, t, BG, CHO, IOB):
    """
    Internal function to create BG and IOB plots.

    Args:
        fig: Matplotlib figure
        ax: List of 2 matplotlib axes
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
    """
    _plot_bg(ax[0], t, BG)
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


def _create_bg_cho_iob_figure(t, BG, CHO, IOB):
    """
    Create a BG/CHO/IOB plot figure.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Glucose subplot ratio 1.2:1 (width:height), IOB is 0.5 height ratio
    plot_height = 5
    plot_width = plot_height * 1.2  # 1.2:1 ratio (wider than tall)
    iob_height = plot_height * 0.5
    fig_height = plot_height + iob_height + 1.5  # extra for labels/legend

    fig = plt.figure(figsize=(plot_width + 1.5, fig_height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    axes = [ax0, ax1]
    _plot_bg_cho_iob(fig, axes, t, BG, CHO, IOB)

    return fig


def plot_bg_cho_iob_and_show(t, BG, CHO, IOB):
    """
    Display BG/CHO/IOB plot.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
    """
    _create_bg_cho_iob_figure(t, BG, CHO, IOB)
    plt.show()


def plot_bg_cho_iob_and_save(t, BG, CHO, IOB, file_name):
    """
    Save BG/CHO/IOB plot to file.

    Args:
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
        file_name: Path to save the figure
    """
    fig = _create_bg_cho_iob_figure(t, BG, CHO, IOB)
    fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
