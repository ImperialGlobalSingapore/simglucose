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
mpl.rcParams["font.size"] = 15
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


def _plot_bg(ax, t, CGM, BG):
    """
    Plot CGM reading and patient glucose data on the given axis.

    Args:
        ax: Matplotlib axis
        t: Time array
        CGM: CGM reading array (in mg/dL, will be converted to mmol/L)
        BG: Patient glucose array (in mg/dL, will be converted to mmol/L)
    """
    # Convert mg/dL to mmol/L (divide by 18)
    CGM_mmol = [cgm / 18.0 for cgm in CGM]
    BG_mmol = [bg / 18.0 for bg in BG]

    ax.plot(t, CGM_mmol, color="#FF69B4", linewidth=3, label="CGM reading")
    ax.plot(t, BG_mmol, color="black", linewidth=3, label="Virtual Participant Glucose")

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
    line1 = ax.plot(t, IOB, color=iob_color, linestyle="-", linewidth=3, label="IOB")
    ax.set_ylabel("IOB (U)", color=iob_color)
    ax.tick_params(axis="y", labelcolor=iob_color)

    # Style primary axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(t))

    # Plot CHO on secondary y-axis
    cho_color = "#E96929"
    ax2 = ax.twinx()
    line2 = ax2.plot(t, CHO, color=cho_color, linestyle="-", linewidth=3, label="CHO")
    ax2.set_ylabel("CHO (g)", color=cho_color)
    ax2.tick_params(axis="y", labelcolor=cho_color)

    # Style secondary axis
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    return lines, labels


def _plot_bg_cho_iob(fig, ax, t, CGM, BG, CHO, IOB, add_labels=False):
    """
    Internal function to create BG and IOB plots.

    Args:
        fig: Matplotlib figure
        ax: List of 2 matplotlib axes
        t: Time array
        CGM: CGM reading array
        BG: Patient glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
        add_labels: Whether to add subplot labels (a, b)
    """
    _plot_bg(ax[0], t, CGM, BG)
    cho_iob_lines, cho_iob_labels = _plot_cho_iob(ax[1], t, CHO, IOB)

    # Add subplot labels if requested
    if add_labels:
        ax[0].text(
            -0.12,
            1.05,
            "a",
            transform=ax[0].transAxes,
            fontsize=21,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax[1].text(
            -0.12,
            1.05,
            "b",
            transform=ax[1].transAxes,
            fontsize=21,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

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


def _create_bg_cho_iob_figure(t, CGM, BG, CHO, IOB, add_labels=False):
    """
    Create a BG/CHO/IOB plot figure.

    Args:
        t: Time array
        CGM: CGM reading array
        BG: Patient glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
        add_labels: Whether to add subplot labels (a, b)

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
    _plot_bg_cho_iob(fig, axes, t, CGM, BG, CHO, IOB, add_labels=add_labels)

    return fig


def plot_bg_cho_iob_and_show(t, CGM, BG, CHO, IOB):
    """
    Display BG/CHO/IOB plot.

    Args:
        t: Time array
        CGM: CGM reading array
        BG: Patient glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
    """
    _create_bg_cho_iob_figure(t, CGM, BG, CHO, IOB)
    plt.show()


def plot_bg_cho_iob_and_save(t, CGM, BG, CHO, IOB, file_name, add_labels=False):
    """
    Save BG/CHO/IOB plot to file.

    Args:
        t: Time array
        CGM: CGM reading array
        BG: Patient glucose array
        CHO: Carbohydrate array
        IOB: Insulin on Board array
        file_name: Path to save the figure
        add_labels: Whether to add subplot labels (a, b)
    """
    fig = _create_bg_cho_iob_figure(t, CGM, BG, CHO, IOB, add_labels=add_labels)
    file_path = Path(file_name)
    fig.savefig(file_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(file_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)


def _plot_bg_merged(ax, t, CGM_no_attack, t_attack, CGM_attack, BG):
    """
    Plot merged glucose data: CGM (no attack), CGM (attack), and patient glucose.

    Args:
        ax: Matplotlib axis
        t: Time array for no-attack data
        CGM_no_attack: CGM reading from no-attack scenario (dark blue)
        t_attack: Time array for attack data
        CGM_attack: CGM reading from attack scenario (pink)
        BG: Patient glucose (black)
    """
    CGM_no_attack_mmol = [cgm / 18.0 for cgm in CGM_no_attack]
    CGM_attack_mmol = [cgm / 18.0 for cgm in CGM_attack]
    BG_mmol = [bg / 18.0 for bg in BG]

    ax.plot(
        t,
        CGM_no_attack_mmol,
        color="#1E90FF",
        linewidth=3,
        label="CGM Reading",
    )
    ax.plot(
        t_attack,
        CGM_attack_mmol,
        color="#FF69B4",
        linewidth=3,
        label="CGM Reading (Attack)",
    )
    ax.plot(t, BG_mmol, color="black", linewidth=3, label="Virtual Participant Glucose")

    max_t = max(max(t), max(t_attack))
    ax.set_xlim(0, max_t)
    ax.set_ylim(3.9, 20)
    ax.set_yticks([4, 6, 8, 10, 12, 14, 16, 18, 20])
    _style_axis(ax, ylabel="Glucose (mmol/L)")


def _plot_cho_iob_no_legend(ax, t, CHO, IOB):
    """Plot IOB and CHO without legend, return lines for shared legend."""
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

    return line1 + line2


def _plot_cho_iob_merged(ax, t, CHO, IOB, t_attack, IOB_attack):
    """
    Plot IOB (no attack), IOB (attack), and CHO on the given axis.

    Args:
        ax: Matplotlib axis (primary, for IOB)
        t: Time array for no-attack data
        CHO: Carbohydrate array
        IOB: Insulin on Board array (no attack)
        t_attack: Time array for attack data
        IOB_attack: Insulin on Board array (attack)

    Returns:
        tuple: (lines, labels) for legend creation
    """
    iob_color = "#2596BE"
    iob_attack_color = "#FF69B4"

    line1 = ax.plot(t, IOB, color=iob_color, linestyle="-", linewidth=3, label="IOB")
    line2 = ax.plot(
        t_attack,
        IOB_attack,
        color=iob_attack_color,
        linestyle="-",
        linewidth=3,
        label="IOB (Attack)",
    )
    ax.set_ylabel("IOB (U)", color=iob_color)
    ax.tick_params(axis="y", labelcolor=iob_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_t = max(max(t), max(t_attack))
    ax.set_xlim(0, max_t)

    cho_color = "#E96929"
    ax2 = ax.twinx()
    line3 = ax2.plot(t, CHO, color=cho_color, linestyle="-", linewidth=3, label="CHO")
    ax2.set_ylabel("CHO (g)", color=cho_color)
    ax2.tick_params(axis="y", labelcolor=cho_color)

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    return lines, labels


def _create_merged_figure(data_no_attack, data_attack, attack_times=None):
    """
    Create a merged plot figure with attack vs no-attack CGM readings.

    Uses data from no-attack for: CGM (no attack), Patient glucose, IOB, CHO
    Uses data from attack for: CGM (attack)

    Args:
        data_no_attack: Dict with keys 't', 'CGM', 'BG', 'CHO', 'IOB'
        data_attack: Dict with keys 't', 'CGM', 'BG', 'CHO', 'IOB'

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot_height = 5
    plot_width = plot_height * 1.2
    iob_height = plot_height * 0.5
    fig_height = plot_height + iob_height + 1.5

    fig = plt.figure(figsize=(plot_width + 1.5, fig_height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.4)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # Plot merged glucose data
    _plot_bg_merged(
        ax0,
        data_no_attack["t"],
        data_no_attack["CGM"],
        data_attack["t"],
        data_attack["CGM"],
        data_no_attack["BG"],
    )

    # Plot IOB/CHO from both datasets
    cho_iob_lines, cho_iob_labels = _plot_cho_iob_merged(
        ax1,
        data_no_attack["t"],
        data_no_attack["CHO"],
        data_no_attack["IOB"],
        data_attack["t"],
        data_attack["IOB"],
    )

    # Annotate attack episode
    if attack_times is not None:
        attack_start, attack_end = attack_times
        for ax in [ax0, ax1]:
            ax.axvspan(attack_start, attack_end, facecolor="lightgrey", alpha=0.3, zorder=0)

        arrow_y = 17.5
        ax0.annotate(
            "", xy=(attack_end, arrow_y), xytext=(attack_start, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="black", lw=2),
        )
        ax0.text(
            (attack_start + attack_end) / 2, 18.0,
            "Attack Episode",
            ha="center", va="bottom", fontsize=15, fontweight="bold",
        )

    # Add subplot labels (a, b)
    ax0.text(
        -0.12,
        1.05,
        "a",
        transform=ax0.transAxes,
        fontsize=21,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax1.text(
        -0.12,
        1.05,
        "b",
        transform=ax1.transAxes,
        fontsize=21,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Add glucose legend below glucose subplot (two centered rows)
    bg_handles, bg_labels = ax0.get_legend_handles_labels()
    leg1 = ax0.legend(
        bg_handles[:2],
        bg_labels[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    ax0.add_artist(leg1)
    ax0.legend(
        bg_handles[2:],
        bg_labels[2:],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
    )

    # Add IOB/CHO legend below its subplot
    ax1.legend(
        cho_iob_lines,
        cho_iob_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )

    # Add x label below legend
    ax1.set_xlabel("Time (min)")
    ax1.xaxis.set_label_coords(0.5, -0.35)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, bottom=0.15)

    return fig


def plot_merged_and_save(data_no_attack, data_attack, file_name, attack_times=None):
    """
    Save a merged plot comparing attack vs no-attack CGM readings.

    Args:
        data_no_attack: Dict with keys 't', 'CGM', 'BG', 'CHO', 'IOB'
        data_attack: Dict with keys 't', 'CGM', 'BG', 'CHO', 'IOB'
        file_name: Path to save the figure
        attack_times: Tuple of (start, end) in minutes for the attack episode
    """
    fig = _create_merged_figure(data_no_attack, data_attack, attack_times)
    file_path = Path(file_name)
    fig.savefig(file_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(file_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
