from pathlib import Path
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import pandas as pd

from simglucose.patient.t1dpatient import PATIENT_PARA_FILE
from matplotlib import gridspec


class PatientType(Enum):
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    CHILD = "child"


class Scenario(Enum):
    NO_MEAL = "no_meal"
    SINGLE_MEAL = "single_meal"
    ONE_DAY = "one_day"
    THREE_DAY = "three_day"

    def get_carb(self, t, body_weight=None):
        """
        t: time in minutes
        body_weight: weight of the patient in kg
        Returns carbs only at exact meal times, 0 otherwise
        """
        # Force t to int for exact time matching
        t = int(t)

        carb_times_in_hour = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [6],  # Assuming a meal at 6:00
            Scenario.ONE_DAY: [7, 12, 19],  # Meals at 7:00, 12:00, 19:00
            Scenario.THREE_DAY: [
                h + 24 * d for d in range(3) for h in [7, 12, 19]
            ],  # Meals at 7:00, 12:00, 19:00 for 3 days
        }

        # Predefined carb amounts for each meal time
        carb_amounts = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [50],  # 50g of carbs for single meal
            Scenario.ONE_DAY: [40, 50, 70],  # 40g, 50g, 70g of carbs per meal
            Scenario.THREE_DAY: [
                40,
                50,
                70,
            ]
            * 3,  # 40g, 50g, 70g of carbs per meal for 3 days
        }
        if body_weight is not None:
            carb_amounts[Scenario.ONE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ]
            carb_amounts[Scenario.THREE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ] * 3

        # Convert hours to minutes for comparison
        carb_times_in_minutes = [h * 60 for h in carb_times_in_hour[self]]

        # Check if current time matches any meal time exactly
        if t in carb_times_in_minutes:
            idx = carb_times_in_minutes.index(t)
            return carb_amounts[self][idx]
        return 0


max_t = {
    Scenario.NO_MEAL: 1000,  # 16 hours + 40 minutes
    Scenario.SINGLE_MEAL: 1080,  # 18 hours
    Scenario.ONE_DAY: 1450,  # 24 hours + 10 minutes
    Scenario.THREE_DAY: 4330,  # 72 hours + 10 minutes
}


def _plot(fig, ax, t, BG, CHO, insulin, target_BG, fig_title):
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
    fig.suptitle(f"PID Controller: {fig_title}")
    fig.tight_layout()


def plot_and_show(t, BG, CHO, insulin, target_BG, fig_title):
    fig, ax = plt.subplots(3, sharex=True)
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, fig_title)
    plt.show()


def save_name_pattern(k_P, k_I, k_D, sample_time, basal_rate, remark=""):
    return f"{remark}pid_st{sample_time}_br{basal_rate}_p{k_P}_i{k_I}_d{k_D}"


def plot_and_save(t, BG, CHO, insulin, target_BG, file_name):
    fig_title = Path(file_name).stem
    fig, ax = plt.subplots(3, sharex=True, figsize=(15, 10))
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, fig_title)
    fig.savefig(f"{file_name}")
    plt.close(fig)


def plot_with_scale_and_save(t, BG, CHO, insulin, target_BG, time_in_range, file_name):
    fig_title = Path(file_name).stem
    fig, ax = plt.subplots(3, sharex=True, figsize=(15, 10))
    # Create a new grid with an extra column for the scale bar
    fig.clf()
    gs = gridspec.GridSpec(
        3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1], wspace=0.05
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    scale_ax = fig.add_subplot(gs[:, 1])

    # Plot the main data
    axes = [ax0, ax1, ax2]
    _plot(fig, axes, t, BG, CHO, insulin, target_BG, fig_title)

    # Configure scale bar with new color scheme and drawing logic
    scale_ax.axis("off")

    # Define colors for each range category
    colors = {
        "very_high": "#FF6B35",
        "high": "#FFB347",
        "target": "#32CE13",
        "low": "#DB2020",
        "very_low": "#8B0000",
    }

    # Create vertical stacked bar
    x_position = 0
    bar_width = 0.8
    bottom = 0

    # Define order (bottom to top)
    order = ["very_low", "low", "target", "high", "very_high"]

    # Draw each segment in specified order
    for key in order:
        if key in time_in_range:
            percentage = time_in_range[key] * 100  # Convert to percentage
            if (
                percentage > 0 and key in colors
            ):  # Only draw if percentage > 0 and color exists
                scale_ax.bar(
                    x_position,
                    percentage,
                    bottom=bottom,
                    width=bar_width,
                    color=colors[key],
                    edgecolor="white",
                    linewidth=1,
                )

                # Add text label (option 1: centered inside bar segment)
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

    # Add threshold labels at boundaries between sections
    thresholds = {
        "very_low": 0,  # Starting point
        "low": 54,  # Between very_low and low
        "target": 70,  # Between low and target
        "high": 180,  # Between target and high
        "very_high": 250,  # Between high and very_high
    }

    # Calculate cumulative heights for positioning
    cumulative_height = 0
    for key in order:
        if key in time_in_range:
            percentage = time_in_range[key] * 100
            if percentage > 0:
                # Add threshold label at the bottom of this section
                if key != "very_low":  # Don't show 0 at the bottom
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[key]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
                cumulative_height += percentage

                # Add top edge threshold for this section
                next_key_index = order.index(key) + 1
                if next_key_index < len(order):
                    next_key = order[next_key_index]
                    # Always show the threshold, regardless of whether next section exists
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[next_key]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    # Set axis limits for vertical bar
    scale_ax.set_xlim(-0.5, 0.5)
    scale_ax.set_ylim(0, 100)
    scale_ax.set_title("Time in Range", fontsize=12)

    fig.savefig(f"{file_name}")
    plt.close(fig)


def save_to_csv(log_dir, t, BG, CHO, insulin, file_name):
    csv_file = log_dir / f"{file_name}.csv"
    with open(csv_file, "w") as f:
        f.write("time,CHO,insulin,BG\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{CHO[i]},{insulin[i]},{BG[i]}\n")


def get_rmse(BG, target_BG):
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    return np.sqrt(np.mean((BG - target_BG) ** 2))


def eval_result(BG, target_BG, k_P, k_I, k_D, sample_time):
    print(f"k_P: {k_P}, k_I: {k_I}, k_D: {k_D}, sample_time: {sample_time}")
    target_BG = np.array(target_BG)
    BG = np.array(BG)
    errors = target_BG - BG
    max_e = np.abs(errors).max()
    mae = np.mean(np.abs(errors))  # mean absolute error
    mse = np.mean(errors**2)  # mean square error
    rmse = np.sqrt(mse)  # root mean square error
    iae = np.sum(np.abs(errors) * sample_time)  # integrated absolute error
    ise = np.sum(errors**2 * sample_time)  # integrated square error
    return {
        "MAX_E": max_e,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "IAE": iae,
        "ISE": ise,
    }


def get_patients():
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    return patient_params.Name.tolist()


def get_patient_by_group(patient_type: PatientType):
    if patient_type == PatientType.ADOLESCENT:
        return [f"adolescent#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.ADULT:
        return [f"adult#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.CHILD:
        return [f"child#00{i}" for i in range(1, 10)]


if __name__ == "__main__":
    # Test exact meal times
    print("Testing exact meal times:")
    print(
        f"NO_MEAL at 420min (7:00): {Scenario.NO_MEAL.get_carb(420, 70)}"
    )  # Should be 0
    print(
        f"SINGLE_MEAL at 360min (6:00): {Scenario.SINGLE_MEAL.get_carb(360, 70)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"THREE_DAY at 420min (7:00 day 1): {Scenario.THREE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35

    print("\nTesting non-meal times (should all return 0):")
    print(f"SINGLE_MEAL at 365min: {Scenario.SINGLE_MEAL.get_carb(365)}")  # Should be 0
    print(f"ONE_DAY at 425min: {Scenario.ONE_DAY.get_carb(425)}")  # Should be 0
    print(f"THREE_DAY at 425min: {Scenario.THREE_DAY.get_carb(425)}")  # Should be 0

    print("\nTesting without body_weight:")
    print(f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420)}")  # Should be 40
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140)}"
    )  # Should be 70
