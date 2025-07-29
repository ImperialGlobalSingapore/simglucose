import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import pandas as pd

from simglucose.patient.t1dpatient import PATIENT_PARA_FILE


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
        """
        default_carb_duration = 15  # hour = 15 minutes
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

        carb_time_range = (
            np.array(carb_times_in_hour[self]) * 60
        )  # Convert hours to minutes
        carb_time_range = np.vstack(
            (carb_time_range, carb_time_range + default_carb_duration)
        ).T
        find_within = carb_time_range - t > 0
        find_within = find_within[:, 0] ^ find_within[:, 1]
        if np.any(find_within):
            idx = np.where(find_within)[0][0]
            return carb_amounts[self][idx]
        return 0


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
    ax[2].set_ylabel("Insulin (U)")
    ax[2].set_xlabel("Time (min)")
    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc="lower center", ncol=3)
    fig.suptitle(f"PID Controller: {fig_title}")


def plot_and_show(t, BG, CHO, insulin, target_BG, fig_title):
    fig, ax = plt.subplots(3, sharex=True)
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, fig_title)
    plt.show()


def save_name_pattern(k_P, k_I, k_D, sample_time, basal_rate, remark=""):
    return f"{remark}pid_st{sample_time}_br{basal_rate}_p{k_P}_i{k_I}_d{k_D}"


def plot_and_save(t, BG, CHO, insulin, target_BG, file_name):
    fig, ax = plt.subplots(3, sharex=True, figsize=(15, 10))
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, file_name)
    fig.savefig(f"{file_name}.png")
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


def get_patient_by_group(patient_type: str):
    if patient_type == PatientType.ADOLESCENT.value:
        return [f"adolescent#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.ADULT.value:
        return [f"adult#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.CHILD.value:
        return [f"child#00{i}" for i in range(1, 10)]


if __name__ == "__main__":
    print(Scenario.THREE_DAY.get_carb(60, 70))
    print(Scenario.ONE_DAY.get_carb(60, 70))
    print(Scenario.SINGLE_MEAL.get_carb(60, 70))
    print(Scenario.NO_MEAL.get_carb(60, 70))

    print(Scenario.NO_MEAL.get_carb(365, 70))
    print(Scenario.SINGLE_MEAL.get_carb(365, 70))
    print(Scenario.ONE_DAY.get_carb(425, 70))
    print(Scenario.THREE_DAY.get_carb(425, 70))
    print(Scenario.ONE_DAY.get_carb(425))
    print(Scenario.THREE_DAY.get_carb(425))
