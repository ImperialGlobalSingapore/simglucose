from simglucose.patient.t1dpatient_2 import T1DPatient, Action, CtrlObservation
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.simple_pid_ctrller import SimplePIDController

from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
# parent folder
parent_folder = file_path.parent

log_dir = parent_folder / "logs"
img_dir = parent_folder / "imgs"

log_dir.mkdir(exist_ok=True)
img_dir.mkdir(exist_ok=True)


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


def plot_and_save(img_dir, t, BG, CHO, insulin, target_BG, file_name):
    fig, ax = plt.subplots(3, sharex=True, figsize=(15, 10))
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, file_name)
    fig.savefig(img_dir / f"{file_name}.png")
    plt.close(fig)


def save_to_csv(log_dir, t, BG, CHO, insulin, file_name):
    csv_file = log_dir / f"{file_name}.csv"
    with open(csv_file, "w") as f:
        f.write("time,CHO,insulin,BG\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{CHO[i]},{insulin[i]},{BG[i]}\n")


def eval_result(BG, target_BG, k_P, k_I, k_D, sample_time):
    print(f"k_P: {k_P}, k_I: {k_I}, k_D: {k_D}, sample_time: {sample_time}")
    target_BG = np.array(target_BG)
    BG = np.array(BG)
    errors = target_BG - BG
    max_e = np.abs(errors).max()
    mae = np.mean(np.abs(errors))  # mean absolute error
    mse = np.mean(errors**2)  # mean square error
    iae = np.sum(np.abs(errors) * sample_time)
    ise = np.sum(errors**2 * sample_time)
    return {
        "MAX_E": max_e,
        "MAE": mae,
        "MSE": mse,
        "IAE": iae,
        "ISE": ise,
    }


# this pid controller is too complex to fix, ignore now"
""""
def run_sim_pid(k_P, k_I, k_D):
    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = PIDController(k_P=0.001, k_I=0.00001, k_D=0.001)
    logger.info("Using PID Controller")

    while p.t < 2000:
        carb = 0

        if p.t == 100:
            carb = 80

        # if p.t == 200:
        #     carb = 50

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    save_to_csv(t, BG, CHO, insulin, k_P, k_I, k_D)
"""


def run_sim_simple_pid_single_meal(k_P, k_I, k_D, sample_time=5):
    single_meal_log_dir = log_dir / "single_meal"
    single_meal_img_dir = img_dir / "single_meal"
    single_meal_log_dir.mkdir(exist_ok=True, parents=True)
    single_meal_img_dir.mkdir(exist_ok=True, parents=True)

    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(k_P=k_P, k_I=k_I, k_D=k_D, sampling_time=sample_time)
    logger.info(f"Testing {patient_name} using Simple PID Controller with single meal")

    while p.t < 2000:
        carb = 0

        if p.t == 100:
            carb = 80

        # if p.t == 200:
        #     carb = 50

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark="simple_"
    )
    # save_to_csv(single_meal_log_dir, t, BG, CHO, insulin, file_name)
    plot_and_save(single_meal_img_dir, t, BG, CHO, insulin, ctrl.target_BG, file_name)
    plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)


def run_sim_simple_pid_no_meal(k_P, k_I, k_D, sample_time=5, basal_rate=0.2):
    no_meal_log_dir = log_dir / "no_meal"
    no_meal_img_dir = img_dir / "no_meal"
    no_meal_log_dir.mkdir(exist_ok=True, parents=True)
    no_meal_img_dir.mkdir(exist_ok=True, parents=True)

    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(
        k_P=k_P,
        k_I=k_I,
        k_D=k_D,
        sampling_time=sample_time,
        basal_rate=basal_rate,
    )
    logger.info(f"Testing {patient_name} using Simple PID Controller without meal")

    while p.t < 2000:
        carb = 0

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
    print("Simulation done")

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark="simple_"
    )
    # save_to_csv(no_meal_log_dir, t, BG, CHO, insulin, file_name)
    plot_and_save(no_meal_img_dir, t, BG, CHO, insulin, ctrl.target_BG, file_name)
    # plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)
    return eval_result(BG, ctrl.target_BG, k_P, k_I, k_D, sample_time)


def run_sim_simple_pid_attack(k_P, k_I, k_D, sample_time=5):
    attack_no_meal_log_dir = log_dir / "attack_no_meal"
    attack_no_meal_img_dir = img_dir / "attack_no_meal"
    attack_no_meal_log_dir.mkdir(exist_ok=True, parents=True)
    attack_no_meal_img_dir.mkdir(exist_ok=True, parents=True)

    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(k_P=k_P, k_I=k_I, k_D=k_D, sampling_time=sample_time)
    logger.info(f"Testing {patient_name} using Simple PID Controller without meal")

    while p.t < 2000:
        carb = 0

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        if p.t == 100:
            ctrl_obs = CtrlObservation(300)
            logger.info(f"Attack at {p.t} min with {ctrl_obs.CGM} CGM")

        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    print("Simulation done")

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark="simple_"
    )
    # save_to_csv(attack_no_meal_log_dir, t, BG, CHO, insulin, file_name)
    plot_and_save(
        attack_no_meal_img_dir, t, BG, CHO, insulin, ctrl.target_BG, file_name
    )
    plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)


def test_range_simple_pid_no_meal(
    k_p: list, k_i: list, k_d: list, sample_time: list, basal_rate: list, csv_name
):
    csv_file = f"{csv_name}.csv"
    with open(csv_file, "w") as f:
        f.write("k_p,k_i,k_d,sample_time,basal_rate,max_e,mae,mse,iae,ise\n")
        for kp in k_p:
            for ki in k_i:
                for kd in k_d:
                    for st in sample_time:
                        for br in basal_rate:
                            print(f"Testing {kp}, {ki}, {kd}, {st}, {br}")
                            result = run_sim_simple_pid_no_meal(kp, ki, kd, st, br)
                            max_e = result["MAX_E"]
                            mae = result["MAE"]
                            mse = result["MSE"]
                            iae = result["IAE"]
                            ise = result["ISE"]
                            f.write(
                                f"{kp},{ki},{kd},{st},{br},{max_e},{mae},{mse},{iae},{ise}\n"
                            )


def find_good_br_kp():
    # step 1
    # k_p = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # k_i = [0]
    # k_d = [0]
    # sample_time = [5]
    # basal_rate = [0, 0.05, 0.1, 0.15, 0.2]
    # csv_name = "pid_no_meal_tunning_step1"

    # step 2
    # k_p = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # k_i = [0]
    # k_d = [0]
    # sample_time = [5]
    # basal_rate = [0.01, 0.02, 0.05, 0.06, 0.07]
    # csv_name = "pid_no_meal_tunning_step2"
    # test_range_simple_pid_no_meal(k_p, k_i, k_d, sample_time, basal_rate, csv_name)

    # step 3
    k_p = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-3, 1e-3]
    k_i = [0]
    k_d = [0]
    sample_time = [5]
    basal_rate = [0.05, 0.055, 0.06, 0.065, 0.07]
    csv_name = "pid_no_meal_tunning_step3"
    test_range_simple_pid_no_meal(k_p, k_i, k_d, sample_time, basal_rate, csv_name)


def find_good_ki_kd():
    k_p = [1e-4, 1e-5, 1e-6]
    basal_rate = [0.05, 0.06, 0.07, 0.08]
    k_i = [0, 1e-5, 1e-3, 1e-1]
    k_d = [0]
    sample_time = [5]
    csv_name = "pid_no_meal_tunning_step4"
    test_range_simple_pid_no_meal(k_p, k_i, k_d, sample_time, basal_rate, csv_name)


if __name__ == "__main__":
    # find good k_p and br range
    find_good_br_kp()

    # find good k_i and k_d range
    find_good_ki_kd()
