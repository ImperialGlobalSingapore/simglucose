from simglucose.patient.t1dpatient_2 import T1DPatient, Action, CtrlObservation
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.simple_pid_ctrller import SimplePIDController

from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

# current file path
file_path = Path(__file__).resolve()
# parent folder
parent_folder = file_path.parent

log_dir = parent_folder / "logs"
img_folder = parent_folder / "imgs"
log_dir.mkdir(exist_ok=True)


def _plot(fig, ax, t, BG, CHO, insulin, target_BG, k_P, k_I, k_D):
    ax[0].plot(t, BG)
    ax[0].plot(t, [target_BG] * len(t), "r--")
    ax[0].grid()
    ax[0].set_ylabel("BG (mg/dL)")
    ax[1].plot(t, CHO)
    ax[1].grid()
    ax[1].set_ylabel("CHO (g)")
    ax[2].plot(t, insulin)
    ax[2].grid()
    ax[2].set_ylabel("Insulin (U)")
    ax[2].set_xlabel("Time (min)")
    fig.suptitle(f"PID Controller: K_P={k_P}, K_I={k_I}, K_D={k_D}")


def plot_and_show(t, BG, CHO, insulin, target_BG, k_P, k_I, k_D):
    fig, ax = plt.subplots(3, sharex=True)
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, k_P, k_I, k_D)
    plt.show()


def plot_and_save(t, BG, CHO, insulin, target_BG, k_P, k_I, k_D, remark=""):
    fig, ax = plt.subplots(3, sharex=True)
    _plot(fig, ax, t, BG, CHO, insulin, target_BG, k_P, k_I, k_D)
    fig.savefig(img_folder / f"{remark}pid_{k_P}_{k_I}_{k_D}.png")
    plt.close(fig)


def save_to_csv(t, BG, CHO, insulin, k_P, k_I, k_D, remark=""):
    csv_file = log_dir / f"{remark}pid_{k_P}_{k_I}_{k_D}.csv"
    with open(csv_file, "w") as f:
        f.write("time,CHO,insulin,BG\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{CHO[i]},{insulin[i]},{BG[i]}\n")


# this pid controller is too complex to fix, should ignore
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


def run_sim_simple_pid(k_P, k_I, k_D):
    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(k_P=k_P, k_I=k_I, k_D=k_D)
    logger.info("Using Simple PID Controller")

    while p.t < 1000:
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

    save_to_csv(t, BG, CHO, insulin, k_P, k_I, k_D, "simple_")
    plot_and_save(t, BG, CHO, insulin, ctrl.target_BG, k_P, k_I, k_D, "simple_")
    plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, k_P, k_I, k_D)


if __name__ == "__main__":
    # run_sim(0.001, 0, 0)
    # run_sim(1e-5, 0, 0)
    # run_sim_pid(0.1, 0, 0)
    run_sim_simple_pid(0.0001, 0, 0)
