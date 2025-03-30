import logging
from pathlib import Path

import pandas as pd
from multiprocessing import Pool
from itertools import product

from simglucose.patient.t1dpatient import T1DPatient, Action, PATIENT_PARA_FILE
from simglucose.patient.t1dpatient_2 import CtrlObservation
from simglucose.controller.simple_pid_ctrller import SimplePIDController

from test_utils import plot_and_show, plot_and_save, save_name_pattern, get_rmse

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

# testing single meal is not necessary for demo, ignore now
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

        # if p.t == 100:
        #     carb = 80

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

        act = Action(insulin=0, CHO=carb)

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
"""


def run_sim_simple_pid_no_meal(
    k_P,
    k_I,
    k_D,
    sample_time=5,
    basal_rate=0.2,
    patient_name="adolescent#003",
    sim_time=2000,
    save_fig=False,
    show_fig=False,
    log=True,
):
    no_meal_log_dir = log_dir / "no_meal"
    no_meal_img_dir = img_dir / "no_meal"
    no_meal_log_dir.mkdir(exist_ok=True, parents=True)
    no_meal_img_dir.mkdir(exist_ok=True, parents=True)

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
    if log:
        logger.info(
            f"Testing {patient_name} using Simple PID Controller p{k_P}, i{k_I}, d{k_D}, br{basal_rate} without meal"
        )

    while p.t < sim_time:

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

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark="simple_"
    )
    if save_fig:
        plot_and_save(no_meal_img_dir, t, BG, CHO, insulin, ctrl.target_BG, file_name)
    if show_fig:
        plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)

    rmse = get_rmse(BG, ctrl.target_BG)
    if log:
        logger.info(f"{patient_name}: {k_P}, {k_I}, {k_D} rmse: {rmse}")
    return rmse


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
    plot_and_save(
        attack_no_meal_img_dir, t, BG, CHO, insulin, ctrl.target_BG, file_name
    )
    plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)


if __name__ == "__main__":
    run_sim_simple_pid_no_meal(
        k_P=0.001, k_I=0.00001, k_D=0.001, sample_time=5, basal_rate=0.2
    )
