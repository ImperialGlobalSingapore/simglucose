from collections import namedtuple
import logging
from tqdm import tqdm
from pathlib import Path


from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController

import sys

sys.path.append(str(Path(__file__).parent.parent))
from simglucose.simulation import scenario
from test_utils import plot_and_show, plot_and_save, get_rmse, Scenario


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"

max_t = {
    Scenario.NO_MEAL: 1000,  # 16 hours + 40 minutes
    Scenario.SINGLE_MEAL: 1080,  # 18 hours
    Scenario.ONE_DAY: 1450,  # 24 hours + 10 minutes
    Scenario.THREE_DAY: 4330,  # 72 hours + 10 minutes
}

profile = {
    # commonly adjusted
    "max_iob": 0,
    "max_daily_safety_multiplier": 3,
    "current_basal_safety_multiplier": 4,
    "autosens_max": 1.2,
    "autosens_min": 0.8,
    "rewind_resets_autosens": True,
    "unsuspend_if_no_temp": False,
    "curve": "rapid-acting",
    # no need to adjust
    "adv_target_adjustments": True,
    # oref1, disabled by default
    "enableSMB_after_carbs": False,
    "enableSMB_with_COB": False,
    "enableSMB_with_temptarget": False,
    "enableUAM": False,
    # default
    "maxCOB": 120,
    # current basal
}

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003",
    scenario=Scenario.NO_MEAL,
    save_fig=False,
):
    p = T1DMPatient.withName(patient_name)
    profile["current_basal"] = p.basal
    ctrl = ORefZeroController(profile=profile, timeout=30000)
    # ctrl = ORefZeroController(current_basal=p.basal, timeout=30000)
    t = []
    CHO = []
    insulin = []
    BG = []

    test_patient_dir = img_dir / f"test_oref0"
    test_patient_dir.mkdir(exist_ok=True)

    from datetime import datetime, timedelta

    sim_t = datetime.now()
    while p.t_elapsed < max_t[scenario]:
        carb = scenario.get_carb(p.t_elapsed, p._params.BW)
        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=sim_t,
        )

        ins = ctrl_action.basal + ctrl_action.bolus
        act = Action(insulin=ins, CHO=carb)

        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
        print(
            f"\033[94mt: {p.t_elapsed}, BG: {p.observation.Gsub}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

        sim_t = sim_t + timedelta(minutes=5)

    fig_title = f"test_patient_{patient_name}_{scenario.name}_new_default_profile"
    if save_fig:
        file_name = test_patient_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, BG[0], file_name)


if __name__ == "__main__":
    patient_oref0(save_fig=True)
