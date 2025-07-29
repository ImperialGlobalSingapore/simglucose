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
}

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003", scenario=Scenario.NO_MEAL, save_fig=False
):
    p = T1DMPatient.withName(patient_name)
    ctrl = ORefZeroController(patient_profile=profile)
    t = []
    CHO = []
    insulin = []
    BG = []

    test_patient_dir = img_dir / f"test_oref0"
    test_patient_dir.mkdir(exist_ok=True)

    while p.t < max_t[scenario]:
        carb = scenario.get_carb(p.t, p._params.BW)
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

    if save_fig:
        fig_title = f"test_patient_{patient_name}_no_basal_{scenario.name}"
        file_name = test_patient_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, BG[0], file_name)


if __name__ == "__main__":
    patient_oref0()
