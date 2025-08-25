from collections import namedtuple
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta


from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController

import sys

sys.path.append(str(Path(__file__).parent.parent))
from simglucose.simulation import scenario
from test_utils import *


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003",
    scenario=Scenario.NO_MEAL,
    save_fig=False,
):

    p = T1DMPatient.withName(patient_name)
    ctrl = ORefZeroController(current_basal=p.basal, timeout=30000)
    # ctrl = ORefZeroController(current_basal=p.basal, timeout=30000)
    t = []
    CHO = []
    insulin = []
    BG = []

    test_patient_dir = img_dir / f"test_oref0"
    test_patient_dir.mkdir(exist_ok=True)

    sim_t = datetime.now()

    while p.t_elapsed < max_t[scenario]:
        carb = scenario.get_carb(p.t_elapsed, p._params.BW)

        ctrl_obs = CtrlObservation(p.observation.Gsub)

        if p.observation.Gsub < 39:
            print("Patient is dead")
            break

        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )

        ins = ctrl_action.basal + ctrl_action.bolus
        act = Action(insulin=ins, CHO=carb)  # U/min

        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
        print(
            f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {p.observation.Gsub}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

        sim_t = sim_t + timedelta(minutes=10)

    fig_title = f"test_patient_{patient_name}_{scenario.name}_new_default_profile"
    if save_fig:
        file_name = test_patient_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, ctrl.target_bg, file_name)


if __name__ == "__main__":
    # test to get the default profile
    patient_oref0(save_fig=True)
    # patient_oref0(save_fig=True, scenario=Scenario.SINGLE_MEAL)
