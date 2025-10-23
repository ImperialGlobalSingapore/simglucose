import logging
from datetime import datetime
from pathlib import Path
from collections import namedtuple
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from simglucose.simulation.scenario_simple import Scenario
from plot_utils import plot_and_save_with_tir
from bg_attacker import BGAttacker
from tests_controller.time_in_range_config import TIRConfig


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
test_patient_dir = img_dir / "test_attack"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_save_dir = test_patient_dir / timestamp
image_save_dir.mkdir(parents=True, exist_ok=True)

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003",
    img_save_dir=image_save_dir,
    scenario: Scenario = Scenario.ONE_DAY,
    attack_step: float = 1.8,  # mg/dL per minute
    attack_maintain: float = 60,  # minutes
    profile=None,
    attacking=True,
    save_fig=False,
):

    p = T1DMPatient.withName(patient_name)
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    ctrl = ORefZeroController(timeout=30000)  # TODO: remove this when not in debug
    ctrl.initialize_patient(patient_name, profile=profile)

    t = []
    CHO = []
    insulin = []
    BG = []

    attacking_ts = [300, 900]
    attacker = BGAttacker(step=attack_step, maintain_duration=attack_maintain)

    while p.t_elapsed < scenario.max_t:
        carb = scenario.get_carb(p.t_elapsed, p._params.BW)

        if p.observation.Gsub < 39:
            print("Patient is dead")
            break

        if p.t_elapsed in attacking_ts:
            print(f"Starting attack at t={p.t_elapsed}")
            attacker.start_attack(p.t_elapsed, p.observation.Gsub)

        if attacking:
            glucose = attacker.get_spoofed_bg(p.t_elapsed, p.observation.Gsub)
        else:
            glucose = p.observation.Gsub

        ctrl_obs = CtrlObservation(glucose)

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
        BG.append(glucose)
        p.step(act)

        print(
            f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {glucose}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    sanitized_patient_name = patient_name.replace("#", "_")
    fig_title = f"test_patient_{sanitized_patient_name}_{scenario.name}"
    if save_fig:
        file_name = img_save_dir / f"{fig_title}.png"
        plot_and_save_with_tir(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_bg,
            file_name,
            time_in_range,
            tir_config,
        )

    return time_in_range


if __name__ == "__main__":
    """
    using android app logic from
    https://github.com/ImperialGlobalSingapore/GoodV/blob/135890492f03820744cdf6efe998eceaa9da4721/app/src/main/java/com/kk4vcz/goodv/CGM_RW_Fragment.java#L282-L350

    to attack the patient CGM readings and see if OpenAPS can be tricked into delivering more insulin
    """
    patient_name = "adult#007"
    target_bg = 100  # mg/dL
    min_bg = 70  # mg/dL
    max_bg = 180  # mg/dL
    profile = {
        "sens": 45,
        "dia": 7.0,
        "carb_ratio": 20,
        "max_iob": 12,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_daily_basal": 0.9,  # from paper
        "max_bg": max_bg,
        "min_bg": min_bg,
        "maxCOB": 120,  # from oref0 code
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 60}]},
        "min_5m_carbimpact": 8,  # from paper and oref0 code
    }

    attack_step = 1.8  # mg/dL per minute
    attack_maintain = 30  # minutes

    patient_oref0(
        patient_name=patient_name,
        profile=profile,
        save_fig=True,
        attacking=True,
        attack_step=attack_step,
        attack_maintain=attack_maintain,
    )
