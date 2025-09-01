import csv
import logging
from pathlib import Path
from collections import namedtuple
import shutil
import matplotlib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController

import sys

sys.path.append(str(Path(__file__).parent.parent))
from simglucose.simulation import scenario
from test_utils import *

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
test_patient_dir = img_dir / "test_oref0"
# delete all the subdir
for child in test_patient_dir.glob("*"):
    if child.is_dir():
        # remove contents of the directory and dont delete the directory itself
        for item in child.glob("*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003",
    img_save_dir=test_patient_dir,
    scenario=Scenario.NO_MEAL,
    profile=None,
    save_fig=False,
):

    p = T1DMPatient.withName(patient_name)
    if profile:
        ctrl = ORefZeroController(
            current_basal=p.basal * 60, profile=profile, timeout=30000
        )  # U/min to U/h
    else:
        ctrl = ORefZeroController(
            current_basal=p.basal * 60, timeout=30000
        )  # U/min to U/h
    t = []
    CHO = []
    insulin = []
    BG = []

    # Directory already created upfront, no need to create here

    time_in_range = {"very_high": 0, "high": 0, "target": 0, "low": 0, "very_low": 0}

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

        # Categorize BG into range groups (very_high, high, target, low, very_low)
        bg = p.observation.Gsub
        if bg > 250:
            time_in_range["very_high"] += 1
        elif bg > 180:
            time_in_range["high"] += 1
        elif bg > 70:
            time_in_range["target"] += 1
        elif bg > 54:
            time_in_range["low"] += 1
        else:
            time_in_range["very_low"] += 1

    timestamp = datetime.now()
    sanitized_patient_name = patient_name.replace("#", "_")
    fig_title = f"test_patient_{sanitized_patient_name}_{scenario.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    if save_fig:
        file_name = img_save_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, ctrl.target_bg, file_name)

    time_in_range = {k: v / len(t) for k, v in time_in_range.items() if v > 0}
    return time_in_range, timestamp


def run_single_patient_test(args):
    """Helper function for parallel execution"""
    patient_name, profile, scenario_val, img_save_dir = args
    try:
        time_in_range, timestamp = patient_oref0(
            patient_name=patient_name,
            img_save_dir=img_save_dir,
            save_fig=True,
            scenario=scenario_val,
            profile=profile,
        )
        return patient_name, args, profile, time_in_range, timestamp
    except Exception as e:
        return patient_name, args, profile, str(e), None


def run_patient_all_profiles(patient_configs):
    """Run all profile tests for a single patient sequentially"""
    results = []
    for config in patient_configs:
        result = run_single_patient_test(config)
        results.append(result)
    return results


if __name__ == "__main__":
    """
    Parameter ranges based on medical literature:

    References:
    1. ISF ranges: PMC8957904 - "Diurnal Variation of Real-Life Insulin Sensitivity Factor Among Children and Adolescents"
       - Children <6y: ISF 1:150 (70-228), 6-12y: ISF 1:90 (50-140), 12-18y: ISF 1:50 (40-80)

    2. ICR ranges: PMC5478012 - "Bolus Calculator Settings in Well-Controlled Prepubertal Children"
       - Children 2-4y: ICR 30 (13-42), 5-7y: ICR 20 (17-33), 8-10y: ICR 16 (10-60)
       - Adults typically ICR 1:10-15

    3. DIA: PMC5478012 & diabetesnet.com
       - Children: 2-3.5 hours, Adults: 4-4.5 hours

    4. Basal rates: PMC8186333 - "Initial Basal and Bolus Rates During Pump Treatment"
       - Children <7y: 0.69 U/kg/day, Adolescents: 0.90-0.97 U/kg/day

    5. COB/carbimpact: AndroidAPS documentation & OpenAPS guidelines
       - Default maxCOB: 120g, min_5m_carbimpact: 8 mg/dL/5min
    """

    patient_groups = [
        PatientType.CHILD,
        PatientType.ADOLESCENT,
        PatientType.ADULT,
    ]

    # Define parameter ranges for each age group based on literature
    patient_group_profiles = {
        PatientType.CHILD: [
            {
                "sens": 150,  # ISF 1:150 - most sensitive (PMC8957904)
                "dia": 2.5,  # Shortest DIA for young children
                "carb_ratio": 30,  # ICR 1:30 - needs least insulin per carb
                "max_iob": 2,  # Conservative for safety
                "max_basal": 0.8,
                "max_daily_basal": 0.8,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 60,  # Smaller meals
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 150}]},
                "min_5m_carbimpact": 6.0,  # Slower absorption
            },
            {
                "sens": 90,  # ISF 1:90 - moderate sensitivity (6-12y range)
                "dia": 3.0,  # Standard child DIA
                "carb_ratio": 20,  # ICR 1:20 - moderate
                "max_iob": 3,
                "max_basal": 1.2,
                "max_daily_basal": 1.2,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 80,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 90}]},
                "min_5m_carbimpact": 8.0,  # Default
            },
            {
                "sens": 70,  # ISF 1:70 - lower end of child range
                "dia": 3.5,  # Longer for older children
                "carb_ratio": 16,  # ICR 1:16 - older child
                "max_iob": 4,
                "max_basal": 1.5,
                "max_daily_basal": 1.5,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 100,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 70}]},
                "min_5m_carbimpact": 10.0,
            },
        ],
        PatientType.ADOLESCENT: [
            {
                "sens": 50,  # ISF 1:50 - typical adolescent (PMC8957904)
                "dia": 3.5,  # Shorter end for adolescents
                "carb_ratio": 12,  # ICR 1:12 - moderate insulin needs
                "max_iob": 4,
                "max_basal": 1.8,
                "max_daily_basal": 1.8,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 100,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 50}]},
                "min_5m_carbimpact": 10.0,
            },
            {
                "sens": 40,  # ISF 1:40 - insulin resistant
                "dia": 4.0,  # Standard adolescent
                "carb_ratio": 10,  # ICR 1:10 - higher insulin needs
                "max_iob": 5,
                "max_basal": 2.5,
                "max_daily_basal": 2.5,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 120,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 40}]},
                "min_5m_carbimpact": 12.0,
            },
            {
                "sens": 30,  # ISF 1:30 - puberty peak resistance
                "dia": 4.5,  # Longer for late adolescents
                "carb_ratio": 8,  # ICR 1:8 - highest insulin needs
                "max_iob": 6,
                "max_basal": 3.0,
                "max_daily_basal": 3.0,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 150,  # Larger meals
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 30}]},
                "min_5m_carbimpact": 14.0,
            },
        ],
        PatientType.ADULT: [
            {
                "sens": 60,  # ISF 1:60 - sensitive adult
                "dia": 4.0,  # Shorter adult DIA
                "carb_ratio": 20,  # ICR 1:20 - sensitive
                "max_iob": 5,
                "max_basal": 2.5,
                "max_daily_basal": 2.5,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 120,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 60}]},
                "min_5m_carbimpact": 10.0,
            },
            {
                "sens": 40,  # ISF 1:40 - typical adult (diabetesnet.com)
                "dia": 4.5,  # Standard adult DIA
                "carb_ratio": 15,  # ICR 1:15 - typical
                "max_iob": 7,
                "max_basal": 3.5,
                "max_daily_basal": 3.5,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 150,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 40}]},
                "min_5m_carbimpact": 12.0,
            },
            {
                "sens": 30,  # ISF 1:30 - insulin resistant adult
                "dia": 5.0,  # Longer for resistant adults
                "carb_ratio": 10,  # ICR 1:10 - higher insulin needs
                "max_iob": 10,
                "max_basal": 5.0,
                "max_daily_basal": 5.0,
                "max_bg": 120,
                "min_bg": 120,
                "maxCOB": 180,
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 30}]},
                "min_5m_carbimpact": 15.0,
            },
        ],
    }

    # Prepare all test configurations grouped by patient
    patient_configs = {}
    for group in patient_groups:
        patients = get_patient_by_group(group)
        # patients = patients[:2]  # Limit to first 2 patients for quick testing
        if patients:
            for patient_name in patients:
                patient_configs[patient_name] = []
                # Create patient folder upfront
                patient_folder = patient_name.replace("#", "_")
                patient_dir = test_patient_dir / patient_folder
                patient_dir.mkdir(exist_ok=True, parents=True)

                for profile_idx, profile in enumerate(patient_group_profiles[group]):
                    patient_configs[patient_name].append(
                        (
                            patient_name,
                            profile,
                            Scenario.SINGLE_MEAL,
                            patient_dir,
                        )
                    )

    # Calculate total configurations
    total_configs = sum(len(configs) for configs in patient_configs.values())

    # Run tests in parallel (by patient, profiles sequential within each patient)
    print(
        f"\n🚀 Starting parallel tests for {total_configs} configurations across {len(patient_configs)} patients..."
    )
    print(f"Using {min(multiprocessing.cpu_count(), len(patient_configs))} CPU cores\n")

    results_list = []  # Initialize results_list before the loop

    with ProcessPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), len(patient_configs))
    ) as executor:
        # Submit one job per patient (each job runs all profiles for that patient sequentially)
        futures = [
            executor.submit(run_patient_all_profiles, configs)
            for patient_name, configs in patient_configs.items()
        ]

        for future in as_completed(futures):
            patient_results = (
                future.result()
            )  # This is a list of results for one patient

            for result in patient_results:
                print(result)
                # Parse result string for status and patient/postfix
                if isinstance(result, tuple) and len(result) == 5:
                    # result is (patient_name, args, profile, time_in_range, timestamp)
                    # or (patient_name, args, profile, str(e), None)
                    patient_name = result[0]
                    args = result[1]
                    profile = result[2]
                    time_in_range = result[3]
                    timestamp = result[4]

                    if timestamp is None:
                        status = "Failed"
                        error_msg = f"Error: {time_in_range}"
                        time_in_range = None
                    else:
                        status = "Completed"
                        error_msg = None
                else:
                    status = "Failed"
                    patient_name = result[0] if len(result) > 0 else "Unknown"
                    profile = f"Error: Unexpected result format"
                    time_in_range = None
                    timestamp = None

                results_list.append(
                    {
                        "patient_name": patient_name,
                        "profile": profile if status == "Completed" else error_msg,
                        "status": status,
                        "very_high": (
                            time_in_range.get("very_high") if time_in_range else None
                        ),
                        "high": time_in_range.get("high") if time_in_range else None,
                        "target": (
                            time_in_range.get("target") if time_in_range else None
                        ),
                        "low": time_in_range.get("low") if time_in_range else None,
                        "very_low": (
                            time_in_range.get("very_low") if time_in_range else None
                        ),
                        "timestamp": timestamp,
                    }
                )
    print(f"\n✅ All tests completed!")

    # Save results to CSV
    try:
        csv_file = parent_folder / "test_results.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "patient_name",
                    "profile",
                    "status",
                    "very_high",
                    "high",
                    "target",
                    "low",
                    "very_low",
                    "timestamp",
                ],
            )
            writer.writeheader()
            for row in results_list:
                writer.writerow(row)
        print(f"\n✅ Results saved to {csv_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
