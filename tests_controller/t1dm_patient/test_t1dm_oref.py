import csv
import json
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

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def patient_oref0(
    patient_name="adolescent#003",
    img_save_dir=test_patient_dir,
    scenario=Scenario.NO_MEAL,
    parameter_idx="0",
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
    time_in_range = {k: v / len(t) for k, v in time_in_range.items() if v > 0}

    sanitized_patient_name = patient_name.replace("#", "_")
    fig_title = f"test_patient_{sanitized_patient_name}_{scenario.name}_param_set_{parameter_idx}"
    if save_fig:
        file_name = img_save_dir / f"{fig_title}.png"
        plot_with_scale_and_save(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_bg,
            time_in_range,
            file_name,
        )

    return time_in_range


def run_single_patient_test(args):
    """Helper function for parallel execution"""
    patient_name, profile, scenario_val, img_save_dir, parameter_idx = args
    try:
        time_in_range = patient_oref0(
            patient_name=patient_name,
            img_save_dir=img_save_dir,
            save_fig=True,
            scenario=scenario_val,
            profile=profile,
            parameter_idx=parameter_idx,
        )
        return (
            patient_name,
            profile,
            scenario_val,
            img_save_dir,
            parameter_idx,
            time_in_range,
        )
    except Exception as e:
        return patient_name, profile, scenario_val, img_save_dir, parameter_idx, str(e)


def run_patient_all_profiles(patient_configs):
    """Run all profile tests for a single patient sequentially"""
    results = []
    for config in patient_configs:
        result = run_single_patient_test(config)
        results.append(result)
    return results


def execute_tests_and_process_results(
    patient_configs_dict, profile_keys, is_retry=False
):
    """
    Execute tests in parallel and process results.

    Args:
        patient_configs_dict: Dictionary mapping patient names to their config lists
        profile_keys: List of profile keys for result dictionaries
        is_retry: Boolean indicating if this is a retry execution

    Returns:
        tuple: (results_list, failed_configs)
    """
    results_list = []
    failed_configs = {}

    with ProcessPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), len(patient_configs_dict))
    ) as executor:
        futures = [
            executor.submit(run_patient_all_profiles, configs)
            for patient_name, configs in patient_configs_dict.items()
        ]

        for future in as_completed(futures):
            patient_results = future.result()

            for result in patient_results:
                if not is_retry:
                    print(result)

                if isinstance(result, tuple) and len(result) == 6:
                    patient_name = result[0]
                    profile = result[1]
                    scenario_val = result[2]
                    img_save_dir = result[3]
                    parameter_idx = result[4]
                    time_in_range = result[5]

                    if isinstance(time_in_range, str):
                        if is_retry:
                            status = "Failed_After_Retry"
                            error_msg = f"Retry failed: {time_in_range}"
                            print(
                                f"  ❌ {patient_name} (param {parameter_idx}) - Still failing after retry"
                            )
                        else:
                            status = "Failed"
                            error_msg = f"Error: {time_in_range}"
                        time_in_range = None

                        # Add to failed configs for potential retry
                        if not is_retry:  # Only collect failed configs on first run
                            if patient_name not in failed_configs:
                                failed_configs[patient_name] = []
                            failed_configs[patient_name].append(
                                (
                                    patient_name,
                                    profile,
                                    scenario_val,
                                    img_save_dir,
                                    parameter_idx,
                                )
                            )
                    else:
                        if is_retry:
                            status = "Completed_On_Retry"
                            print(
                                f"  ✅ {patient_name} (param {parameter_idx}) - Succeeded on retry"
                            )
                        else:
                            status = "Completed"
                        error_msg = None
                else:
                    # Unexpected result format
                    status = "Failed_After_Retry" if is_retry else "Failed"
                    patient_name = result[0] if len(result) > 0 else None
                    profile = result[1] if len(result) > 1 else None
                    scenario_val = result[2] if len(result) > 2 else None
                    img_save_dir = result[3] if len(result) > 3 else None
                    parameter_idx = result[4] if len(result) > 4 else None
                    time_in_range = result[5] if len(result) > 5 else None
                    error_msg = "Unexpected result format"

                    if not is_retry:
                        if patient_name not in failed_configs:
                            failed_configs[patient_name] = []
                        failed_configs[patient_name].append(
                            (
                                patient_name,
                                profile,
                                scenario_val,
                                img_save_dir,
                                parameter_idx,
                            )
                        )

                # Build result dict
                result_dict = {"patient_name": patient_name}

                # Add profile keys
                if status in ["Completed", "Completed_On_Retry"] and isinstance(
                    profile, dict
                ):
                    for key in profile_keys:
                        result_dict[key] = profile.get(key)
                else:
                    for key in profile_keys:
                        result_dict[key] = None

                # Add remaining fields
                result_dict.update(
                    {
                        "scenario": scenario_val.name if scenario_val else None,
                        "parameter_idx": parameter_idx,
                        "error_msg": error_msg if "Failed" in status else None,
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
                        "status": status,
                    }
                )

                results_list.append(result_dict)

    return results_list, failed_configs


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
        # PatientType.CHILD,
        # PatientType.ADOLESCENT,
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

    parameter_group = {
        0: {"min_bg": 120, "max_bg": 120},
        1: {"min_bg": 90, "max_bg": 140},
    }

    combined_profiles = {}
    for group, profiles in patient_group_profiles.items():
        combined_profiles[group] = {}
        for idx, profile in enumerate(profiles):
            for param_idx, param in parameter_group.items():
                parameter_idx = f"{idx}_{param_idx}"
                # Merge profile and param, and add an index for identification
                temp_profile = profile.copy()
                temp_profile.update(param)
                combined_profiles[group][parameter_idx] = temp_profile
                # TODO, for quick testing
                # break
            # TODO, for quick testing
            # break

    scenarios = [Scenario.SINGLE_MEAL, Scenario.ONE_DAY, Scenario.THREE_DAY]

    # Prepare all test configurations grouped by patient
    patient_configs = {}
    for group in patient_groups:
        patients = get_patient_by_group(group)
        # TODO:
        # patients = patients[:1]  # for quick testing
        if patients:
            for patient_name in patients:
                patient_configs[patient_name] = []
                # Create patient folder upfront
                for parameter_idx, profile_data in combined_profiles[group].items():
                    patient_folder = patient_name.replace("#", "_")
                    print(f"Created directory: {patient_dir}")
                    for sc in scenarios:
                        patient_dir = test_patient_dir / patient_folder / sc.value
                        patient_dir.mkdir(exist_ok=True, parents=True)
                        patient_configs[patient_name].append(
                            (patient_name, profile_data, sc, patient_dir, parameter_idx)
                        )
        break

    # Calculate total configurations
    total_configs = sum(len(configs) for configs in patient_configs.values())

    # Run tests in parallel (by patient, profiles sequential within each patient)
    print(
        f"\n🚀 Starting parallel tests for {total_configs} configurations across {len(patient_configs)} patients..."
    )
    print(f"Using {min(multiprocessing.cpu_count(), len(patient_configs))} CPU cores\n")

    profile_keys = [k for k in patient_group_profiles[PatientType.CHILD][0].keys()]

    # Run initial tests
    results_list, failed_configs = execute_tests_and_process_results(
        patient_configs, profile_keys, is_retry=False
    )

    print(f"\n✅ Initial test run completed!")

    # Retry failed tests if any
    if failed_configs:
        print(
            f"\n🔄 Retrying {sum(len(configs) for configs in failed_configs.values())} failed configurations..."
        )

        # Run retry tests
        retry_results, _ = execute_tests_and_process_results(
            failed_configs, profile_keys, is_retry=True
        )

        # Update original results with retry results
        for retry_result in retry_results:
            # Find matching original result and update it
            for i, original_result in enumerate(results_list):
                if (
                    original_result["patient_name"] == retry_result["patient_name"]
                    and original_result["parameter_idx"]
                    == retry_result["parameter_idx"]
                    and original_result["scenario"] == retry_result["scenario"]
                    and original_result["status"] == "Failed"
                ):
                    # Replace the failed result with the retry result
                    results_list[i] = retry_result
                    break

        print(f"✅ Retry completed!")

    print(f"\n✅ All tests completed!")

    # Save results to CSV
    fieldnames = [
        "patient_name",
        *profile_keys,
        "scenario",
        "parameter_idx",
        "error_msg",
        "very_high",
        "high",
        "target",
        "low",
        "very_low",
        "status",
    ]

    # Save to CSV
    try:
        csv_file = test_patient_dir / "test_results.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_list:
                writer.writerow(row)
        print(f"\n✅ Results saved to {csv_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        try:
            error_log_file = test_patient_dir / "error_log.txt"
            with open(error_log_file, "a") as ef:
                ef.write(f"{datetime.now()}: Error saving results to CSV: {e}\n")
            print(f"CSV error logged to {error_log_file}")
        except Exception as log_error:
            print(f"Error writing to log file: {log_error}")

    # Always save to JSON as well
    try:
        json_file = test_patient_dir / "test_results.json"
        with open(json_file, "w") as f:
            json.dump(results_list, f, indent=2, default=str)
        print(f"✅ Results saved to {json_file} (JSON backup)")
    except Exception as json_error:
        print(f"Error saving results to JSON: {json_error}")
        try:
            error_log_file = test_patient_dir / "error_log.txt"
            with open(error_log_file, "a") as ef:
                ef.write(
                    f"{datetime.now()}: Error saving results to JSON: {json_error}\n"
                )
            print(f"JSON error logged to {error_log_file}")
        except Exception as log_error:
            print(f"Error writing to log file: {log_error}")

    # Print final summary
    failed_count = sum(1 for r in results_list if "Failed" in r.get("status", ""))
    success_count = sum(
        1
        for r in results_list
        if r.get("status") in ["Completed", "Completed_On_Retry"]
    )

    print(f"\n📊 Final Summary:")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  Total: {len(results_list)}")
