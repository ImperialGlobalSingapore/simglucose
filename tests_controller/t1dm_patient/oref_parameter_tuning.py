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
test_patient_dir = img_dir / "oref0_parameter_tuning"

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


def generate_profiles_by_group(group: PatientType):
    import numpy as np
    from itertools import product

    target_bg = 100  # mg/dL
    min_bg = 90  # mg/dL
    max_bg = target_bg * 2 - min_bg  # mg/dL

    # Define parameter ranges for each age group based on literature
    patient_group_default_profiles = {
        PatientType.CHILD: {
            "sens": 150,
            "dia": 7,
            "carb_ratio": 30,
            "max_iob": 20,  # from paper
            "max_basal": 5,  # from paper
            "max_daily_basal": 0.8,
            "max_bg": max_bg,
            "min_bg": min_bg,
            "maxCOB": 120,  # from oref0 code
            "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 150}]},
            "min_5m_carbimpact": 8,  # from paper and oref0 code
        },
        PatientType.ADULT: {
            "sens": 60,
            "dia": 7.0,
            "carb_ratio": 20,
            "max_iob": 30,  # from paper
            "max_basal": 10,  # from paper
            "max_daily_basal": 2.5,
            "max_bg": max_bg,
            "min_bg": min_bg,
            "maxCOB": 120,  # from oref0 code
            "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 60}]},
            "min_5m_carbimpact": 8,  # from paper and oref0 code
        },
    }

    parameter_group = {
        PatientType.CHILD: {
            "sens": {"step_count": 3, "range": (50, 100)},  # 1:50 to 1:100, gpt
            "dia": {"step_count": 3, "range": (5, 8)},  # DIA 5 to 8 hours, from paper
            "carb_ratio": {"step_count": 3, "range": (15, 20)},  # ICR 1:15 to 1:20, gpt
            # "max_iob": {"step_count": 3, "range": (2, 4)},
            # "max_basal": {
            #     "step_count": 3,
            #     "range": (0.8, 1.5),
            # },
            "max_daily_basal": {
                "step_count": 3,
                "range": (0.5, 1),
            },  # u/kg/day, gpt
        },
        PatientType.ADULT: {
            "sens": {"step_count": 3, "range": (30, 50)},  # ISF 1:30 to 1:50, gpt
            "dia": {"step_count": 3, "range": (5, 8)},  # DIA 5 to 8 hours, from paper
            "carb_ratio": {"step_count": 3, "range": (10, 15)},  # ICR 1:10 to 1:15, gpt
            # "max_iob": {"step_count": 3, "range": (5, 10)},
            # "max_basal": {
            #     "step_count": 3,
            #     "range": (2.5, 5.0),
            # },
            "max_daily_basal": {
                "step_count": 3,
                "range": (0.4, 1),
            },  # u/kg/day, gpt
        },
    }

    # Get base profile and parameter ranges for the group
    base_profile = patient_group_default_profiles[group]
    param_ranges = parameter_group[group]

    # Generate value arrays for each parameter
    param_values = {}
    for param_name, param_config in param_ranges.items():
        step_count = param_config["step_count"]
        min_val, max_val = param_config["range"]
        param_values[param_name] = np.linspace(min_val, max_val, step_count)

    # Generate all combinations
    param_names = list(param_values.keys())
    value_combinations = product(*[param_values[name] for name in param_names])

    # Create profiles
    profiles = []
    for idx, values in enumerate(value_combinations):
        profile = base_profile.copy()

        # Update each parameter value
        sens_value = None
        for param_name, value in zip(param_names, values):
            profile[param_name] = value
            if param_name == "sens":
                sens_value = value

        # Update isfProfile sensitivity to match sens value
        if sens_value is not None:
            profile["isfProfile"] = {
                "sensitivities": [{"offset": 0, "sensitivity": sens_value}]
            }

        profiles.append(profile)

    return profiles


if __name__ == "__main__":
    """
    following paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    resplit patients into 2 groups, children (7-12), adults (16-70)
    hence, select patients given age and randomness
    child: child#002, child#008, child#010
    adult: adolescent#003, adult#006, adult#009
    the goal is to have an optimal parameter set for each patient,
    that it has a realistic time-in-range distribution following the paper
    """

    scenarios = [Scenario.THREE_DAY]

    # patients_by_group = {
    #     PatientType.CHILD: ["child#002", "child#008", "child#010"],
    #     PatientType.ADULT: ["adolescent#003", "adult#006", "adult#009"],
    # }
    patients_by_group = {
        PatientType.ADULT: ["adult#007"],
    }

    # Generate profiles for each patient group
    patient_profiles_by_group = {}
    combined_profiles = {}
    for group in patients_by_group.keys():
        patient_profiles_by_group[group] = generate_profiles_by_group(group)
        combined_profiles[group] = {}
        for idx, profile in enumerate(patient_profiles_by_group[group]):
            parameter_idx = str(idx)
            combined_profiles[group][parameter_idx] = profile

    # Prepare all test configurations grouped by patient
    patient_configs = {}
    for group in patients_by_group.keys():
        for patient_name in patients_by_group[group]:
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

    # Calculate total configurations
    total_configs = sum(len(configs) for configs in patient_configs.values())

    # Run tests in parallel (by patient, profiles sequential within each patient)
    print(
        f"\n🚀 Starting parallel tests for {total_configs} configurations across {len(patient_configs)} patients..."
    )
    print(f"Using {min(multiprocessing.cpu_count(), len(patient_configs))} CPU cores\n")

    # Get profile keys from the first generated profile
    first_group = list(patients_by_group.keys())[0]
    profile_keys = [k for k in patient_profiles_by_group[first_group][0].keys()]

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
