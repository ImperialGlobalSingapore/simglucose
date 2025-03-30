import os
import json
from itertools import product
from functools import partial
import pandas as pd
from test_t1dpatient_pid import run_sim_simple_pid_no_meal
from test_utils import get_patients

def test_range_simple_pid_no_meal(
    k_p: list,
    k_i: list,
    k_d: list,
    sample_time: list,
    basal_rate: list,
    csv_name,
    save_csv=False,
):
    csv_file = f"{csv_name}.csv"
    combinations = list(product(k_p, k_i, k_d, sample_time, basal_rate))
    results = []
    for combination in combinations:
        kp, ki, kd, st, br = combination
        rmse = run_sim_simple_pid_no_meal(kp, ki, kd, st, br)
        results.append(rmse)

    if save_csv:
        with open(csv_file, "w") as f:
            f.write(f"k_p,k_i,k_d,sample_time,basal_rate,rmse\n")
            for i in range(len(combinations)):
                f.write(
                    f"{combinations[i][0]},{combinations[i][1]},{combinations[i][2]},{combinations[i][3]},{combinations[i][4]},{results[i]}\n"
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
    k_p = [1e-6, 1e-5, 1e-4, 1e-3]
    k_i = [0]
    k_d = [0]
    sample_time = [5]
    basal_rate = [0.05, 0.06, 0.07]
    csv_name = "pid_no_meal_tunning_step3"
    test_range_simple_pid_no_meal(k_p, k_i, k_d, sample_time, basal_rate, csv_name)


def find_good_ki_kd():
    k_p = [1e-4, 1e-5, 1e-6]
    basal_rate = [0.05, 0.06, 0.07, 0.08]
    k_i = [1e-10, 5e-9, 1e-7]
    k_d = [0, 1e-8, 1e-7]
    sample_time = [5]
    csv_name = "pid_no_meal_tunning_step4"
    test_range_simple_pid_no_meal(k_p, k_i, k_d, sample_time, basal_rate, csv_name)


def run_single_simulation(
    params: tuple,
    sample_time: int,
    sim_time: int,
):
    kp, ki, kd, br, patient_name = params
    try:
        rmse = run_sim_simple_pid_no_meal(
            k_P=kp,
            k_I=ki,
            k_D=kd,
            sample_time=sample_time,
            basal_rate=br,
            patient_name=patient_name,
            sim_time=sim_time,
            save_fig=False,
            show_fig=False,
        )
        return {"k_p": kp, "k_i": ki, "k_d": kd, "basal_rate": br, "rmse": rmse}
    except Exception as e:
        print(f"Error running simulation with params {params}: {str(e)}")
        return None


def parallel_test_pid_parameters(
    k_p_range=[1e-6, 1e-5, 1e-4, 1e-3],
    k_i_range=[0, 1e-10, 1e-9, 1e-8, 1e-7],
    k_d_range=[0, 1e-8, 1e-7],
    sample_time=5,
    basal_rate=[0, 0.05, 0.1, 0.15, 0.2],
    sim_time=2000,
    n_jobs=4,
):
    """
    Run parallel simulations to test different PID parameters
    """
    from multiprocessing import Pool
    from itertools import product

    # Generate parameter combinations
    patients = get_patients()

    param_combinations = list(
        product(k_p_range, k_i_range, k_d_range, basal_rate, patients)
    )

    # Filter combinations where ki is smaller than kp (if ki is not 0)
    valid_combinations = [
        (kp, ki, kd, br, patient_name)
        for kp, ki, kd, br, patient_name in param_combinations
        if ki == 0 or ki < kp
    ]

    print(f"Testing {len(valid_combinations)} parameter combinations...")

    simulation_with_fixed_params = partial(
        run_single_simulation,
        sample_time=sample_time,
        sim_time=sim_time,
    )
    # Run parallel simulations
    with Pool(processes=n_jobs) as pool:
        results = pool.map(simulation_with_fixed_params, valid_combinations)

    # Filter out failed simulations and convert to DataFrame
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    # Find best 5 parameters for each patient and dump to one json file
    best_params_dict = {}
    for patient_name in patients:
        best_params = (
            df[df["patient_name"] == patient_name].sort_values(by="rmse").head(5)
        )
        best_params_dict[patient_name] = best_params.to_dict(orient="records")
    json_file = f"pid_no_meal_tunning_step5_{sample_time}min_{sim_time}min.json"
    with open(json_file, "w") as f:
        json.dump(best_params_dict, f)

    return json_file


def plot_best_5_params(
    json_file: str, patient_name: str, sample_time: int, sim_time: int
):
    with open(json_file, "r") as f:
        best_params = json.load(f)
    print(best_params)

    for i, row in enumerate(best_params):
        print(f"No.{i+1}")
        print(
            f"k_p={row['k_p']}, k_i={row['k_i']}, k_d={row['k_d']}, basal_rate={row['basal_rate']}, rmse={row['rmse']:.4f}"
        )

        # Run simulation with best parameters and save plot
        run_sim_simple_pid_no_meal(
            k_P=row["k_p"],
            k_I=row["k_i"],
            k_D=row["k_d"],
            sample_time=sample_time,
            basal_rate=row["basal_rate"],
            patient_name=patient_name,
            sim_time=sim_time,
            save_fig=True,
            log=False,
        )


if __name__ == "__main__":

    # find good k_p and br range
    # find_good_br_kp()

    # find good k_i and k_d range
    # find_good_ki_kd()

    json_file = parallel_test_pid_parameters(
        k_p_range=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        k_i_range=[0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        k_d_range=[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        basal_rate=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        n_jobs=128,
    )

    # plot best 5 params
    json_file = "adolescent#003_best_5_params_5min_2000min.json"
    plot_best_5_params(
        json_file=json_file,
        patient_name="adolescent#003",
        sample_time=5,
        sim_time=2000,
    )
