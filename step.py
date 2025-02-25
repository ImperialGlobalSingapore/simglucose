from collections import namedtuple
from click import File
import fire
import json
import pandas as pd
import numpy as np
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient


CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
PatientAction = namedtuple("patient_action", ["CHO", "insulin"])
PatientObservation = namedtuple("observation", ["Gsub"])

contoller_map = {"basal_bolus": BBController, "pid": PIDController}


def main(
    input_state_file: str,
    output_state_file: str,
    glucose_reading: float,
    carbs: int = 0,
    delta_time: int = 1,
    controller_algorithm: str = "basal_bolus",  # todo: adjust to default
    pump: str = "DefaultPump",  # todo adjust but not relevant
):
    with open(input_state_file, "r") as f:
        input_state = json.load(f)
    input_state['state'] = np.load(input_state_file.replace('json', 'npy'))
    patient = T1DPatient(
        params=pd.Series(input_state["params"]),
        init_state=np.array(input_state["state"]),
        seed=input_state["seed"],
        t0=input_state["t"],
    )
    patient.is_eating = input_state["is_eating"]
    patient.planned_meal = input_state["planned_meal"]
    patient._last_action = PatientAction(*input_state['last_action'])
    print(input_state["t"], patient.planned_meal)
    ctrl = contoller_map[controller_algorithm]()
    ctrl_obs = CtrlObservation(glucose_reading)
    ctrl_action = ctrl.policy(
        observation=ctrl_obs, reward=0, done=False, patient_name=input_state["params"]["Name"], meal=carbs
    )

    insulin = ctrl_action.basal + ctrl_action.bolus
    patient_action = PatientAction(insulin=insulin, CHO=carbs)
    patient.step(patient_action)

    glucose = patient.observation.Gsub

    output_state = {
        "params": patient._params.to_dict(),
        "seed": patient._seed,
        "t": patient.t,
        "planned_meal": patient.planned_meal,
        "is_eating": patient.is_eating,
        "glucose": glucose,
        "insulin": insulin,
        "last_action": patient._last_action
    }
    with open(output_state_file, "w+") as f:
        json.dump(output_state, f)
    np.save(output_state_file.replace('json', 'npy'), patient.state)
    print({"glucose": glucose, "insulin": insulin})


if __name__ == "__main__":
    fire.Fire(main)
