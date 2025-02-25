from collections import namedtuple
from uuid import uuid4
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


class Server:
    def __init__(self):
        self.patient_map = {}

    def init(output_file: str, patient: str = "adolescent#003"):
        patient = T1DPatient.withName(patient, seed=42)

        patient_id = uuid4()
        return {"initial_glucose": patient.observation.Gsub, "patient_id": patient_id}

    def step(
        self,
        patient_id: int,
        glucose_reading: float,
        carbs: int = 0,
        delta_time: int = 1,
        controller_algorithm: str = "basal_bolus",  # todo: adjust to default
        pump: str = "DefaultPump",  # todo adjust but not relevant
    ):
        patient = self.patient_map[patient_id]
        ctrl = contoller_map[controller_algorithm]()
        ctrl_obs = CtrlObservation(glucose_reading)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs, reward=0, done=False, patient_name=patient.name, meal=carbs
        )

        insulin = ctrl_action.basal + ctrl_action.bolus
        patient_action = PatientAction(insulin=insulin, CHO=carbs)
        patient.step(patient_action)

        glucose = patient.observation.Gsub

        return({"glucose": glucose, "insulin": insulin})


if __name__ == "__main__":
    #fire.Fire(main)
    ...