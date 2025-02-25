from collections import namedtuple
from click import File
import fire
import json
import pandas as pd
import numpy as np
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from matplotlib import pyplot as plt
import json

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
Action = namedtuple("patient_action", ["CHO", "insulin"])
PatientObservation = namedtuple("observation", ["Gsub"])

if __name__ == "__main__":
    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name, seed=42)
    # basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    states = []
    ctrl = BBController(target=140)
    while p.t < 50:
        # ins = basal
        carb = 0

        carb = 0
        if p.t == 5:
            carb = 20
        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(observation=ctrl_obs, reward=0, done=False, patient_name=patient_name, meal=carb)
        ins = ctrl_action.basal + ctrl_action.bolus
        print(p.t, p.planned_meal, ins)
        # states.append(p.)

        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
        print(20*'----')
        if p.t == 50:
            print({k:v.tolist() if k == "init_state" else v for k,v in p.__dict__.items() if k in ("_init_state", "random_init_bg", "_seed", "t0", "init_state", "random_state", "_last_Qsto", "_last_foodtaken", "name", "_last_action", "is_eating", "planned_meal")})
            assert False
        states.append(
            {
                "params": p._params.to_dict(),
                "state": p.state.tolist(),
                "seed": p._seed,
                "t": p.t,
                "planned_meal": p.planned_meal,
                "is_eating": p.is_eating,
                "glucose": p.observation.Gsub,
                "insulin": ins,
            }
        )

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.savefig("fig1.png")
    #plt.show()
    with open("sim.json", "w+") as f:
        json.dump({"bg": BG, "cho": CHO, "insulin": insulin, "states": states}, f)
