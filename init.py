from click import File
import fire
import json

import numpy as np

from simglucose.patient.t1dpatient import T1DPatient


def main(output_file: str, patient: str = "adolescent#003"):
    # TODO: get state here
    patient = T1DPatient.withName(patient, seed=42)

    state = {
        "params": patient._params.to_dict(),
        # "state": patient.state.tolist(),
        "seed": patient._seed,
        "t": patient.t,
        "planned_meal": patient.planned_meal,
        "is_eating": patient.is_eating,
        "glucose": patient.observation.Gsub,
        "last_action": patient._last_action
    }
    np.save(output_file.replace('json', 'npy'), patient.state)
    # print(patient._params.to_dict())
    # for k, v in state.items():
    #     print(k, type(v))
    with open(output_file, "w+") as f:
        json.dump(state, f)
    print(patient.observation.Gsub)

if __name__ == "__main__":
    fire.Fire(main)
