import logging
import matplotlib
from pathlib import Path
from collections import namedtuple
from typing import Any, Dict, List
import numpy as np
from itertools import product

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController
from simglucose.simulation.scenario_simple import Scenario
from glucose_control_analytics import PatientType, OpenAPSParameterTuningBase

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent
result_dir = parent_folder / "imgs" / "oref0_parameter_tuning"


CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def parse_virtual_patient_id(virtual_patient_id: str) -> tuple[str, int, str]:
    """
    Parse a virtual patient ID to extract patient name, parameter index, and scenario.

    Args:
        virtual_patient_id: Virtual patient ID (e.g., "adult_001_param_0_NO_MEAL")

    Returns:
        Tuple of (patient_name, param_idx, scenario_name)
        e.g., ("adult#001", 0, "NO_MEAL")
    """
    parts = virtual_patient_id.split("_")

    # Find the index of "param"
    param_idx_pos = parts.index("param")

    # Patient name is everything before "param"
    patient_parts = parts[:param_idx_pos]
    patient_name = "#".join(patient_parts)  # Restore # separator

    # Parameter index is right after "param"
    param_idx = int(parts[param_idx_pos + 1])

    # Scenario is everything after param index
    scenario_name = "_".join(parts[param_idx_pos + 2 :])

    return patient_name, param_idx, scenario_name


class T1DMOpenAPSParameterTuning(OpenAPSParameterTuningBase):
    """
    T1DM-specific OpenAPS parameter tuning implementation.

    This class extends OpenAPSParameterTuningBase to provide T1DM patient-specific
    parameter tuning using the ORef0 controller.
    """

    def create_virtual_patient_id(
        self, patient_name: str, param_idx: int, scenario_name: str
    ) -> str:
        """
        Create a virtual patient ID that encodes patient, parameter index, and scenario.

        Args:
            patient_name: Patient name (e.g., "adult#001")
            param_idx: Parameter set index
            scenario_name: Scenario name (e.g., "NO_MEAL")

        Returns:
            Virtual patient ID (e.g., "adult_001_param_0_NO_MEAL")
        """
        base_name = patient_name.replace("#", "_")
        return f"{base_name}_param_{param_idx}_{scenario_name}"

    @staticmethod
    def simulation_loop(virtual_patient_id, patient_map):
        """
        Simulation loop for T1DM patient with ORef0 controller.

        This implements the closed-loop control simulation for Type 1 Diabetes patients
        using the OpenAPS ORef0 algorithm.

        Args:
            virtual_patient_id: Virtual patient ID
            patient_map: Dictionary mapping virtual patient IDs to (patient_name, profile, scenario)

        Returns:
            Tuple of (t, BG, CHO, insulin, target_bg)
        """
        patient_name, _, scenario_name = parse_virtual_patient_id(virtual_patient_id)
        scenario = Scenario[scenario_name]

        p = T1DMPatient.withName(patient_name)
        ctrl = ORefZeroController(timeout=30000)  # TODO: remove this when not in debug

        profile = patient_map[virtual_patient_id][1]

        if profile is not None:
            profile["carb_ratio"] = p.carb_ratio
            profile["current_basal"] = p.basal * 60  # U/min to U/h

        if not ctrl.initialize_patient(virtual_patient_id, profile=profile):
            logger.error(
                f"Failed to initialize controller for patient {virtual_patient_id}"
            )
            return None

        t = []
        CHO = []
        insulin = []
        BG = []

        while p.t_elapsed < scenario.max_t:
            carb = scenario.get_carb(p.t_elapsed, p._params.BW)

            ctrl_obs = CtrlObservation(p.observation.Gsub)

            if p.observation.Gsub < 39:
                print("Patient is dead")
                break

            ctrl_action = ctrl.policy(
                observation=ctrl_obs,
                reward=0,
                done=False,
                patient_name=virtual_patient_id,
                meal=carb,
                time=p.t,
            )

            ins = ctrl_action.basal + ctrl_action.bolus
            act = Action(insulin=ins, CHO=carb)  # U/min
            p.step(act)

            t.append(p.t_elapsed)
            CHO.append(act.CHO)
            insulin.append(act.insulin)
            BG.append(p.observation.Gsub)
            print(
                f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {p.observation.Gsub}, CHO: {carb}, Insulin: {ins}\033[0m"
            )

        return t, BG, CHO, insulin, ctrl.target_bg

    def get_patients_by_group(self) -> Dict[PatientType, List[str]]:
        """
        Define T1DM patients to use for parameter tuning.

        Following paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
        Resplit patients into 2 groups: children (7-12), adults (16-70)

        Returns:
            Dictionary mapping PatientType to list of patient names
        """
        return {
            # PatientType.CHILD: ["child#002", "child#008", "child#010"],
            # PatientType.ADULT: ["adolescent#003", "adult#006", "adult#009"],
            PatientType.ADULT: ["adult#007"],
        }

    def get_scenarios(self) -> List[Scenario]:
        """
        Define scenarios to run for T1DM patients.

        Returns:
            List of Scenario objects
        """
        return [Scenario.ONE_DAY]

    def generate_profiles_by_group(self, group: PatientType) -> List[Dict[str, Any]]:
        """
        Generate OpenAPS parameter profiles for T1DM patient groups.

        This method creates parameter combinations based on literature values
        for different age groups.

        Args:
            group: Patient type group (CHILD or ADULT)

        Returns:
            List of profile dictionaries with OpenAPS parameters
        """
        target_bg = 100  # mg/dL
        min_bg = 90  # mg/dL
        max_bg = target_bg * 2 - min_bg  # mg/dL

        # Define parameter ranges for each age group based on literature
        patient_group_default_profiles = {
            PatientType.CHILD: {
                "sens": 150,
                "dia": 7,
                "carb_ratio": 30,
                "max_iob": 3,  # from paper, max 20, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
                "max_basal": 4,  # from paper, max 10
                "max_daily_basal": 0.9,  # from paper
                "max_bg": max_bg,
                "min_bg": min_bg,
                "maxCOB": 120,  # from oref0 code
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 150}]},
                "min_5m_carbimpact": 8,  # from paper and oref0 code
            },
            PatientType.ADULT: {
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
            },
        }

        parameter_group = {
            PatientType.CHILD: {
                "sens": {"step_count": 3, "range": (50, 100)},  # 1:50 to 1:100, gpt
                "dia": {"step_count": 3, "range": (5, 8)},  # DIA 5 to 8 hours, from paper
            },
            PatientType.ADULT: {
                "sens": {"step_count": 1, "range": (30, 50)},  # ISF 1:30 to 1:50, gpt
                "dia": {"step_count": 2, "range": (5, 8)},  # DIA 5 to 8 hours, from paper
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
        for values in value_combinations:
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

    def generate_patient_map_and_simulation_config(
        self,
        run_dir: Path,
    ) -> tuple[Dict[str, tuple[str, Dict[str, Any], Scenario]], List[tuple[str, Path]]]:
        """
        Generate patient map and simulation configs for T1DM OpenAPS parameter tuning.

        Args:
            run_dir: Directory to save simulation results

        Returns:
            Tuple of (patient_map, simulation_configs)
            - patient_map: Dictionary mapping virtual_patient_id -> (patient_name, profile, scenario)
            - simulation_configs: List of tuples (virtual_patient_id, img_save_dir)
        """
        # Get experiment configuration
        patients_by_group = self.get_patients_by_group()
        scenarios = self.get_scenarios()

        # Generate profiles for each patient group
        patient_profiles_by_group = {}
        for group in patients_by_group.keys():
            patient_profiles_by_group[group] = self.generate_profiles_by_group(group)

        # Build patient map and simulation configs
        patient_map = {}
        simulation_configs = []

        for group in patients_by_group.keys():
            profiles = patient_profiles_by_group[group]

            for patient_name in patients_by_group[group]:
                for param_idx, profile in enumerate(profiles):
                    for scenario in scenarios:
                        virtual_patient_id = self.create_virtual_patient_id(
                            patient_name, param_idx, scenario.name
                        )
                        patient_map[virtual_patient_id] = (patient_name, profile, scenario)

                        # Create patient-specific directory
                        patient_folder = patient_name.replace("#", "_")
                        patient_dir = run_dir / patient_folder / scenario.value
                        patient_dir.mkdir(exist_ok=True, parents=True)
                        simulation_configs.append(
                            (
                                virtual_patient_id,
                                patient_dir,
                            )
                        )

        return patient_map, simulation_configs


if __name__ == "__main__":
    """
    T1DM OpenAPS parameter tuning experiment.

    Following paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    Resplit patients into 2 groups: children (7-12), adults (16-70)
    Selected patients:
        child: child#002, child#008, child#010
        adult: adolescent#003, adult#006, adult#009

    Goal: Find optimal parameter set for each patient that achieves
    realistic time-in-range distribution following the paper.
    """
    # Create and run the experiment
    experiment = T1DMOpenAPSParameterTuning(output_dir=result_dir)
    results = experiment.run()

    logger.info(f"Experiment completed with {len(results)} results")
