import logging
import matplotlib
from pathlib import Path
from collections import namedtuple
from typing import Any, Dict, List
import numpy as np
from itertools import product

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero_with_meal_bolus import ORefZeroWithMealBolus, CtrlObservation
from glucose_control_analytics import PatientType, OpenAPSParameterTuningBase

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent
result_dir = parent_folder / "imgs" / "oref0_parameter_tuning"


class T1DMOpenAPSParameterTuning(OpenAPSParameterTuningBase):
    """
    T1DM-specific OpenAPS parameter tuning implementation.

    This class extends OpenAPSParameterTuningBase to provide T1DM patient-specific
    parameter tuning using the ORef0 controller.
    """

    @staticmethod
    def parse_virtual_patient_id(virtual_patient_id: str) -> tuple[str, int, int]:
        """
        Parse a virtual patient ID to extract patient name, parameter index, and carb amount.

        Args:
            virtual_patient_id: Virtual patient ID (e.g., "adult_001_param_0_carb_50")

        Returns:
            Tuple of (patient_name, param_idx, carb_amount)
            e.g., ("adult#001", 0, 50)
        """
        parts = virtual_patient_id.split("_")

        # Find the index of "param"
        param_idx_pos = parts.index("param")

        # Patient name is everything before "param"
        patient_parts = parts[:param_idx_pos]
        patient_name = "#".join(patient_parts)  # Restore # separator

        # Parameter index is right after "param"
        param_idx = int(parts[param_idx_pos + 1])

        # Carb amount is after "carb"
        carb_idx_pos = parts.index("carb")
        carb_amount = int(parts[carb_idx_pos + 1])

        return patient_name, param_idx, carb_amount

    def create_virtual_patient_id(
        self, patient_name: str, param_idx: int, carb_amount: int
    ) -> str:
        """
        Create a virtual patient ID that encodes patient, parameter index, and carb amount.

        Args:
            patient_name: Patient name (e.g., "adult#001")
            param_idx: Parameter set index
            carb_amount: Amount of carbs in grams (e.g., 50)

        Returns:
            Virtual patient ID (e.g., "adult_001_param_0_carb_50")
        """
        base_name = patient_name.replace("#", "_")
        return f"{base_name}_param_{param_idx}_carb_{carb_amount}"

    @staticmethod
    def simulation_loop(virtual_patient_id, patient_map):
        """
        Simulation loop for T1DM patient with ORef0 + Meal Bolus controller.

        This implements the closed-loop control simulation for Type 1 Diabetes patients
        using the OpenAPS ORef0 algorithm with meal announcement bolus.

        Args:
            virtual_patient_id: Virtual patient ID
            patient_map: Dictionary mapping virtual patient IDs to (patient_name, profile, carb_amount)

        Returns:
            Tuple of (t, BG, CHO, insulin, target_bg)
        """
        patient_name, _, carb_amount = T1DMOpenAPSParameterTuning.parse_virtual_patient_id(virtual_patient_id)

        p = T1DMPatient.withName(patient_name)

        profile = patient_map[virtual_patient_id][1]

        if profile is not None:
            profile["carb_ratio"] = p.carb_ratio
            profile["current_basal"] = p.basal * 60  # U/min to U/h

        # Fixed simulation time and meal delivery at 20 minutes
        max_simulation_time = 1450  # 24 hours + 10 minutes
        meal_time = 20  # Deliver meal at 20 minutes

        # Create meal schedule - single meal at meal_time
        meal_schedule = [(meal_time, carb_amount)]

        ctrl = ORefZeroWithMealBolus(
            patient_name=virtual_patient_id,
            server_url="http://localhost:3000",
            timeout=30000,  # TODO: remove timeout when not in debug
            profile=profile,
            meal_schedule=meal_schedule,
            carb_factor=profile["carb_ratio"] if profile and "carb_ratio" in profile else 10,
            release_time_before_meal=10,  # Release bolus 10 minutes before meal
            carb_estimation_error=0.3,  # 30% carb estimation error
            sample_time=p.SAMPLE_TIME,
            t_start=p.t_start,
        )

        if not ctrl.initialize():
            logger.error(
                f"Failed to initialize controller for patient {virtual_patient_id}"
            )
            return None

        t = []
        CHO = []
        insulin = []
        BG = []

        while p.t_elapsed < max_simulation_time:
            # Deliver meal at exactly 20 minutes
            carb = carb_amount if int(p.t_elapsed) == meal_time else 0

            ctrl_obs = CtrlObservation(CGM=p.observation.Gsub)

            if p.observation.Gsub < 39:
                print("Patient is dead")
                break

            ctrl_action = ctrl.policy(
                observation=ctrl_obs,
                reward=0,
                done=False,
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
                "dia": {
                    "step_count": 3,
                    "range": (5, 8),
                },  # DIA 5 to 8 hours, from paper
            },
            PatientType.ADULT: {
                "sens": {"step_count": 1, "range": (30, 50)},  # ISF 1:30 to 1:50, gpt
                "dia": {
                    "step_count": 2,
                    "range": (5, 8),
                },  # DIA 5 to 8 hours, from paper
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
    ) -> tuple[Dict[str, tuple[str, Dict[str, Any], int]], List[tuple[str, Path]]]:
        """
        Generate patient map and simulation configs for T1DM OpenAPS parameter tuning.

        Args:
            run_dir: Directory to save simulation results

        Returns:
            Tuple of (patient_map, simulation_configs)
            - patient_map: Dictionary mapping virtual_patient_id -> (patient_name, profile, carb_amount)
            - simulation_configs: List of tuples (virtual_patient_id, img_save_dir)
        """
        # Get experiment configuration
        patients_by_group = self.get_patients_by_group()

        # Generate carb amounts from 10 to 100
        carb_amounts = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]

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
                # Create single patient-specific directory
                patient_folder = patient_name.replace("#", "_")
                patient_dir = run_dir / patient_folder
                patient_dir.mkdir(exist_ok=True, parents=True)

                for param_idx, profile in enumerate(profiles):
                    for carb_amount in carb_amounts:
                        virtual_patient_id = self.create_virtual_patient_id(
                            patient_name, param_idx, carb_amount
                        )
                        patient_map[virtual_patient_id] = (
                            patient_name,
                            profile,
                            carb_amount,
                        )

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
