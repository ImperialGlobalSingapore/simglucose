"""
Simple example demonstrating how to use T1DM patient with ORef0 controller.

This script shows a basic simulation of a T1DM patient using the OpenAPS ORef0
algorithm for automated insulin delivery. It's meant as a reference implementation
and testing example.
"""

import logging
from pathlib import Path
from collections import namedtuple

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController

import sys

sys.path.append(str(Path(__file__).parent.parent))
from simglucose.simulation.scenario_simple import Scenario
from tests_controller.plot_utils import calculate_time_in_range, plot_and_show

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Named tuple for controller observation
CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


def run_patient_with_oref0(
    patient_name="adult#001",
    scenario=Scenario.ONE_DAY,
    profile=None,
    show_plot=True,
):
    """
    Run a simulation of a T1DM patient with ORef0 controller.

    Args:
        patient_name: Name of the patient (e.g., "adult#001", "child#001")
        scenario: Meal scenario to simulate (NO_MEAL, SINGLE_MEAL, ONE_DAY, THREE_DAY)
        profile: Optional ORef0 profile dict with parameters like sens, dia, carb_ratio, etc.
        show_plot: Whether to display the results plot

    Returns:
        dict: Time in range statistics
    """
    # Initialize patient
    p = T1DMPatient.withName(patient_name)
    logger.info(f"Patient {patient_name} initialized")

    # Initialize controller
    if profile:
        ctrl = ORefZeroController(
            current_basal=p.basal * 60,  # Convert U/min to U/h
            profile=profile,
        )
        logger.info("ORef0 controller initialized with custom profile")
    else:
        ctrl = ORefZeroController(
            current_basal=p.basal * 60,  # Convert U/min to U/h
        )
        logger.info("ORef0 controller initialized with default profile")

    # Storage for simulation data
    t = []
    CHO = []
    insulin = []
    BG = []

    # Run simulation
    logger.info(f"Starting simulation with scenario: {scenario.name}")
    while p.t_elapsed < scenario.max_t:
        # Get meal for current time
        carb = scenario.get_carb(p.t_elapsed, p._params.BW)

        # Create observation for controller
        ctrl_obs = CtrlObservation(p.observation.Gsub)

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")
            break

        # Get controller action
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )

        # Calculate total insulin (basal + bolus)
        ins = ctrl_action.basal + ctrl_action.bolus
        act = Action(insulin=ins, CHO=carb)

        # Record data
        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)

        # Step the patient simulation
        p.step(act)

        # Log progress (every hour)
        if p.t_elapsed % 60 == 0:
            logger.info(
                f"Time: {p.t_elapsed}min, BG: {p.observation.Gsub:.1f} mg/dL, "
                f"CHO: {carb}g, Insulin: {ins:.3f} U/min"
            )

    # Calculate time in range statistics
    time_in_range = calculate_time_in_range(BG)

    logger.info("\n=== Time in Range Results ===")
    for category, percentage in time_in_range.items():
        logger.info(f"{category}: {percentage*100:.1f}%")

    # Display plot if requested
    if show_plot:
        plot_and_show(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_bg,
            f"T1DM Patient {patient_name} with ORef0 - {scenario.name}",
        )

    return time_in_range


if __name__ == "__main__":
    # Example: Run with custom profile
    # refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    print("\n" + "=" * 60)
    print("Example: Adult patient with custom ORef0 profile")
    print("=" * 60)

    custom_profile = {
        "sens": 45,
        "dia": 7.0,
        "carb_ratio": 20,
        "max_iob": 12,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_daily_basal": 0.9,  # from paper
        "max_bg": 140,
        "min_bg": 90,
        "maxCOB": 120,  # from oref0 code
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 60}]},
        "min_5m_carbimpact": 8,  # from paper and oref0 code
    }

    run_patient_with_oref0(
        patient_name="adult#007",
        scenario=Scenario.ONE_DAY,
        profile=custom_profile,
        show_plot=True,
    )
