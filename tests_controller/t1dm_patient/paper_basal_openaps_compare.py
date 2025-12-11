import logging

from pathlib import Path
from datetime import datetime
from simglucose.patient.t1dm_patient import T1DMPatient, Action as PatientAction
from simglucose.controller.oref_zero_with_meal_bolus import (
    ORefZeroWithMealBolus,
    CtrlObservation,
)
from simglucose.controller.base import Controller, Action as CtrlAction
from simglucose.controller.meal_bolus_ctrller import MealAnnouncementBolusController
from glucose_control_analytics import TIRConfig, plot_bg_cho_iob_and_save_with_tir

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
test_patient_dir = img_dir / "paper" / "basal_openaps_compare"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_save_dir = test_patient_dir / timestamp
image_save_dir.mkdir(parents=True, exist_ok=True)


class T1DMBBController(Controller):
    """
    Basal-Bolus Controller using T1DM patient information.

    This controller follows the same logic as BBController but uses patient-specific
    parameters (CR, CF, u2ss, BW) directly from the T1DM patient JSON files
    instead of external CSV files.

    Basal rate: u2ss * BW / 6000 (U/min)
    Bolus: (carbs / CR) + (glucose > 150) * (glucose - target) / CF (U)

    Optionally supports meal announcement bolus via meal_schedule parameter.
    """

    def __init__(
        self,
        patient_name: str,
        target: float = 140,
        meal_schedule=None,
        release_time_before_meal: int = 10,
    ):
        """
        Initialize the basal-bolus controller for a T1DM patient.

        Args:
            patient_name: Name of the T1DM patient (e.g., "adult#001")
            target: Target blood glucose level (mg/dL)
            meal_schedule: List of tuples (time_minutes, carbs_grams), e.g.,
                          [(120, 50), (360, 75), (720, 60)]
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Percentage of error in carbohydrate estimation (e.g., 0.3 for +/- 30%)
            t_start: Patient simulation start time as datetime object (optional)
        """
        self._patient_params = T1DMPatient.withName(patient_name)._params
        self.patient_name = patient_name
        self.target = target

        self.CR = self._patient_params.CR  # Carbohydrate Ratio (g/U)
        self.CF = self._patient_params.CF  # Correction Factor (mg/dL/U)
        self.u2ss = self._patient_params.u2ss  # Steady-state insulin (pmol/(L*kg))
        self.BW = self._patient_params.BW  # Body weight (kg)

        self.release_time_before_meal = release_time_before_meal
        self._meal_schedule = meal_schedule if meal_schedule is not None else []

        logger.info(f"T1DMBBController initialized for {patient_name}")
        logger.info(f"  CR: {self.CR:.2f} g/U, CF: {self.CF:.2f} mg/dL/U")
        logger.info(f"  Basal: {self.u2ss * self.BW / 6000 * 60:.3f} U/h")
        if meal_schedule:
            logger.info(f"  Meal schedule: {meal_schedule}")

    def policy(self, observation, reward, done, **kwargs):
        """
        Compute basal-bolus action using the same formula as BBController.

        Args:
            observation: NamedTuple with CGM field (blood glucose in mg/dL)
            reward: Current reward (unused)
            done: Episode completion flag (unused)
            **kwargs: meal (g/min), sample_time (min), time (datetime for meal bolus)

        Returns:
            Action namedtuple with (basal, bolus) both in U/min
        """

        sample_time = kwargs.get("sample_time", 1)
        elapsed_time = kwargs.get("time", None)  # datetime for meal announcement bolus
        glucose = observation.CGM

        # Basal rate: u2ss * BW / 6000 (U/min)
        basal = self.u2ss * self.BW / 6000
        bolus = 0
        if elapsed_time is not None:
            elapsed_time = int(elapsed_time)

            target_meal_time = elapsed_time + self.release_time_before_meal

            for meal_time, meal_amount in self._meal_schedule:
                if meal_time == target_meal_time:
                    bolus = (meal_amount * sample_time) / self.CR + (glucose > 150) * (
                        glucose - self.target
                    ) / self.CF

        bolus = bolus / sample_time

        return CtrlAction(basal=basal, bolus=bolus)

    def reset(self):
        """Reset controller state."""
        pass


def run_patient_with_basal_bolus(
    patient_name="adult#001",
    save_plot=True,
    meal_time=360,  # Single meal at 6 hours (360 minutes)
    meal_amount=50,  # 50g carbs
    target=140,  # Target blood glucose (mg/dL)
    simulation_time=720,  # 12 hours
):
    """
    Run a simulation of a T1DM patient with Basal-Bolus controller.

    Args:
        patient_name: Name of the patient (e.g., "adult#001", "child#001")
        show_plot: Whether to display the results plot
        meal_time: Time in minutes when meal is delivered
        meal_amount: Amount of carbs in grams
        target: Target blood glucose level (mg/dL)
        simulation_time: Total simulation time in minutes

    Returns:
        dict: Time in range statistics
    """
    # Initialize patient
    p = T1DMPatient.withName(patient_name)
    logger.info(f"Patient {patient_name} initialized")

    # Initialize controller with patient-specific parameters
    ctrl = T1DMBBController(
        patient_name=patient_name,
        target=target,
        meal_schedule=[(meal_time, meal_amount)],
    )

    # Storage for simulation data
    t = []
    BG = []
    IOB = []
    CHO = []

    # Run simulation
    logger.info(f"Starting simulation for {simulation_time} minutes")
    while p.t_elapsed < simulation_time:
        # Deliver meal at meal_time
        carb = meal_amount if p.t_elapsed == meal_time else 0

        # Create observation for controller
        ctrl_obs = CtrlObservation(CGM=p.observation.Gsub)

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")
            break

        # Get controller action
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            sample_time=p.sample_time,
            time=p.t_elapsed,  # datetime for meal announcement bolus
        )

        # Calculate total insulin (basal + bolus)
        ins = ctrl_action.basal + ctrl_action.bolus  # U/min
        act = PatientAction(CHO=carb, insulin=ins)

        # Record data
        t.append(p.t_elapsed)
        BG.append(p.observation.Gsub)
        CHO.append(carb)
        IOB.append(p.get_iob())

        # Step the patient simulation
        p.step(act)

        # Log progress (every hour)
        if p.t_elapsed % 60 == 0:
            logger.info(
                f"Time: {p.t_elapsed}min, BG: {p.observation.Gsub:.1f} mg/dL, "
                f"CHO: {carb}g, Insulin: {ins:.3f} U/min"
            )

    # Calculate time in range statistics
    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    logger.info("\n=== Time in Range Results ===")
    for category, percentage in time_in_range.items():
        logger.info(f"{category.value}: {percentage:.1f}%")

    file_name = (
        image_save_dir
        / f"t1dm_{patient_name}_basal_bolus_{meal_amount}g_at_{meal_time}min.png"
    )
    # Display plot if requested
    if save_plot:
        plot_bg_cho_iob_and_save_with_tir(
            t,
            BG,
            CHO,
            IOB,
            target,
            file_name,
            time_in_range,
            tir_config,
        )

    return time_in_range, t, BG, IOB, CHO


def run_patient_with_oref0_bolus(
    patient_name="adult#001",
    profile=None,
    save_plot=True,
    meal_time=360,  # Single meal at 6 hours (360 minutes)
    meal_amount=50,  # 50g carbs
    release_time_before_meal=10,  # minutes before meal to release bolus
    carb_estimation_error=0.3,  # +/- percentage of carb estimation error
    simulation_time=720,  # 12 hours
):
    """
    Run a simulation of a T1DM patient with ORef0 + Meal Bolus controller.

    Args:
        patient_name: Name of the patient (e.g., "adult#001", "child#001")
        profile: Optional ORef0 profile dict with parameters like sens, dia, carb_ratio, etc.
        show_plot: Whether to display the results plot
        meal_time: Time in minutes when meal is delivered
        meal_amount: Amount of carbs in grams
        release_time_before_meal: Minutes before meal to release bolus
        carb_estimation_error: Percentage of error in carb estimation
        simulation_time: Total simulation time in minutes

    Returns:
        dict: Time in range statistics
    """
    # Initialize patient
    p = T1DMPatient.withName(patient_name)
    logger.info(f"Patient {patient_name} initialized")
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    # Single meal schedule
    meal_schedule = [(meal_time, meal_amount)]
    logger.info(f"Meal schedule: {meal_schedule}")

    # Initialize combined controller
    combined_ctrl = ORefZeroWithMealBolus(
        patient_name=patient_name,
        server_url="http://localhost:3000",
        timeout=3000,  # TODO: DEBUG only
        profile=profile,
        meal_schedule=meal_schedule,
        carb_factor=(
            profile["carb_ratio"] if profile and "carb_ratio" in profile else 10
        ),
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
        t_start=p.t_start,  # Pass patient start time for meal bolus calculation
    )

    # Initialize patient on the controller
    if not combined_ctrl.initialize():
        raise ValueError("Failed to initialize ORefZero controller")

    logger.info("ORefZeroWithMealBolus controller initialized")

    # Storage for simulation data
    t = []
    BG = []
    IOB = []
    CHO = []

    # Run simulation
    logger.info(f"Starting simulation for {simulation_time} minutes")
    while p.t_elapsed < simulation_time:
        # Deliver meal at meal_time
        carb = meal_amount if p.t_elapsed == meal_time else 0

        # Create observation for controller (just CGM for ORefZeroWithMealBolus)
        ctrl_obs = CtrlObservation(CGM=p.observation.Gsub)

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")
            break

        # Get combined controller action (ORefZero basal + meal bolus)
        combined_action = combined_ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            time=p.t,  # datetime for both controllers (meal_bolus calculates elapsed from t_start)
        )

        # Calculate total insulin (basal + bolus)
        ins = combined_action.basal + combined_action.bolus  # U/min
        act = PatientAction(insulin=ins, CHO=carb)

        # Record data
        t.append(p.t_elapsed)
        BG.append(p.observation.Gsub)
        IOB.append(p.get_iob())
        CHO.append(carb)

        # Step the patient simulation
        p.step(act)

        # Log progress (every hour)
        if p.t_elapsed % 60 == 0:
            logger.info(
                f"Time: {p.t_elapsed}min, BG: {p.observation.Gsub:.1f} mg/dL, "
                f"CHO: {carb}g, Insulin: {ins:.3f} U/min"
            )

    # Calculate time in range statistics
    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    logger.info("\n=== Time in Range Results ===")
    for category, percentage in time_in_range.items():
        logger.info(f"{category.value}: {percentage:.1f}%")

    # Display plot if requested
    file_name = (
        image_save_dir
        / f"t1dm_{patient_name}_oref0_bolus_{meal_amount}g_at_{meal_time}min.png"
    )
    if save_plot:
        plot_bg_cho_iob_and_save_with_tir(
            t,
            BG,
            CHO,
            IOB,
            combined_ctrl.target_bg,
            file_name,
            time_in_range,
            tir_config,
        )

    return time_in_range, t, BG, IOB, CHO


if __name__ == "__main__":
    # Common simulation parameters
    patient_name = "adult#007"
    meal_time = 20  # Meal at 20 minutes
    meal_amount = 75  # 75g carbs
    simulation_time = 720  # 12 hours

    # =========================================================================
    # Example 1: Run with Basal-Bolus Controller
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Adult patient with Basal-Bolus Controller")
    print("=" * 60)

    run_patient_with_basal_bolus(
        patient_name=patient_name,
        save_plot=True,
        meal_time=meal_time,
        meal_amount=meal_amount,
        target=140,
        simulation_time=simulation_time,
    )

    exit()
    # =========================================================================
    # Example 2: Run with ORef0 + Meal Bolus Controller
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Adult patient with ORef0 + Meal Bolus Controller")
    print("=" * 60)

    # Custom ORef0 profile (refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913)
    custom_profile = {
        "sens": 45,
        "dia": 7.0,
        "carb_ratio": 10,  # changed later from patient
        "max_iob": 12,  # from paper, max 30
        "max_basal": 4,  # from paper, max 10
        "max_daily_basal": 0.9,  # from paper
        "max_bg": 140,
        "min_bg": 90,
        "maxCOB": 120,  # from oref0 code
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 45}]},
        "min_5m_carbimpact": 8,  # from paper and oref0 code
    }

    run_patient_with_oref0_bolus(
        patient_name=patient_name,
        profile=custom_profile,
        save_plot=True,
        meal_time=meal_time,
        meal_amount=meal_amount,
        release_time_before_meal=10,
        carb_estimation_error=0,  # 30% carb estimation error
        simulation_time=simulation_time,
    )
