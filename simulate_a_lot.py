from simglucose.patient.t1dpatient import T1DPatient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import fire
import logging
from datetime import datetime, timedelta
import pkg_resources
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Named tuples for actions and observations
Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["CGM"])  # Changed from Gsub to CGM

# Meal parameters
MEAL_WINDOWS = {
    "breakfast": {"start": 7*60, "end": 9*60, "mean": 8*60, "std": 30},  # 7-9 AM
    "lunch": {"start": 12*60, "end": 14*60, "mean": 13*60, "std": 30},   # 12-2 PM
    "dinner": {"start": 18*60, "end": 20*60, "mean": 19*60, "std": 30},  # 6-8 PM
    "snack_morning": {"start": 10*60, "end": 11*60, "mean": 10.5*60, "std": 15},  # 10-11 AM
    "snack_afternoon": {"start": 15*60, "end": 16*60, "mean": 15.5*60, "std": 15},  # 3-4 PM
}

# Carb content ranges (grams)
CARB_RANGES = {
    "breakfast": {"min": 30, "max": 70, "mean": 50, "std": 10},
    "lunch": {"min": 40, "max": 90, "mean": 65, "std": 15},
    "dinner": {"min": 50, "max": 100, "mean": 75, "std": 15},
    "snack_morning": {"min": 10, "max": 30, "mean": 20, "std": 5},
    "snack_afternoon": {"min": 10, "max": 30, "mean": 20, "std": 5},
}

# Time conversions
def minutes_to_time_str(minutes):
    """Convert minutes since midnight to HH:MM format"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def generate_meal_schedule(patient_name: str) -> Dict[str, Dict]:
    """
    Generate a realistic meal schedule for one day
    
    Args:
        patient_name: Patient identifier, affects carb scaling for children
        
    Returns:
        Dictionary with meal times and carb amounts
    """
    meals = {}
    
    # Determine if patient is a child (scale carbs accordingly)
    is_child = "child#" in patient_name.lower()
    carb_scale = 0.5 if is_child else 1.0
    
    # Decide which meals to include
    has_breakfast = random.random() < 0.85  # 85% chance of having breakfast
    has_lunch = random.random() < 0.9  # 90% chance of having lunch
    has_dinner = random.random() < 0.95  # 95% chance of having dinner
    has_morning_snack = random.random() < 0.3  # 30% chance of morning snack
    has_afternoon_snack = random.random() < 0.4  # 40% chance of afternoon snack
    
    # Generate meal times and carb amounts
    if has_breakfast:
        breakfast_time = int(np.random.normal(
            MEAL_WINDOWS["breakfast"]["mean"], 
            MEAL_WINDOWS["breakfast"]["std"]
        ))
        breakfast_carbs = int(np.random.normal(
            CARB_RANGES["breakfast"]["mean"],
            CARB_RANGES["breakfast"]["std"]
        ) * carb_scale)
        breakfast_carbs = max(CARB_RANGES["breakfast"]["min"] * carb_scale, 
                             min(CARB_RANGES["breakfast"]["max"] * carb_scale, breakfast_carbs))
        meals["breakfast"] = {
            "time": breakfast_time,
            "time_str": minutes_to_time_str(breakfast_time),
            "carbs": breakfast_carbs
        }
    
    if has_lunch:
        lunch_time = int(np.random.normal(
            MEAL_WINDOWS["lunch"]["mean"], 
            MEAL_WINDOWS["lunch"]["std"]
        ))
        lunch_carbs = int(np.random.normal(
            CARB_RANGES["lunch"]["mean"],
            CARB_RANGES["lunch"]["std"]
        ) * carb_scale)
        lunch_carbs = max(CARB_RANGES["lunch"]["min"] * carb_scale, 
                         min(CARB_RANGES["lunch"]["max"] * carb_scale, lunch_carbs))
        meals["lunch"] = {
            "time": lunch_time,
            "time_str": minutes_to_time_str(lunch_time),
            "carbs": lunch_carbs
        }
    
    if has_dinner:
        dinner_time = int(np.random.normal(
            MEAL_WINDOWS["dinner"]["mean"], 
            MEAL_WINDOWS["dinner"]["std"]
        ))
        dinner_carbs = int(np.random.normal(
            CARB_RANGES["dinner"]["mean"],
            CARB_RANGES["dinner"]["std"]
        ) * carb_scale)
        dinner_carbs = max(CARB_RANGES["dinner"]["min"] * carb_scale, 
                          min(CARB_RANGES["dinner"]["max"] * carb_scale, dinner_carbs))
        meals["dinner"] = {
            "time": dinner_time,
            "time_str": minutes_to_time_str(dinner_time),
            "carbs": dinner_carbs
        }
    
    if has_morning_snack and has_breakfast:  # Only have morning snack if had breakfast
        snack_time = int(np.random.normal(
            MEAL_WINDOWS["snack_morning"]["mean"], 
            MEAL_WINDOWS["snack_morning"]["std"]
        ))
        snack_carbs = int(np.random.normal(
            CARB_RANGES["snack_morning"]["mean"],
            CARB_RANGES["snack_morning"]["std"]
        ) * carb_scale)
        snack_carbs = max(CARB_RANGES["snack_morning"]["min"] * carb_scale, 
                         min(CARB_RANGES["snack_morning"]["max"] * carb_scale, snack_carbs))
        meals["snack_morning"] = {
            "time": snack_time,
            "time_str": minutes_to_time_str(snack_time),
            "carbs": snack_carbs
        }
    
    if has_afternoon_snack and has_lunch:  # Only have afternoon snack if had lunch
        snack_time = int(np.random.normal(
            MEAL_WINDOWS["snack_afternoon"]["mean"], 
            MEAL_WINDOWS["snack_afternoon"]["std"]
        ))
        snack_carbs = int(np.random.normal(
            CARB_RANGES["snack_afternoon"]["mean"],
            CARB_RANGES["snack_afternoon"]["std"]
        ) * carb_scale)
        snack_carbs = max(CARB_RANGES["snack_afternoon"]["min"] * carb_scale, 
                         min(CARB_RANGES["snack_afternoon"]["max"] * carb_scale, snack_carbs))
        meals["snack_afternoon"] = {
            "time": snack_time,
            "time_str": minutes_to_time_str(snack_time),
            "carbs": snack_carbs
        }
    
    return meals

def get_random_patient() -> T1DPatient:
    """Get a random patient from the available patients"""
    # Read available patient parameters
    patient_params = pd.read_csv(pkg_resources.resource_filename(
        "simglucose", "params/vpatient_params.csv"
    ))
    
    # Select a random patient
    patient_id = random.choice(patient_params['Name'].tolist())
    
    return T1DPatient.withName(patient_id)

def simulate_day(
    patient: T1DPatient, 
    controller_type: str = 'pid', 
    meals: Optional[Dict] = None,
    bg_noise_std: float = 2.0,  # Standard deviation for BG measurement noise in mg/dL
    silent: bool = False
) -> Dict:
    """
    Simulate a full day (24 hours = 1440 minutes) for a given patient
    
    Args:
        patient: T1DPatient object
        controller_type: 'pid' or 'bb' for controller selection
        meals: Optional dictionary with meal times and carbs. If None, will generate random meals.
        bg_noise_std: Standard deviation for blood glucose measurement noise
        silent: If True, suppress logging output
    
    Returns:
        Dictionary with simulation results
    """
    # Generate random meals if not provided
    if meals is None:
        meals = generate_meal_schedule(patient.name)
    
    # Select controller
    if controller_type.lower() == 'pid':
        ctrl = PIDController()
        if not silent:
            logger.info(f"Using PID Controller for patient {patient.name}")
    else:
        ctrl = BBController()
        if not silent:
            logger.info(f"Using Basal-Bolus Controller for patient {patient.name}")
    
    # Check if patient is a child (for logging)
    is_child = "child#" in patient.name.lower()
    if not silent and is_child:
        logger.info(f"Patient is a child, carbs are scaled down by 50%")
    
    # Initialize simulation
    basal = patient._params.u2ss * patient._params.BW / 6000  # U/min
    BG = []
    insulin = []
    insulin_events = []
    last_insulin = None
    
    # Reset patient state
    patient.reset()
    
    # Set up simulation start time (start at midnight)
    start_time = datetime.combine(datetime.now().date(), datetime.min.time())
    current_time = start_time
    
    # Create meal events sorted by time
    meal_events = []
    for meal_name, meal_data in meals.items():
        meal_events.append({
            "time": meal_data["time"],
            "carbs": meal_data["carbs"],
            "name": meal_name
        })
    
    # Sort meal events by time
    meal_events.sort(key=lambda x: x["time"])
    
    # Create list for carb events (stored as dicts rather than timeline)
    carb_events = []
    
    # Simulate for 24 hours (1440 minutes) with 1-minute resolution
    for minute in range(1440):
        # Check if there's a meal at this time
        carb = 0
        meal_name = None
        for meal in meal_events:
            if meal["time"] == minute:
                carb = meal["carbs"]
                meal_name = meal["name"]
                if not silent:
                    logger.info(f"Meal at {minutes_to_time_str(minute)}: {meal_name} with {carb}g carbs")
                
                # Add to carb events
                carb_events.append({
                    "time": minute,
                    "time_str": minutes_to_time_str(minute),
                    "carbs": carb,
                    "meal_type": meal_name
                })
                break
        
        # Get true BG value without noise for controller input
        true_bg = patient.observation.Gsub
        
        # Add noise to get CGM reading
        bg_with_noise = true_bg + np.random.normal(0, bg_noise_std)
        bg_with_noise = max(0, bg_with_noise)  # Ensure no negative values
        
        # Use observation with CGM instead of Gsub
        ctrl_obs = Observation(CGM=true_bg)  # Use true BG for control decisions
        
        # Get control action
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient.name,
            meal=carb,
            time=current_time
        )
        
        # Apply insulin from controller
        ins = ctrl_action.basal + ctrl_action.bolus
        
        # Take action
        act = Action(insulin=ins, CHO=carb)
        
        # Record data (BG and insulin)
        BG.append(bg_with_noise)
        insulin.append(act.insulin)
        
        # Record insulin as events when it changes
        if last_insulin is None or last_insulin != act.insulin:
            insulin_events.append({
                "time": minute,
                "time_str": minutes_to_time_str(minute),
                "insulin": act.insulin
            })
            last_insulin = act.insulin
        
        # Step patient
        patient.step(act)
        
        # Update simulation time
        current_time += timedelta(minutes=1)
    
    # Prepare results
    results = {
        "patient_id": patient.name,
        "meals": meals,
        "carb_events": carb_events,
        "bg_mgdl": BG,
        "insulin_events": insulin_events,
        "controller": controller_type,
        "simulation_date": start_time.strftime("%Y-%m-%d")
    }
    
    return results

def process_sample(
    i: int,
    n: int,
    controller_type: str,
    bg_noise_std: float,
    silent: bool,
    plot_samples: int
) -> Dict:
    """Process a single sample for parallel execution"""
    # Get random patient
    patient = get_random_patient()
    
    # Generate meal schedule with patient name to account for child scaling
    meals = generate_meal_schedule(patient.name)
    
    # Simulate day
    results = simulate_day(patient, controller_type, meals, bg_noise_std, silent)
    
    # Plot if requested (only for the first few samples)
    if plot_samples > 0 and i < plot_samples:
        plot_simulation(results, i)
    
    return results

def generate_dataset(
    n: int = 10, 
    output_file: str = "glucose_dataset.jsonl",
    controller_type: str = 'pid',
    plot_samples: int = 0,
    bg_noise_std: float = 2.0,  # Standard deviation for BG measurement noise
    n_jobs: int = -1,  # Number of parallel jobs (-1 = all cores)
    silent: bool = False  # Suppress logging output except for progress bar
):
    """
    Generate a dataset of n simulated days from random patients
    
    Args:
        n: Number of samples to generate
        output_file: Output file path (.jsonl)
        controller_type: Controller type ('pid' or 'bb')
        plot_samples: Number of random samples to plot (0 to disable)
        bg_noise_std: Standard deviation for BG measurement noise
        n_jobs: Number of parallel jobs (-1 to use all available cores)
        silent: If True, suppress logging output
    """
    if not silent:
        logger.info(f"Generating dataset with {n} samples using {controller_type} controller")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine number of jobs based on CPU cores
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = min(n_jobs, multiprocessing.cpu_count())
    
    if not silent:
        logger.info(f"Using {n_jobs} parallel processes")
    
    # Process samples in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(i, n, controller_type, bg_noise_std, silent, plot_samples) 
        for i in tqdm(range(n), desc="Generating samples", disable=silent)
    )
    
    # Save results
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    if not silent:
        logger.info(f"Dataset saved to {output_file}")

def events_to_timeline(events, length=1440, default_value=0, value_field=None):
    """
    Convert events list to a timeline array
    
    Args:
        events: List of event dictionaries with 'time' and value fields
        length: Length of the timeline to create
        default_value: Default value for the timeline
        value_field: Name of the field containing the value (if None, will try to detect)
        
    Returns:
        List representing the timeline with values at each point
    """
    # Create timeline with default values
    timeline = [default_value] * length
    
    # Sort events by time
    sorted_events = sorted(events, key=lambda x: x["time"])
    
    # Fill timeline based on events
    current_value = default_value
    for event in sorted_events:
        time = event["time"]
        # Get the value field (could be 'insulin', 'carbs', etc.)
        if value_field is None:
            value_field = next(key for key in event.keys() if key not in ["time", "time_str", "meal_type"])
        current_value = event[value_field]
        
        # Update all subsequent positions until next event
        for i in range(time, length):
            timeline[i] = current_value
            
    return timeline

def timeline_to_events(timeline, value_field="value"):
    """
    Convert a timeline array to a list of events
    
    Args:
        timeline: List of values
        value_field: Name of the field to store the value
        
    Returns:
        List of event dictionaries with 'time' and value fields
    """
    events = []
    current_value = None
    
    for i, value in enumerate(timeline):
        if current_value is None or value != current_value:
            events.append({
                "time": i,
                "time_str": minutes_to_time_str(i),
                value_field: value
            })
            current_value = value
            
    return events

def plot_simulation(data: Dict, sample_idx: int = 0):
    """Plot simulation results"""
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Create time axis (minutes since midnight)
    time_minutes = list(range(1440))  # 24 hours in minutes
    
    # Get times for x-axis labels
    time_labels = [minutes_to_time_str(m) for m in range(0, 1440, 60)]
    
    # Plot glucose
    ax[0].plot(time_minutes, data["bg_mgdl"])
    ax[0].set_ylabel("BG (mg/dL)")
    ax[0].set_title(f"Patient {data['patient_id']} - {data['simulation_date']}")
    
    # Add glucose reference lines
    ax[0].axhline(y=70, color='r', linestyle='-', alpha=0.3)
    ax[0].axhline(y=180, color='r', linestyle='-', alpha=0.3)
    ax[0].text(1, 72, "70 mg/dL", color='r')
    ax[0].text(1, 182, "180 mg/dL", color='r')
    
    # Plot carbs as bars at event times
    for event in data["carb_events"]:
        ax[1].bar(event["time"], event["carbs"], width=5)
        ax[1].text(event["time"], event["carbs"] + 5, 
                   f"{event['meal_type']}\n{event['carbs']}g", 
                   ha='center', va='bottom', fontsize=9)
    
    ax[1].set_ylabel("Carbs (g)")
    ax[1].set_ylim(0, max([e["carbs"] for e in data["carb_events"]], default=100) + 20)
    
    # Convert insulin events to timeline for plotting
    insulin_timeline = events_to_timeline(data["insulin_events"], value_field="insulin")
    
    # Plot insulin
    ax[2].plot(time_minutes, insulin_timeline)
    ax[2].set_ylabel("Insulin (U/min)")
    ax[2].set_xlabel("Time of Day")
    
    # Mark insulin events with points
    for event in data["insulin_events"]:
        ax[2].scatter(event["time"], event["insulin"], color='red', s=20)
    
    # Format x-axis
    ax[2].set_xticks(list(range(0, 1440, 60)))  # Every hour
    ax[2].set_xticklabels(time_labels)
    
    plt.tight_layout()
    plt.savefig(f"simulation_sample_{sample_idx}.png")
    plt.close()

def main(
    n: int = 10, 
    output: str = "glucose_dataset.jsonl",
    controller: str = "pid",
    plot: int = 2,
    noise: float = 2.0,  # Standard deviation for BG measurement noise
    jobs: int = -1,     # Number of parallel jobs (-1 = all cores)
    silent: bool = False  # Suppress logging output
):
    """
    Generate a dataset of simulated glucose data from T1D patients
    
    Args:
        n: Number of samples to generate
        output: Output file path (.jsonl)
        controller: Controller type ('pid' or 'bb')
        plot: Number of random samples to plot (0 to disable)
        noise: Standard deviation of glucose measurement noise (mg/dL)
        jobs: Number of parallel jobs (-1 to use all cores)
        silent: If True, suppress logging output except for progress bar
    """
    generate_dataset(
        n=n, 
        output_file=output, 
        controller_type=controller, 
        plot_samples=plot,
        bg_noise_std=noise,
        n_jobs=jobs,
        silent=silent
    )

if __name__ == "__main__":
    fire.Fire(main)