from collections import namedtuple
import numpy as np

from simglucose.controller.base import Controller, Action
import logging
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json


logger = logging.getLogger(__name__)


class ORefZeroController:
    """
    OpenAPS oref0 controller that communicates with Node.js server
    for insulin dosage recommendations
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        patient_profile: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        frequency=5,  # in minutes
    ):
        """
        Initialize the ORefZero controller

        Args:
            server_url: URL of the Node.js OpenAPS server
            patient_profile: Custom patient profile, uses default if None
            timeout: Request timeout in seconds
        """
        self.frequency = frequency
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Patient state tracking
        self.patient_profiles = {}  # patientId -> profile mapping
        self.last_glucose_time = {}  # patientId -> last glucose timestamp
        self.glucose_history = {}  # patientId -> list of glucose readings
        self.meal_history = {}  # patientId -> list of meal entries
        self.pump_history = {}  # patientId -> list of pump events

        # Default patient profile (based on your server's test cases)
        self.default_profile = patient_profile or {
            "carb_ratio": 10,
            "sens": 50,
            "dia": 6,
            "max_bg": 120,
            "min_bg": 120,
            "max_basal": 4.0,
            "current_basal": 1.0,
            "max_iob": 6.0,
            "maxCOB": 100,
            "max_daily_safety_multiplier": 4,
            "current_basal_safety_multiplier": 5,
            "autosens_max": 2,
            "autosens_min": 0.5,
            "autosens": False,
            "enableSMB_with_bolus": True,
            "enableSMB_with_COB": True,
            "curve": "rapid-acting",
            "insulinPeakTime": 75,
            "basalprofile": [
                {"minutes": 0, "rate": 1.0, "start": "00:00:00", "i": 0},
                {"minutes": 360, "rate": 0.8, "start": "06:00:00", "i": 1},
                {"minutes": 720, "rate": 1.2, "start": "12:00:00", "i": 2},
                {"minutes": 1080, "rate": 0.9, "start": "18:00:00", "i": 3},
            ],
            "isfProfile": {
                "first": 1,
                "sensitivities": [
                    {"endOffset": 1440, "offset": 0, "x": 0, "sensitivity": 50, "start": "00:00:00", "i": 0}
                ],
                "user_preferred_units": "mg/dL",
                "units": "mg/dL",
            },
        }

        logger.info(f"ORefZero Controller initialized with server: {self.server_url}")

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to the server with error handling"""
        url = f"{self.server_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PATCH":
                response = self.session.patch(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise Exception("Server request timed out")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to {url}")
            raise Exception("Could not connect to OpenAPS server")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}: {e.response.text}")
            raise Exception(f"Server returned error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            raise

    def _initialize_patient(self, patient_name: str, profile: Optional[Dict] = None) -> bool:
        patient_name = patient_name.replace("#", "")
        """Initialize patient on the server"""
        if patient_name in self.patient_profiles:
            logger.debug(f"Patient {patient_name} already initialized")
            return True

        patient_profile = profile or self.default_profile.copy()

        # Prepare initialization data
        init_data = {
            "profile": patient_profile,
            "initialData": {"glucoseHistory": [], "pumpHistory": [], "carbHistory": []},
            "settings": {"timezone": "UTC", "historyRetentionHours": 24, "autoCleanup": True},
        }

        try:
            response = self._make_request("POST", f"/patients/{patient_name}/initialize", init_data)
            self.patient_profiles[patient_name] = patient_profile
            self.glucose_history[patient_name] = []
            self.meal_history[patient_name] = []
            self.pump_history[patient_name] = []
            self.last_glucose_time[patient_name] = None

            logger.info(f"Patient {patient_name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize patient {patient_name}: {str(e)}")
            return False

    def _convert_time_to_timestamp(self, time_obj) -> str:
        """Convert time object to ISO timestamp string"""
        if isinstance(time_obj, datetime):
            return time_obj.isoformat() + "Z"
        elif isinstance(time_obj, str):
            # Assume it's already a valid timestamp
            return time_obj
        else:
            # Default to current time
            return datetime.utcnow().isoformat() + "Z"

    def _prepare_new_data(self, patient_name: str, glucose: float, meal: float, timestamp: str) -> Dict[str, Any]:
        """Prepare new data to send to the server"""
        new_data = {}

        # Add glucose reading if we have one
        if glucose > 0:
            glucose_entry = {
                "date": int(datetime.fromisoformat(timestamp.rstrip("Z")).timestamp() * 1000),
                "glucose": glucose,
                "timestamp": timestamp,
            }
            new_data["glucoseReadings"] = [glucose_entry]
            self.glucose_history[patient_name].append(glucose_entry)

        # Add carb entry if we have a meal
        if meal > 0:
            carb_entry = {"timestamp": timestamp, "carbs": meal, "enteredBy": "controller"}
            new_data["carbEntries"] = [carb_entry]
            self.meal_history[patient_name].append(carb_entry)

        return new_data

    def policy(self, observation, reward: float, done: bool, patient_name: str, meal: float, time) -> Action:
        """
        Get insulin dosage recommendation from OpenAPS server

        Args:
            observation: CtrlObservation object with glucose level
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            patient_name: Unique patient identifier
            meal: Carbohydrate amount in grams
            time: Current simulation time

        Returns:
            CtrlAction with basal and bolus insulin recommendations
        """
        patient_name = patient_name.replace("#", "")
        # Initialize patient if not already done
        if not self._initialize_patient(patient_name):
            logger.warning(f"Failed to initialize patient {patient_name}, using default action")
            # return Action(basal=1.0, bolus=0.0)  # Default safe action
            raise ValueError("Forgot to initialise patient.")
        # Extract glucose level
        glucose_level = observation.CGM  # if hasattr(observation, "CGM") else 100.0
        print(glucose_level)
        # Convert time to timestamp
        timestamp = self._convert_time_to_timestamp(time)

        # Prepare new data
        new_data = self._prepare_new_data(patient_name, glucose_level, meal, timestamp)
        print(new_data)
        # Prepare calculation request
        calc_data = {
            "currentTime": timestamp,
            "newData": new_data,
            "options": {
                "microbolus": True,  # Enable super micro bolus if available
                "overrideProfile": {},  # Could be used for temporary profile changes
            },
        }

        # Make calculation request
        response = self._make_request("POST", f"/patients/{patient_name}/calculate", calc_data)
        print(response["suggestion"])
        # Extract recommendation
        suggestion = response.get("suggestion", {})
        basal_rate = (suggestion.get("rate", 0.0) / 60) * self.frequency  # Default basal rate
        if basal_rate != 0:
            print(basal_rate)
        # Calculate bolus recommendation
        # OpenAPS typically provides basal adjustments, bolus calculation
        # might need to be derived from the suggestion
        bolus_amount = 0.0

        # Check if there's a microbolus recommendation
        if "units" in suggestion:
            bolus_amount = suggestion.get("units", 0.0)

        # Check for SMB (Super Micro Bolus) recommendation
        if "microbolus" in suggestion:
            bolus_amount += suggestion.get("microbolus", 0.0)

        # Log the recommendation
        logger.debug(
            f"Patient {patient_name}: BG={glucose_level:.1f}, "
            f"Meal={meal:.1f}g, Basal={basal_rate:.3f}, Bolus={bolus_amount:.3f}"
        )

        # Store the last glucose time
        self.last_glucose_time[patient_name] = timestamp

        return Action(basal=basal_rate, bolus=bolus_amount)

    def get_patient_status(self, patient_name: str) -> Optional[Dict[str, Any]]:
        """Get current patient status from the server"""
        try:
            if patient_name not in self.patient_profiles:
                logger.warning(f"Patient {patient_name} not initialized")
                return None

            response = self._make_request("GET", f"/patients/{patient_name}/status")
            return response

        except Exception as e:
            logger.error(f"Error getting patient status for {patient_name}: {str(e)}")
            return None

    def update_patient_profile(self, patient_name: str, profile_updates: Dict[str, Any]) -> bool:
        """Update patient profile on the server"""
        try:
            if patient_name not in self.patient_profiles:
                logger.warning(f"Patient {patient_name} not initialized")
                return False

            response = self._make_request("PATCH", f"/patients/{patient_name}/profile", profile_updates)

            # Update local copy
            self.patient_profiles[patient_name].update(profile_updates)

            logger.info(f"Updated profile for patient {patient_name}: {list(profile_updates.keys())}")
            return True

        except Exception as e:
            logger.error(f"Error updating patient profile for {patient_name}: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if the OpenAPS server is responding"""
        try:
            response = self._make_request("GET", "/health")
            logger.info(f"Server health check: {response.get('message', 'OK')}")
            return True
        except Exception as e:
            logger.error(f"Server health check failed: {str(e)}")
            return False


# Example usage and integration
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    # Test the controller
    ctrl = ORefZeroController()

    # Health check
    if not ctrl.health_check():
        print("Warning: OpenAPS server is not responding")
    CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
    # Test policy call
    for k in range(5):
        test_obs = CtrlObservation(220.0 + k)  # 120 mg/dL glucose
        test_time = datetime.now() + timedelta(minutes=5 * k)

        action = ctrl.policy(
            observation=test_obs,
            reward=0,
            done=False,
            patient_name="test_patient",
            meal=30.0,
            time=test_time,  # 30g carbs
        )

        print(f"Recommended action: Basal={action.basal:.3f}, Bolus={action.bolus:.3f}")
