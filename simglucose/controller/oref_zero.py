import zoneinfo
import logging

import requests
from collections import namedtuple
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from simglucose.controller.base import Controller, Action


logger = logging.getLogger(__name__)


class ORefZeroController(Controller):
    """
    OpenAPS oref0 controller that communicates with Node.js server
    for insulin dosage recommendations
    """

    DEFAULT_TIMEZONE = "UTC"
    MINIMAL_TIMESTEP = 5  # in minutes

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        timeout: int = 30,
    ):
        """
        Initialize the ORefZero controller (supports multiple patients)

        Args:
            server_url: URL of the Node.js OpenAPS server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Patient state tracking (multi-patient support)
        self.patient_profiles = {}  # patientId -> profile mapping
        self.last_glucose_time = {}  # patientId -> last glucose timestamp
        self.glucose_history = {}  # patientId -> list of glucose readings
        self.meal_history = {}  # patientId -> list of meal entries
        self.pump_history = {}  # patientId -> list of pump events
        self.collect_meal = {}  # patientId -> accumulated meal amount
        self.last_insulin = {}  # patientId -> last insulin recommendation

        # Store the required profile
        # Set default profile, but override defaults with any keys present in 'profile'
        self.default_profile = {
            "current_basal": None,  # Current basal rate in U/h
            "sens": 50,  # Insulin Sensitivity Factor (ISF)
            "dia": 6,  # Duration of Insulin Action in hours
            "carb_ratio": 10,  # Carb Ratio (g/U)
            "max_iob": 6,  # Maximum insulin on board allowed
            "max_basal": 3.5,  # Maximum temporary basal rate in U/h
            "max_daily_basal": 3.5,  # Maximum daily basal rate in units per day
            "max_bg": 120,  # Upper target
            "min_bg": 120,  # Lower target
            "maxCOB": 120,  # Maximum carbs on board
            "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 50}]},
            "min_5m_carbimpact": 12.0,  # Minimum carb absorption rate
            "type": "current",  # Profile type
        }

        logger.info(f"ORefZero Controller initialized with server: {self.server_url}")

    @property
    def target_bg(self) -> float:
        """Get the target blood glucose level."""
        return (self.default_profile["min_bg"] + self.default_profile["max_bg"]) / 2

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
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
            logger.error(
                f"HTTP error {e.response.status_code} for {url}: {e.response.text}"
            )

        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            raise Exception("Unexpected error during server request")

    def initialize_patient(
        self, patient_name: str, profile: Optional[Dict] = None
    ) -> bool:
        """Public method to initialize patient with given profile"""
        patient_name = patient_name.replace("#", "")
        patient_profile = self.default_profile.copy()
        if profile:
            patient_profile.update(profile)

        if not self._check_and_correct_profile(patient_profile):
            logger.error(f"Invalid profile for patient {patient_name}")
            return False

        # Prepare initialization data
        init_data = {
            "profile": patient_profile,
            "initialData": {"glucoseHistory": [], "pumpHistory": [], "carbHistory": []},
            "settings": {
                "timezone": self.DEFAULT_TIMEZONE,
                "historyRetentionHours": 24,
                "autoCleanup": True,
            },
        }

        try:
            response = self._make_request(
                "POST", f"/patients/{patient_name}/initialize", init_data
            )
            self.patient_profiles[patient_name] = patient_profile
            self.glucose_history[patient_name] = []
            self.meal_history[patient_name] = []
            self.pump_history[patient_name] = []
            self.last_glucose_time[patient_name] = None
            self.collect_meal[patient_name] = 0
            self.last_insulin[patient_name] = {"basal": 0.0, "bolus": 0.0}

            logger.info(f"Patient {patient_name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize patient {patient_name}: {str(e)}")
            return False

    def is_patient_initialized(
        self, patient_name: str, profile: Optional[Dict] = None
    ) -> bool:
        patient_name = patient_name.replace("#", "")
        """Initialize patient on the server"""
        if patient_name in self.patient_profiles:
            logger.debug(f"Patient {patient_name} already initialized")
            return True
        return False

    def _check_and_correct_profile(self, profile: Dict):
        # make sure current_basal is set
        if profile["current_basal"] is None:
            logger.error("Profile must include current_basal rate")
            return False

        # make sure min_bg < max_bg
        if profile["min_bg"] > profile["max_bg"]:
            logger.error("Profile min_bg must be less than max_bg")
            return False

        # make sure isfProfile has same sensitivity as sens
        profile["isfProfile"] = {
            "sensitivities": [{"offset": 0, "sensitivity": profile["sens"]}]
        }
        return True

    @staticmethod
    def _local_timezone() -> datetime:
        return datetime.now().astimezone()

    def _convert_to_utc(self, local_time: datetime) -> datetime:
        if local_time.tzinfo is None:
            # If naive datetime, assume local timezone
            local_time = local_time.replace(tzinfo=self._local_timezone().tzinfo)
        utc_time = local_time.astimezone(zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE))
        # Remove the timezone info (make it naive)
        return utc_time.replace(tzinfo=None)

    def _convert_time_to_timestamp(self, time_obj) -> str:
        """Convert time object to ISO timestamp string"""
        if isinstance(time_obj, str):

            # Assume it's already a valid timestamp
            try:
                time_obj = datetime.fromisoformat(time_obj.rstrip("Z"))
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )
        elif isinstance(time_obj, (int, float)):
            # Assume it's a Unix timestamp
            try:
                time_obj = datetime.fromtimestamp(time_obj)
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )

        # now we have a datetime object
        if not isinstance(time_obj, datetime):
            raise ValueError(f"Expected datetime object, got {type(time_obj)}")
        # Convert to UTC and format as ISO string
        utc_time = self._convert_to_utc(time_obj)
        return utc_time.isoformat() + "Z"  # Append 'Z' to indicate

    def _prepare_new_data(
        self, patient_name: str, glucose: float, meal: float, timestamp: str
    ) -> Dict[str, Any]:
        """Prepare new data to send to the server"""
        new_data = {}

        # Add glucose reading if we have one
        if glucose > 0:
            # Convert UTC timestamp string to local datetime, then to int (epoch seconds)
            utc_dt = datetime.fromisoformat(timestamp.rstrip("Z")).replace(
                tzinfo=zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE)
            )
            glucose_entry = {
                "date": int(utc_dt.timestamp() * 1000),  # Convert to milliseconds
                "glucose": glucose,
                "timestamp": timestamp,
            }
            new_data["glucoseReadings"] = [glucose_entry]
            self.glucose_history[patient_name].append(glucose_entry)

        # Add carb entry if we have a meal
        if meal > 0:
            carb_entry = {
                "created_at": timestamp,
                "carbs": meal,
                "enteredBy": "controller",
            }
            new_data["carbEntries"] = [carb_entry]
            self.meal_history[patient_name].append(carb_entry)

        return new_data

    def policy(
        self,
        observation,
        reward: float,
        done: bool,
        patient_name: str,
        meal: float,
        time,
    ) -> Action:
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
        if not self.is_patient_initialized(patient_name):
            if not self.initialize_patient(patient_name):
                logger.warning(f"Failed to initialize patient {patient_name}")
                # return Action(basal=1.0, bolus=0.0)  # Default safe action
                raise ValueError("Forgot to initialise patient.")

        # Convert time to timestamp
        timestamp = self._convert_time_to_timestamp(time)
        previous_timestamp = self.last_glucose_time.get(patient_name)

        # accumulate meal data
        self.collect_meal[patient_name] += meal

        # if previous_timestamp is not None and timestamp difference < MINIMAL_TIMESTEP
        if previous_timestamp is not None and (
            datetime.fromisoformat(timestamp.rstrip("Z"))
            < datetime.fromisoformat(previous_timestamp.rstrip("Z"))
            + timedelta(minutes=self.MINIMAL_TIMESTEP)
        ):
            return Action(
                basal=self.last_insulin[patient_name]["basal"],
                bolus=self.last_insulin[patient_name]["bolus"]
            )
            # return Action(basal=0, bolus=0)

        # Extract glucose level
        glucose_level = observation.CGM  # if hasattr(observation, "CGM") else 100.0

        # Prepare new data
        new_data = self._prepare_new_data(
            patient_name, glucose_level, self.collect_meal[patient_name], timestamp
        )
        self.collect_meal[patient_name] = 0  # Reset after sending
        print(new_data)
        # Prepare calculation request
        calc_data = {
            "currentTime": timestamp,
            "newData": new_data,
            "options": {
                "microbolus": False,  # Disable, since oref0 doesn't support microbolus, though it might be available in the future for oref1
                "overrideProfile": {},  # Could be used for temporary profile changes
            },
        }

        # Make calculation request
        endpoint = f"/patients/{patient_name}/calculate"
        response = self._make_request("POST", endpoint, calc_data)
        print(response["suggestion"])
        # Extract recommendation
        suggestion = response.get("suggestion", {})
        basal_rate = response.get("IIR", 0.0) / 60  # U/h -> U/min

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

        # Store the last glucose time and insulin recommendation
        self.last_glucose_time[patient_name] = timestamp
        self.last_insulin[patient_name] = {"basal": basal_rate, "bolus": bolus_amount}

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

    def update_patient_profile(
        self, patient_name: str, profile_updates: Dict[str, Any]
    ) -> bool:
        """Update patient profile on the server"""
        try:
            if patient_name not in self.patient_profiles:
                logger.warning(f"Patient {patient_name} not initialized")
                return False

            response = self._make_request(
                "PATCH", f"/patients/{patient_name}/profile", profile_updates
            )

            # Update local copy
            self.patient_profiles[patient_name].update(profile_updates)

            logger.info(
                f"Updated profile for patient {patient_name}: {list(profile_updates.keys())}"
            )
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
