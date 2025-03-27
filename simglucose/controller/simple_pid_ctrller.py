"""A simple PID controller

@date: 2020.03.26
@editor: Kexin Wei

"""

import numpy as np
from .base import Controller, Action
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class SimplePIDController(Controller):
    def __init__(
        self, target_BG=120, basal_rate=0.2, k_P=0.01, k_I=0.1, k_D=0, sampling_time=5
    ):
        """
        PID Controller for insulin delivery

        Args:
            target_BG (float): Target blood glucose in mg/dL
            basal_rate (float): Default basal rate in U/h
            k_P (float): Proportional gain in U/h / (100 mg/dL)
            k_I (float): Integral gain in U/h / (100 mg/dL) / h
            k_D (float): Derivative gain in U/h / (100 mg/dL) * h
            sampling_time (int): Sampling time in minutes
        """
        self.target_BG = target_BG
        self.basal_rate = basal_rate
        self.k_P = k_P
        self.k_I = k_I
        self.k_D = k_D
        self.sampling_time = sampling_time

        # State variables
        self.t_last = None
        self.e_integral = 0  # Integral of blood glucose error in mg/dL*min
        self.e_old = None  # Previous blood glucose error for derivative calculation
        self.last_insulin_rate = basal_rate  # Store the last calculated insulin rate

    def reset(self):
        """Reset the controller state"""
        self.t_last = None
        self.e_integral = 0
        self.e_old = None
        self.last_insulin_rate = self.basal_rate
        self.suspend_count = 0
        self.last_normal_dose_time = None
        self.controller_run_count = 0

    def policy(self, observation, reward, done, **info) -> Action:
        # Get current glucose and time
        current_glucose = observation.CGM
        # Get current time - handle case when time is not provided
        if "time" in info:
            current_time = info["time"]
        else:
            # If no time is provided, create a time step based on sampling time
            if self.t_last is None:
                current_time = datetime.now()
            else:
                # Create a new time that's sampling_time minutes after the last time
                current_time = self.t_last + timedelta(minutes=self.sampling_time)
            logger.warning(
                "No 'time' provided in info dictionary. Using simulated time step."
            )

            # Initialize on first call
        if self.t_last is None:
            self.t_last = current_time
            self.e_integral = 0
            self.e_old = None
            self.last_normal_dose_time = current_time
            self.controller_run_count = 0
            # Start with a conservative dose - 50% of basal if glucose is high
            initial_basal = self.basal_rate * (
                0.5 if current_glucose > self.target_BG else 0.3
            )
            logger.info(
                f"PID Controller initialized - Starting with conservative basal rate: {initial_basal}"
            )
            self.last_insulin_rate = initial_basal
            return Action(basal=initial_basal, bolus=0)

        # Increment controller run counter
        self.controller_run_count += 1

        # Check if it's time to update (based on sampling time)
        time_diff_minutes = (
            (current_time - self.t_last).total_seconds() / 60
            if hasattr(current_time, "total_seconds")
            else self.sampling_time
        )

        if time_diff_minutes < self.sampling_time:
            return Action(basal=0, bolus=0)

        # Calculate error
        error = self.target_BG - current_glucose
        if error >= 0:
            logger.debug(f"Glucose too low, insulin suspend")
            return Action(basal=0, bolus=0)
        # Update integral term
        self.e_integral += error * self.sampling_time

        # Calculate
        # insulin_rate = self.basal_rate - self.k_P * error - self.k_I * self.e_integral
        insulin_rate = self.basal_rate - self.k_P * error - self.k_I * self.e_integral

        # Calculate derivative term (if we have previous error)
        if self.e_old is not None:
            de_dt = (error - self.e_old) / self.sampling_time
            insulin_rate -= self.k_D * de_dt

        # Ensure non-negative insulin rate
        insulin_rate = max(0, insulin_rate)

        # Update state variables
        self.t_last = current_time
        self.e_old = error

        # Return calculated insulin rate
        return Action(basal=insulin_rate, bolus=0)
