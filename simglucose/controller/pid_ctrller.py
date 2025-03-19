import numpy as np
from .base import Controller, Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self):
        self.target_BG = 120  # Target blood glucose in mg/dL
        self.basal_rate = 1.0  # Default basal rate in U/h
        self.k_p = 1.0  # Increase proportional gain for more responsiveness
        self.k_i = 0.05  # Increase integral gain
        self.k_d = 0.2  # Add some derivative gain
        self.sampling_time = 5  # Sampling time in minutes
        
        # State variables
        self.last_time = None
        self.e_integral = 0  # Integral of blood glucose error in mg/dL*min
        self.e_old = None  # Previous blood glucose error for derivative calculation
        
    def policy(self, observation, reward, done, **info):
        # Extract current glucose reading
        current_glucose = observation.CGM
        
        # Get current time or use a simple counter if time not provided
        current_time = info.get('time', None)
        
        # Log input parameters
        logger.info(f"PID Controller input - CGM: {current_glucose}, Time: {current_time}, Meal: {info.get('meal', 0)}")
        
        # Initialize time on first call
        if self.last_time is None:
            self.last_time = current_time if current_time is not None else 0
            self.e_old = None
            self.e_integral = 0
            logger.info(f"PID Controller initialized - Returning default basal rate: {self.basal_rate}")
            return Action(basal=self.basal_rate, bolus=0)
        
        # Calculate time difference in minutes
        if current_time is not None:
            dt = (current_time - self.last_time) / 60  # Convert to minutes
        else:
            dt = self.sampling_time  # Default to sampling time
        
        logger.info(f"Time difference: {dt} minutes")
        
        # Calculate control error (target - actual)
        error = self.target_BG - current_glucose
        logger.info(f"Control error: {error} mg/dL (Target: {self.target_BG} - Current: {current_glucose})")
        
        # Update integral term
        self.e_integral += error / 60 * dt
        logger.info(f"Updated integral term: {self.e_integral}")
        
        # Calculate base insulin rate
        base_insulin = self.basal_rate
        logger.info(f"Base insulin rate: {base_insulin} U/h")
        
        # Calculate proportional component
        p_component = (self.k_p / 100) * error
        logger.info(f"Proportional component: {p_component} U/h (k_p: {self.k_p} * error: {error} / 100)")
        
        # Calculate integral component
        i_component = (self.k_i / 100) * self.e_integral
        logger.info(f"Integral component: {i_component} U/h (k_i: {self.k_i} * integral: {self.e_integral} / 100)")
        
        # Initialize derivative component
        d_component = 0
        
        # Add derivative component if we have a previous error
        if self.e_old is not None and dt > 0:
            # Rate of change
            de_dt = (error - self.e_old) * 60 / dt
            # Add differential component
            d_component = (self.k_d / 100) * de_dt
            logger.info(f"Derivative component: {d_component} U/h (k_d: {self.k_d} * de_dt: {de_dt} / 100)")
        
        # Calculate insulin rate using PID formula
        # Note: error is (target - actual), so we subtract the components
        insulin_rate = base_insulin - p_component - i_component - d_component
        logger.info(f"Calculated insulin rate: {insulin_rate} U/h = {base_insulin} - {p_component} - {i_component} - {d_component}")
        
        # Store current error for next iteration
        self.e_old = error
        
        # Update last time
        self.last_time = current_time if current_time is not None else self.last_time + self.sampling_time
        
        # Handle meal information if provided
        meal = info.get('meal', 0)
        bolus = 0
        
        # Simple bolus calculator based on meal size
        if meal > 0:
            # Insulin to carb ratio - simplified for this example
            icr = 10  # 1 unit per 10g of carbs
            bolus = meal / icr
            logger.info(f"Meal bolus: {bolus} U (meal: {meal}g / ICR: {icr})")
        
        # Ensure insulin rate is non-negative
        basal = max(0, insulin_rate)
        if basal != insulin_rate:
            logger.info(f"Clamped negative insulin rate to zero: {insulin_rate} → {basal}")
        
        logger.info(f"Final insulin delivery - Basal: {basal} U/h, Bolus: {bolus} U, Total: {basal + bolus} U")
        
        return Action(basal=basal, bolus=bolus)
    
    def reset(self):
        self.last_time = None
        self.e_integral = 0
        self.e_old = None
