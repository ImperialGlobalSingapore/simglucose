import random
from collections import namedtuple
from datetime import datetime


Action = namedtuple("meal_bolus_action", ["bolus", "estimated_carbs"])


class MealAnnouncementBolusController:
    """
    Simple meal bolus controller that releases insulin bolus before meals.

    The controller calculates bolus based on upcoming meals in the scenario,
    releasing insulin a specified time before the meal occurs.
    """

    def __init__(
        self,
        meal_schedule=None,
        carb_factor: float = 10,
        release_time_before_meal: int = 10,  # minutes before meal to release bolus
        carb_estimation_error: bool = True,  # flag: enable realistic carb mis-estimation
        t_start=None,  # patient start time (datetime)
    ):
        """
        Initialize the meal bolus controller.

        Args:
            meal_schedule: List of tuples (time_minutes, carbs_grams), e.g.,
                          [(120, 50), (360, 75), (720, 60)]
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Flag to enable carb mis-estimation. When True, the
                          patient's estimate deviates from the true carbs by a random
                          magnitude in [11.2%, 30.6%] (sign random), reflecting the
                          20.9 +/- 9.7% carb-counting error reported in the literature.
                          Set to False to disable the error (use exact carbs).
            sample_time: Time period over which to deliver bolus in minutes (default: 1)
            t_start: Patient simulation start time as datetime object (optional)
        """
        self._meal_schedule = meal_schedule if meal_schedule is not None else []
        self.carb_factor = carb_factor
        self.release_time_before_meal = release_time_before_meal
        self.carb_estimation_error = carb_estimation_error
        self.t_start = t_start

    def policy(self, t):
        """
        Get bolus action for the current time.

        Args:
            t: Current time - can be either:
                - elapsed time in minutes (int/float), or
                - datetime object (will calculate elapsed time from t_start)

        Returns:
            Action namedtuple with bolus amount in U/min (insulin rate)
        """
        # Calculate elapsed time in minutes
        if isinstance(t, datetime):
            if self.t_start is None:
                raise ValueError("t_start must be set when using datetime for policy")
            elapsed_time = (t - self.t_start).total_seconds() / 60
        else:
            elapsed_time = t

        # Force to int for exact time matching
        elapsed_time = int(elapsed_time)

        # Check if there's a meal coming up at the release time
        target_meal_time = elapsed_time + self.release_time_before_meal

        for meal_time, meal_amount in self._meal_schedule:
            if meal_time == target_meal_time:
                # Simulate patient carb-counting error. Patients misjudge meal carbs
                # by 20.9 +/- 9.7% on average (range ~11.2%-30.6%), so:
                #   patient_estimate = true_carbs +/- uniform(11.2%, 30.6%) * true_carbs
                if self.carb_estimation_error:
                    error_magnitude = random.uniform(0.112, 0.306)
                    sign = random.choice([-1, 1])
                    meal_amount *= 1 + sign * error_magnitude

                # Calculate bolus in total units: meal amount / carb factor
                bolus = meal_amount / self.carb_factor  # U
                estimated = round(meal_amount, 1)

                return Action(bolus=bolus, estimated_carbs=estimated)  # U

        # No meal coming up, return zero bolus
        return Action(bolus=0, estimated_carbs=0.0)
