import random
from collections import namedtuple


Action = namedtuple("meal_bolus_action", ["bolus"])


class MealAnnouncementBolusController:
    """
    Simple meal bolus controller that releases insulin bolus before meals.

    The controller calculates bolus based on upcoming meals in the scenario,
    releasing insulin a specified time before the meal occurs.
    """

    def __init__(
        self,
        scenario,
        carb_factor=10,
        release_time_before_meal=10,  # minutes before meal to release bolus
        carb_estimation_error=0.3,  # +/- percentage of carb estimation error
        body_weight=None,
    ):
        """
        Initialize the meal bolus controller.

        Args:
            scenario: Scenario enum from scenario_simple.py (e.g., Scenario.ONE_DAY)
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Percentage of error in carbohydrate estimation (e.g., 0.3 for +/- 30%)
            body_weight: Patient body weight in kg (optional, for scenario meal calculation)
        """
        self.scenario = scenario
        self.carb_factor = carb_factor
        self.release_time_before_meal = release_time_before_meal
        self.carb_estimation_error = carb_estimation_error
        self.body_weight = body_weight

        # Pre-calculate all meal times and amounts for efficiency
        self._meal_schedule = self._build_meal_schedule()

    def _build_meal_schedule(self):
        """
        Build a schedule of all meals in the scenario.

        Returns:
            List of tuples (meal_time_minutes, meal_amount_grams)
        """
        meal_schedule = []

        # Scan through the scenario's max time to find all meals
        max_time = self.scenario.max_t
        for t in range(0, max_time + 1):
            action = self.scenario.get_action(t, self.body_weight)
            if action.meal > 0:
                meal_schedule.append((t, action.meal))

        return meal_schedule

    def policy(self, t):
        """
        Get bolus action for the current time.

        Args:
            t: Current time in minutes

        Returns:
            Action namedtuple with bolus amount in U (units of insulin)
        """
        # Force t to int for exact time matching
        t = int(t)

        # Check if there's a meal coming up at the release time
        target_meal_time = t + self.release_time_before_meal

        for meal_time, meal_amount in self._meal_schedule:
            if meal_time == target_meal_time:
                # Add randomness to meal amount to simulate patient uncertainty
                if self.carb_estimation_error > 0:
                    random_factor = random.uniform(
                        -self.carb_estimation_error, self.carb_estimation_error
                    )
                    meal_amount *= 1 + random_factor

                # Calculate bolus: meal amount / carb factor
                bolus = meal_amount / self.carb_factor
                return Action(bolus=bolus)  # U/min

        # No meal coming up, return zero bolus
        return Action(bolus=0)

    def reset(self):
        """Reset the controller state."""
        # Rebuild meal schedule in case scenario changed
        self._meal_schedule = self._build_meal_schedule()
