from enum import Enum
from simglucose.simulation.scenario import Action


class Scenario(Enum):
    NO_MEAL = "no_meal"
    SINGLE_MEAL = "single_meal"
    ONE_DAY = "one_day"
    THREE_DAY = "three_day"

    def get_action(self, t, body_weight=None):
        """
        Get meal action for the given time.

        Args:
            t: time in minutes
            body_weight: weight of the patient in kg (optional)

        Returns:
            Action namedtuple with meal amount
        """
        # Force t to int for exact time matching
        t = int(t)

        carb_times_in_hour = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [6],  # Assuming a meal at 6:00
            Scenario.ONE_DAY: [7, 12, 19],  # Meals at 7:00, 12:00, 19:00
            Scenario.THREE_DAY: [
                h + 24 * d for d in range(3) for h in [7, 12, 19]
            ],  # Meals at 7:00, 12:00, 19:00 for 3 days
        }

        # Predefined carb amounts for each meal time
        carb_amounts = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [50],  # 50g of carbs for single meal
            Scenario.ONE_DAY: [40, 50, 70],  # 40g, 50g, 70g of carbs per meal
            Scenario.THREE_DAY: [
                40,
                50,
                70,
            ]
            * 3,  # 40g, 50g, 70g of carbs per meal for 3 days
        }
        if body_weight is not None:
            carb_amounts[Scenario.ONE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ]
            carb_amounts[Scenario.THREE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ] * 3

        # Convert hours to minutes for comparison
        carb_times_in_minutes = [h * 60 for h in carb_times_in_hour[self]]

        # Check if current time matches any meal time exactly
        if t in carb_times_in_minutes:
            idx = carb_times_in_minutes.index(t)
            return Action(meal=carb_amounts[self][idx])
        return Action(meal=0)

    def get_carb(self, t, body_weight=None):
        """
        Returns carbs only at exact meal times, 0 otherwise.
        """
        return self.get_action(t, body_weight).meal

    @property
    def max_t(self):
        """Maximum simulation time in minutes for this scenario."""
        return {
            Scenario.NO_MEAL: 1000,  # 16 hours + 40 minutes
            Scenario.SINGLE_MEAL: 1080,  # 18 hours
            Scenario.ONE_DAY: 1450,  # 24 hours + 10 minutes
            Scenario.THREE_DAY: 4330,  # 72 hours + 10 minutes
        }[self]


if __name__ == "__main__":
    # Test exact meal times
    print("Testing exact meal times:")
    print(
        f"NO_MEAL at 420min (7:00): {Scenario.NO_MEAL.get_carb(420, 70)}"
    )  # Should be 0
    print(
        f"SINGLE_MEAL at 360min (6:00): {Scenario.SINGLE_MEAL.get_carb(360, 70)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"THREE_DAY at 420min (7:00 day 1): {Scenario.THREE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35

    print("\nTesting non-meal times (should all return 0):")
    print(f"SINGLE_MEAL at 365min: {Scenario.SINGLE_MEAL.get_carb(365)}")  # Should be 0
    print(f"ONE_DAY at 425min: {Scenario.ONE_DAY.get_carb(425)}")  # Should be 0
    print(f"THREE_DAY at 425min: {Scenario.THREE_DAY.get_carb(425)}")  # Should be 0

    print("\nTesting without body_weight:")
    print(f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420)}")  # Should be 40
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140)}"
    )  # Should be 70
