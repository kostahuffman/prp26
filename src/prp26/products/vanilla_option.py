from datetime import datetime
import json

from prp26.products import Basket, StructuredProduct, Underlying
from prp26.products.schedules import ObservationSchedule


class VanillaOption(StructuredProduct):
    """Generic vanilla option supporting European, American, Bermudan, and Asian exercise.

    This product can be priced using either:
    1. Appropriate evaluator (via get_evaluator()) for direct pricing
    2. ComposablePayoff with matching option component for compositional approach

    Examples:
        # European option
        euro_option = VanillaOption(
            product_id="SPX_CALL_1Y",
            currency="USD",
            notional=100000,
            underlying=Underlying("SPX", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="european"
        )

        # American option (can exercise any time)
        american_option = VanillaOption(
            product_id="SPX_CALL_1Y_AMER",
            currency="USD",
            notional=100000,
            underlying=Underlying("SPX", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="put",
            exercise_style="american",
            observation_times=[0.25, 0.5, 0.75, 1.0]  # Quarterly observations
        )

        # Bermudan option (can exercise on specific dates)
        bermudan_option = VanillaOption(
            product_id="SPX_CALL_1Y_BERM",
            currency="USD",
            notional=100000,
            underlying=Underlying("SPX", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="bermudan",
            observation_times=[0.25, 0.5, 0.75, 1.0],
            exercise_times=[0.5, 1.0]  # Can only exercise at 6M and 1Y
        )

        # Asian option (pays based on average)
        asian_option = VanillaOption(
            product_id="SPX_ASIAN_CALL_1Y",
            currency="USD",
            notional=100000,
            underlying=Underlying("SPX", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="asian",
            observation_times=[0.25, 0.5, 0.75, 1.0],
            averaging_type="average_price"  # or "average_strike"
        )
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        underlying: Underlying,
        strike: float,
        maturity: float,
        option_type: str = "call",  # "call" or "put"
        exercise_style: str = "european",  # "european", "american", "bermudan", or "asian"
        observation_times: list[float] = None,  # For american/bermudan/asian
        exercise_times: list[float] = None,  # For bermudan only
        averaging_type: str = "average_price",  # For asian only: "average_price" or "average_strike"
        issue_date: datetime = None,
        maturity_date: datetime = None,
    ):
        super().__init__(
            product_id=product_id,
            currency=currency,
            notional=notional,
            issue_date=issue_date,
            maturity_date=maturity_date,
        )
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type.lower()
        self.exercise_style = exercise_style.lower()
        self.averaging_type = averaging_type.lower() if averaging_type else "average_price"

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        if self.exercise_style not in ("european", "american", "bermudan", "asian"):
            raise ValueError(
                f"exercise_style must be 'european', 'american', 'bermudan', or 'asian', got {exercise_style}"
            )

        # Handle observation times
        if observation_times is None:
            if self.exercise_style == "european":
                self.observation_times = [maturity]
            else:
                # Default quarterly observations for american/bermudan/asian
                n_quarters = max(1, int(maturity * 4))
                self.observation_times = [
                    maturity * (i + 1) / n_quarters for i in range(n_quarters)
                ]
        else:
            self.observation_times = sorted(observation_times)
            if abs(self.observation_times[-1] - maturity) > 1e-10:
                raise ValueError("observation_times must include maturity")

        # Handle exercise times (bermudan only)
        if self.exercise_style == "bermudan":
            if exercise_times is None:
                raise ValueError("exercise_times required for bermudan options")
            self.exercise_times = sorted(exercise_times)
            # Validate exercise times are subset of observation times
            for ex_time in self.exercise_times:
                if not any(abs(ex_time - obs) < 1e-10 for obs in self.observation_times):
                    raise ValueError(f"Exercise time {ex_time} not in observation_times")
        else:
            self.exercise_times = None

        # Create basket with single underlying
        self.basket = Basket(
            underlyings=[underlying],
            weights=[1.0],
            worst_of=False,
        )

        # Create observation schedule
        self.observation_schedule = ObservationSchedule(times=self.observation_times)

    def get_evaluator(self):
        """Get payoff evaluator for pricing engine.

        Returns:
            Appropriate evaluator based on exercise_style:
            - European: EuropeanOptionEvaluator
            - American: AmericanOptionEvaluator
            - Bermudan: BermudanOptionEvaluator
            - Asian: AsianOptionEvaluator
        """
        if self.exercise_style == "european":
            from .payoffs.evaluators import EuropeanOptionEvaluator

            return EuropeanOptionEvaluator(
                maturity=self.maturity,
                strike=self.strike,
                option_type=self.option_type,
                notional=self.notional,
            )
        elif self.exercise_style == "american":
            from .payoffs.evaluators import AmericanOptionEvaluator

            return AmericanOptionEvaluator(
                observation_times=self.observation_times,
                strike=self.strike,
                option_type=self.option_type,
                notional=self.notional,
            )
        elif self.exercise_style == "asian":
            from .payoffs.evaluators import AsianOptionEvaluator

            return AsianOptionEvaluator(
                observation_times=self.observation_times,
                strike=self.strike,
                option_type=self.option_type,
                averaging_type=self.averaging_type,
                notional=self.notional,
            )
        else:  # bermudan
            from .payoffs.evaluators import BermudanOptionEvaluator

            return BermudanOptionEvaluator(
                observation_times=self.observation_times,
                exercise_times=self.exercise_times,
                strike=self.strike,
                option_type=self.option_type,
                notional=self.notional,
            )

    def to_composable_payoff(self):
        """Convert to ComposablePayoff with appropriate option component.

        This demonstrates the compositional approach.

        Returns:
            ComposablePayoff with matching option component based on exercise_style
        """
        from .payoffs import (
            AmericanOption,
            AsianOption,
            BermudanOption,
            ComposablePayoff,
            EuropeanOption,
        )

        if self.exercise_style == "european":
            component = EuropeanOption(
                strike=self.strike,
                option_type=self.option_type,
                participation=1.0,
            )
        elif self.exercise_style == "american":
            component = AmericanOption(
                strike=self.strike,
                option_type=self.option_type,
                participation=1.0,
            )
        elif self.exercise_style == "asian":
            component = AsianOption(
                strike=self.strike,
                option_type=self.option_type,
                asian_type=self.averaging_type,  # AsianOption uses 'asian_type' parameter
                participation=1.0,
            )
        else:  # bermudan
            component = BermudanOption(
                strike=self.strike,
                option_type=self.option_type,
                exercise_times=self.exercise_times,
                participation=1.0,
            )

        return ComposablePayoff(components=[component])

    def validate(self) -> bool:
        """Validate product definition."""
        super().validate()

        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.strike <= 0:
            raise ValueError("Strike must be positive")
        if not self.observation_times:
            raise ValueError("observation_times cannot be empty")
        if self.exercise_style == "bermudan" and not self.exercise_times:
            raise ValueError("exercise_times required for bermudan options")

        return True

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "product_id": self.product_id,
            "currency": self.currency,
            "notional": self.notional,
            "underlying": self.underlying.symbol,
            "strike": self.strike,
            "maturity": self.maturity,
            "option_type": self.option_type,
            "exercise_style": self.exercise_style,
            "observation_times": self.observation_times,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date else None,
        }
        if self.exercise_times:
            data["exercise_times"] = self.exercise_times
        return json.dumps(data)
