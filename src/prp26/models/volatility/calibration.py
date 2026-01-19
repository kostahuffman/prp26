"""
Volatility model calibration utilities.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .base import VolSurface
from .local_vol import LocalVolModel
from .slv import HestonModel, SLVModel


class VolatilityCalibrator:
    """
    Calibration engine for volatility models.

    Supports:
    - Heston model calibration to vanilla options
    - SLV two-stage calibration
    - Local vol from implied vols
    """

    def __init__(
        self,
        implied_surface: VolSurface,
        spot: float = 100.0,
        rate: float = 0.0,
    ):
        """
        Initialize calibrator.

        Args:
            implied_surface: Market implied volatility surface
            spot: Current spot price
            rate: Risk-free rate
        """
        self.implied_surface = implied_surface
        self.spot = spot
        self.rate = rate

    def calibrate_heston(
        self,
        method: str = "least_squares",
        bounds: dict | None = None,
    ) -> HestonModel:
        """
        Calibrate Heston model to implied volatility surface.

        Args:
            method: Optimization method (least_squares, global, moments)
            bounds: Parameter bounds dict

        Returns:
            Calibrated HestonModel
        """
        if method == "moments":
            return self._calibrate_heston_moments()

        # Define parameter bounds
        if bounds is None:
            bounds = {
                "kappa": (0.1, 10.0),
                "theta": (0.01, 1.0),
                "xi": (0.01, 2.0),
                "rho": (-1.0, 0.0),
                "v0": (0.01, 1.0),
            }

        # Initial guess
        x0 = np.array(
            [
                2.0,  # kappa
                0.04,  # theta
                0.3,  # xi
                -0.7,  # rho
                0.04,  # v0
            ]
        )

        # Bounds array
        bounds_array = [
            bounds["kappa"],
            bounds["theta"],
            bounds["xi"],
            bounds["rho"],
            bounds["v0"],
        ]

        def objective(params):
            """Objective function: sum of squared errors."""
            model = HestonModel(*params)
            return self._calibration_error(model)

        if method == "least_squares":
            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds_array, options={"maxiter": 100}
            )
            optimal_params = result.x

        elif method == "global":
            result = differential_evolution(
                objective,
                bounds_array,
                maxiter=50,
                seed=42,
            )
            optimal_params = result.x

        else:
            raise ValueError(f"Unknown method: {method}")

        return HestonModel(*optimal_params)

    def _calibrate_heston_moments(self) -> HestonModel:
        """Simplified moment-based calibration."""
        times = self.implied_surface.times
        atm_vols = np.array([self.implied_surface.get_atm_vol(t) for t in times])

        v0 = atm_vols[0] ** 2
        theta = np.mean(atm_vols) ** 2
        kappa = 2.0
        xi = 0.3
        rho = -0.7

        return HestonModel(kappa, theta, xi, rho, v0)

    def _calibration_error(self, heston: HestonModel) -> float:
        """
        Compute calibration error as sum of squared vol errors.

        For production: use option price errors instead.
        """
        total_error = 0.0
        n_points = 0

        times = self.implied_surface.times
        strikes = self.implied_surface.strikes

        for i, t in enumerate(times):
            for j, k in enumerate(strikes):
                market_vol = self.implied_surface.vols[i, j]
                model_vol = self._heston_implied_vol(heston, t, k)

                error = (model_vol - market_vol) ** 2
                total_error += error
                n_points += 1

        return total_error / n_points if n_points > 0 else 1e10

    def _heston_implied_vol(self, heston: HestonModel, t: float, k: float) -> float:
        """
        Compute Heston implied volatility using characteristic function.

        Simplified: returns approximation.
        Production: use Carr-Madan FFT or Lewis formula.
        """
        # Approximate with average variance
        v_t = heston.theta + (heston.v0 - heston.theta) * np.exp(-heston.kappa * t)
        return np.sqrt(v_t)

    def calibrate_slv(self) -> SLVModel:
        """
        Two-stage SLV calibration.

        Stage 1: Calibrate Heston to ATM and skew
        Stage 2: Calibrate leverage function to full surface

        Returns:
            Calibrated SLVModel
        """
        # Stage 1: Heston
        print("Calibrating Heston model...")
        heston = self.calibrate_heston(method="moments")

        # Stage 2: Leverage function
        print("Calibrating leverage function...")
        leverage_func = self._calibrate_leverage_function(heston)

        return SLVModel(
            heston_model=heston,
            leverage_function=leverage_func,
        )

    def _calibrate_leverage_function(self, heston: HestonModel) -> callable:
        """
        Calibrate leverage function L(t, K) to match implied surface.

        Uses particle method or mixing formula.
        Simplified: returns ratio of local vol to Heston vol.
        """
        # Get local vol surface
        times = self.implied_surface.times
        strikes = self.implied_surface.strikes

        # Compute leverage as ratio
        leverage_grid = np.zeros_like(self.implied_surface.vols)

        for i, t in enumerate(times):
            heston_vol = np.sqrt(
                heston.theta + (heston.v0 - heston.theta) * np.exp(-heston.kappa * t)
            )

            for j, _k in enumerate(strikes):
                market_vol = self.implied_surface.vols[i, j]
                leverage_grid[i, j] = market_vol / heston_vol if heston_vol > 0 else 1.0

        # Create interpolated leverage function
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(
            (times, strikes),
            leverage_grid,
            bounds_error=False,
            fill_value=1.0,
        )

        def leverage(t: float, s):
            """Leverage function L(t, S)."""
            # Handle both scalar and array inputs
            s_scalar = np.mean(s) if hasattr(s, '__len__') else s
            k = s_scalar / self.spot  # Convert to moneyness
            return float(interpolator([[t, k]])[0])

        return leverage

    def calibrate_local_vol(self) -> LocalVolModel:
        """
        Calibrate local volatility model from implied surface.

        Uses Dupire's formula to convert implied to local vol.

        Returns:
            Calibrated LocalVolModel
        """

        # Create mock market data provider
        class MockMarketData:
            def __init__(self, surface, rate):
                self.surface = surface
                self.rate = rate

            def get_implied_vol_surface(self):
                return self.surface

            def get_rate(self):
                return self.rate

        market_data = MockMarketData(self.implied_surface, self.rate)

        # Create and calibrate local vol model
        lv_model = LocalVolModel(vol_surface=self.implied_surface)
        lv_model.calibrate(market_data)

        return lv_model


class CalibrationResult:
    """Container for calibration results and diagnostics."""

    def __init__(
        self,
        model,
        calibration_error: float,
        residuals: np.ndarray | None = None,
        diagnostics: dict | None = None,
    ):
        self.model = model
        self.calibration_error = calibration_error
        self.residuals = residuals
        self.diagnostics = diagnostics or {}

    def summary(self) -> str:
        """Generate calibration summary report."""
        lines = [
            "=" * 60,
            "Calibration Result Summary",
            "=" * 60,
            f"Model type: {type(self.model).__name__}",
            f"Calibration error (RMSE): {np.sqrt(self.calibration_error):.6f}",
        ]

        if self.residuals is not None:
            lines.extend(
                [
                    f"Max residual: {np.max(np.abs(self.residuals)):.6f}",
                    f"Mean abs residual: {np.mean(np.abs(self.residuals)):.6f}",
                ]
            )

        if self.diagnostics:
            lines.append("\nDiagnostics:")
            for key, value in self.diagnostics.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60)

        return "\n".join(lines)
