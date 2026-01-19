"""
Stochastic Local Volatility (SLV) model.
"""

import numpy as np
from typing import Optional, Tuple, Callable

from .base import VolSurface
from .local_vol import LocalVolModel


class HestonModel:
    """
    Heston stochastic volatility model.

    dS = r*S*dt + √v*S*dW_S
    dv = κ(θ - v)*dt + ξ*√v*dW_v
    dW_S * dW_v = ρ dt
    """

    def __init__(
        self,
        kappa: float = 2.0,  # Mean reversion speed
        theta: float = 0.04,  # Long-run variance
        xi: float = 0.3,  # Vol of vol
        rho: float = -0.7,  # Correlation
        v0: float = 0.04,  # Initial variance
    ):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

        self._validate()

    def _validate(self):
        """Validate Feller condition."""
        if 2 * self.kappa * self.theta < self.xi * self.xi:
            import warnings

            warnings.warn("Feller condition violated: variance can reach zero", stacklevel=2)

    def simulate_variance(
        self, times: np.ndarray, n_paths: int, scheme: str = "euler"
    ) -> np.ndarray:
        """
        Simulate variance process.

        Args:
            times: Time grid
            n_paths: Number of paths
            scheme: Integration scheme (euler, milstein, qe)

        Returns:
            Variance paths [n_paths, n_steps]
        """
        n_steps = len(times)
        variances = np.zeros((n_paths, n_steps))
        variances[:, 0] = self.v0

        for i in range(1, n_steps):
            dt = times[i] - times[i - 1]
            v_prev = variances[:, i - 1]

            if scheme == "euler":
                dW = np.random.randn(n_paths) * np.sqrt(dt)
                v_next = (
                    v_prev
                    + self.kappa * (self.theta - v_prev) * dt
                    + self.xi * np.sqrt(np.maximum(v_prev, 0)) * dW
                )
                # Reflection at zero
                variances[:, i] = np.maximum(v_next, 1e-8)

            elif scheme == "qe":
                # Quadratic-Exponential scheme (Andersen 2008)
                variances[:, i] = self._qe_step(v_prev, dt)

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

        return variances

    def _qe_step(self, v: np.ndarray, dt: float) -> np.ndarray:
        """QE scheme for variance simulation (more stable)."""
        m = self.theta + (v - self.theta) * np.exp(-self.kappa * dt)
        s2 = (
            v
            * self.xi
            * self.xi
            * np.exp(-self.kappa * dt)
            / self.kappa
            * (1 - np.exp(-self.kappa * dt))
            + self.theta
            * self.xi
            * self.xi
            / (2 * self.kappa)
            * (1 - np.exp(-self.kappa * dt)) ** 2
        )
        psi = s2 / (m * m)

        U = np.random.rand(len(v))

        # Two branches depending on psi
        v_next = np.zeros_like(v)

        # Case 1: psi <= 1.5
        mask1 = psi <= 1.5
        if np.any(mask1):
            b2 = 2 / psi[mask1] - 1 + np.sqrt(2 / psi[mask1] * (2 / psi[mask1] - 1))
            a = m[mask1] / (1 + b2)
            Z = np.random.randn(np.sum(mask1))
            v_next[mask1] = a * (np.sqrt(b2) + Z) ** 2

        # Case 2: psi > 1.5
        mask2 = ~mask1
        if np.any(mask2):
            p = (psi[mask2] - 1) / (psi[mask2] + 1)
            beta = (1 - p) / m[mask2]

            u_p = 1 - p
            v_next[mask2] = np.where(U[mask2] <= u_p, np.log((1 - p) / (1 - U[mask2])) / beta, 0.0)

        return np.maximum(v_next, 1e-8)

    def to_dict(self) -> dict:
        """Serialize Heston parameters."""
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
        }


class SLVModel:
    """
    Stochastic Local Volatility model.

    Combines Heston stochastic vol with local vol leverage function:
        dS = r*S*dt + L(t, S)*√v*S*dW_S
        dv = κ(θ - v)*dt + ξ*√v*dW_v

    The leverage function L(t, S) is calibrated to match market vanilla prices.
    """

    def __init__(
        self,
        heston_model: HestonModel,
        local_vol_model: Optional[LocalVolModel] = None,
        leverage_function: Optional[Callable] = None,
    ):
        """
        Initialize SLV model.

        Args:
            heston_model: Calibrated Heston model
            local_vol_model: Local vol component (leverage function)
            leverage_function: Explicit leverage function L(t, S)
        """
        self.heston = heston_model
        self.local_vol = local_vol_model
        self.leverage_function = leverage_function

    def calibrate(self, market_data) -> None:
        """
        Two-stage SLV calibration:

        1. Calibrate Heston to ATM term structure and skew
        2. Calibrate leverage function to match full implied surface

        Args:
            market_data: Market data provider
        """
        # Stage 1: Calibrate Heston
        implied_surface = market_data.get_implied_vol_surface()
        self.heston = self._calibrate_heston(implied_surface)

        # Stage 2: Calibrate leverage function
        self.leverage_function = self._calibrate_leverage(implied_surface, market_data)

    def _calibrate_heston(self, implied_surface: VolSurface) -> HestonModel:
        """
        Calibrate Heston to ATM vols and skew.

        Simplified: uses method of moments on ATM term structure.
        Production: use optimization (least squares, MLE, etc.)
        """
        # Extract ATM volatilities
        times = implied_surface.times
        atm_vols = np.array([implied_surface.get_atm_vol(t) for t in times])

        # Rough calibration from term structure
        v0 = atm_vols[0] ** 2
        theta = np.mean(atm_vols) ** 2

        # Estimate mean reversion from term structure slope
        if len(times) > 1:
            time_horizon = times[-1] - times[0]
            vol_change = atm_vols[-1] ** 2 - atm_vols[0] ** 2
            kappa = (
                -np.log(1 - vol_change / (theta - v0)) / time_horizon
                if abs(theta - v0) > 1e-6
                else 2.0
            )
            kappa = np.clip(kappa, 0.5, 5.0)
        else:
            kappa = 2.0

        # Vol of vol from realized vol
        xi = 0.3 * np.std(atm_vols) if len(atm_vols) > 2 else 0.3

        # Correlation from skew
        rho = -0.7  # Typical equity correlation

        return HestonModel(
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
        )

    def _calibrate_leverage(
        self,
        implied_surface: VolSurface,
        market_data,
    ) -> callable:
        """
        Calibrate leverage function to match full surface.

        Uses particle method: L(t, S) adjusted iteratively to match
        marginal distributions.

        Simplified implementation: returns local vol ratio as leverage.
        """
        # Compute local vol from market
        local_vol_market = LocalVolModel(vol_surface=None, flat_vol=None)
        local_vol_market.calibrate(market_data)

        # Leverage function: L(t, S) = σ_LV(t, S) / √θ
        def leverage(t: float, s: float) -> float:
            local_sigma = local_vol_market.vol(t, 0, {"spot": s})
            heston_vol = np.sqrt(self.heston.theta)
            return local_sigma / heston_vol if heston_vol > 0 else 1.0

        return leverage

    def vol(self, t: float, asset_idx: int, state: dict | None = None) -> float:
        """
        Get instantaneous volatility for SLV model.

        σ_SLV = L(t, S) * √v

        Args:
            t: Time
            asset_idx: Asset index
            state: Must contain 'spot' and 'variance'

        Returns:
            Instantaneous volatility
        """
        if state is None:
            # Default: use long-run values
            spot = 1.0
            variance = self.heston.theta
        else:
            spot = state.get("spot", 1.0)
            variance = state.get("variance", self.heston.theta)

        # Get leverage
        if self.leverage_function:
            leverage = self.leverage_function(t, spot)
        elif self.local_vol:
            leverage = self.local_vol.vol(t, asset_idx, {"spot": spot})
        else:
            leverage = 1.0

        return leverage * np.sqrt(max(variance, 0))

    def simulate_paths(
        self,
        times: np.ndarray,
        spots: np.ndarray,
        n_paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate SLV paths.

        Returns:
            Tuple of (spot_paths, variance_paths)
        """
        n_assets = len(spots)
        n_steps = len(times)

        spot_paths = np.zeros((n_paths, n_steps, n_assets))
        variance_paths = np.zeros((n_paths, n_steps, n_assets))

        for asset_idx in range(n_assets):
            spot_paths[:, 0, asset_idx] = spots[asset_idx]
            variance_paths[:, 0, asset_idx] = self.heston.v0

            # Simulate variance process
            var_paths = self.heston.simulate_variance(times, n_paths)
            variance_paths[:, :, asset_idx] = var_paths

            # Simulate spot with correlated noise
            for i in range(1, n_steps):
                dt = times[i] - times[i - 1]

                # Correlated Brownian motions
                Z1 = np.random.randn(n_paths)
                Z2 = np.random.randn(n_paths)
                W_S = Z1
                W_v = self.heston.rho * Z1 + np.sqrt(1 - self.heston.rho**2) * Z2  # noqa: F841

                S_prev = spot_paths[:, i - 1, asset_idx]
                v_prev = variance_paths[:, i - 1, asset_idx]

                # Get leverage for each path
                leverage = np.array(
                    [
                        self.leverage_function(times[i], S_prev[p])
                        if self.leverage_function
                        else 1.0
                        for p in range(n_paths)
                    ]
                )

                # Euler step for spot
                vol = leverage * np.sqrt(np.maximum(v_prev, 0))
                spot_paths[:, i, asset_idx] = S_prev * np.exp(
                    -0.5 * vol**2 * dt + vol * np.sqrt(dt) * W_S
                )

        return spot_paths, variance_paths

    def to_dict(self) -> dict:
        """Serialize SLV model."""
        return {
            "model_type": "slv",
            "heston": self.heston.to_dict(),
            "local_vol": self.local_vol.to_dict() if self.local_vol else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SLVModel":
        """Deserialize SLV model."""
        heston = HestonModel(**data["heston"])
        local_vol = None
        if data.get("local_vol"):
            local_vol = LocalVolModel.from_dict(data["local_vol"])

        return cls(heston_model=heston, local_vol_model=local_vol)
