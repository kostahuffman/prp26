"""
Visualization tools for structured products analysis.

This module provides institutional-grade visualization for:
- Monte Carlo path analysis
- Greek exposures and sensitivities
- P&L tracking and attribution
- Risk heat maps
- Portfolio-level analytics
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class StructuredProductVisualizer:
    """Visualization suite for structured products."""

    def __init__(self, product=None, save_plots: bool = False, output_dir: str = "./plots"):
        """Initialize visualizer.

        Args:
            product: StructuredProduct instance (optional)
            save_plots: Whether to save plots to disk
            output_dir: Directory for saving plots
        """
        self.product = product
        self.save_plots = save_plots
        self.output_dir = output_dir

    def plot_monte_carlo_paths(
        self,
        paths: np.ndarray,
        times: np.ndarray,
        tickers: list[str],
        initial_spots: np.ndarray,
        n_paths_to_show: int = 100,
        figsize: tuple = (14, 8),
    ):
        """Plot Monte Carlo path samples.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            times: Time grid (n_steps,)
            tickers: List of ticker names
            initial_spots: Initial spot prices (n_assets,)
            n_paths_to_show: Number of paths to display
            figsize: Figure size
        """
        n_paths, n_steps, n_assets = paths.shape

        fig, axes = plt.subplots(1, n_assets, figsize=figsize, sharey=False)
        if n_assets == 1:
            axes = [axes]

        for i, (ax, ticker) in enumerate(zip(axes, tickers, strict=False)):
            # Normalize to percentage of initial
            normalized_paths = paths[:n_paths_to_show, :, i] / initial_spots[i] * 100

            # Plot individual paths
            for path_idx in range(n_paths_to_show):
                ax.plot(
                    times,
                    normalized_paths[path_idx, :],
                    alpha=0.1,
                    color="steelblue",
                    linewidth=0.5,
                )

            # Plot mean path
            mean_path = np.mean(paths[:, :, i], axis=0) / initial_spots[i] * 100
            ax.plot(times, mean_path, color="red", linewidth=2, label="Mean Path", alpha=0.8)

            # Add 100% reference line
            ax.axhline(y=100, color="black", linestyle="--", linewidth=1, alpha=0.5)

            ax.set_title(
                f"{ticker} Paths (showing {n_paths_to_show}/{n_paths})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Time (Years)")
            ax.set_ylabel("% of Initial Spot")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/monte_carlo_paths.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_payoff_distribution(
        self,
        payoffs: np.ndarray,
        notional: float = 1_000_000,
        figsize: tuple = (12, 6),
    ):
        """Plot distribution of payoffs.

        Args:
            payoffs: Discounted payoffs per path (n_paths,)
            notional: Contract notional
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(payoffs, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax1.axvline(
            np.mean(payoffs),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: ${np.mean(payoffs):,.0f}",
        )
        ax1.axvline(
            np.median(payoffs),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: ${np.median(payoffs):,.0f}",
        )
        ax1.set_title("Payoff Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Payoff ($)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CDF
        sorted_payoffs = np.sort(payoffs)
        cdf = np.arange(1, len(sorted_payoffs) + 1) / len(sorted_payoffs)
        ax2.plot(sorted_payoffs, cdf, color="steelblue", linewidth=2)
        ax2.set_title("Cumulative Distribution Function", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Payoff ($)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.grid(True, alpha=0.3)

        # Add percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(payoffs, p)
            ax2.axhline(y=p / 100, color="red", linestyle=":", alpha=0.3)
            ax2.text(val, p / 100, f" P{p}: ${val:,.0f}", fontsize=8, va="bottom")

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/payoff_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_greeks_dashboard(
        self,
        greeks: dict[str, Any],
        spots: dict[str, float],
        figsize: tuple = (14, 10),
    ):
        """Plot comprehensive Greeks dashboard.

        Args:
            greeks: Greeks dict with delta, gamma, vega, rho
            spots: Current spot prices
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Delta bar chart
        ax1 = fig.add_subplot(gs[0, :])
        tickers = list(greeks["delta"].keys())
        deltas = list(greeks["delta"].values())
        colors = ["green" if d > 0 else "red" for d in deltas]
        bars = ax1.bar(tickers, deltas, color=colors, alpha=0.7, edgecolor="black")
        ax1.set_title("Delta Exposure ($ per 1% spot move)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Delta ($)")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${height:,.0f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
            )

        # Gamma bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        gammas = list(greeks["gamma"].values())
        colors_gamma = ["blue" if g > 0 else "orange" for g in gammas]
        ax2.bar(tickers, gammas, color=colors_gamma, alpha=0.7, edgecolor="black")
        ax2.set_title("Gamma (Convexity)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Gamma")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis="y")

        # Rho display
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        rho_text = f"Rho ($ per 1bp rate move)\n\n${greeks['rho']:,.2f}"
        ax3.text(
            0.5,
            0.5,
            rho_text,
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        # Spot ladder (Delta vs Spot shocks)
        ax4 = fig.add_subplot(gs[2, :])
        spot_shocks = np.linspace(-20, 20, 9)  # -20% to +20%

        for ticker in tickers:
            delta = greeks["delta"][ticker]
            spot = spots[ticker]
            # Approximate P&L from delta
            pnl_approx = delta * spot_shocks / 100 * spot
            ax4.plot(spot_shocks, pnl_approx, marker="o", label=ticker, linewidth=2)

        ax4.set_title(
            "Approximate P&L vs Spot Shocks (Delta Approximation)", fontsize=12, fontweight="bold"
        )
        ax4.set_xlabel("Spot Shock (%)")
        ax4.set_ylabel("Approximate P&L ($)")
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Greeks Risk Dashboard", fontsize=16, fontweight="bold", y=0.995)

        if self.save_plots:
            plt.savefig(f"{self.output_dir}/greeks_dashboard.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_pnl_history(
        self,
        pnl_history: list[dict],
        figsize: tuple = (14, 8),
    ):
        """Plot P&L history over time.

        Args:
            pnl_history: List of dicts with keys: date, pv, pnl, spots
            figsize: Figure size
        """
        dates = [entry["date"] for entry in pnl_history]
        pvs = [entry["pv"] for entry in pnl_history]
        pnls = [entry["pnl"] for entry in pnl_history]

        # Extract spot prices
        tickers = list(pnl_history[0]["spots"].keys())
        spot_series = {
            ticker: [entry["spots"][ticker] for entry in pnl_history] for ticker in tickers
        }

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # PV over time
        ax1.plot(dates, pvs, marker="o", linewidth=2, markersize=8, color="steelblue")
        ax1.set_title("Present Value Over Time", fontsize=14, fontweight="bold")
        ax1.set_ylabel("PV ($)")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        # Cumulative P&L
        colors_pnl = ["green" if p >= 0 else "red" for p in pnls]
        ax2.bar(dates, pnls, color=colors_pnl, alpha=0.7, edgecolor="black")
        ax2.set_title("Cumulative P&L", fontsize=14, fontweight="bold")
        ax2.set_ylabel("P&L ($)")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}K"))

        # Spot prices
        for ticker in tickers:
            ax3.plot(dates, spot_series[ticker], marker="o", linewidth=2, label=ticker)
        ax3.set_title("Underlying Spot Prices", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Spot Price ($)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/pnl_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_termination_analysis(
        self,
        termination_times: np.ndarray,
        observation_times: list[float],
        figsize: tuple = (12, 6),
    ):
        """Plot termination time distribution.

        Args:
            termination_times: When each path terminated (n_paths,)
            observation_times: Observation dates
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram of termination times
        ax1.hist(
            termination_times,
            bins=len(observation_times),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax1.set_title("Early Termination Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Termination Time (Years)")
        ax1.set_ylabel("Number of Paths")
        ax1.grid(True, alpha=0.3, axis="y")

        # Survival curve
        unique_times = sorted(set(termination_times))
        survival_rates = []
        for t in unique_times:
            survival_rate = np.mean(termination_times >= t)
            survival_rates.append(survival_rate)

        ax2.plot(
            unique_times, survival_rates, marker="o", linewidth=2, markersize=8, color="steelblue"
        )
        ax2.set_title(
            "Survival Curve (Probability of Not Being Called)", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Time (Years)")
        ax2.set_ylabel("Survival Probability")
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)

        # Add observation markers
        for obs_time in observation_times:
            ax2.axvline(x=obs_time, color="red", linestyle="--", alpha=0.3, linewidth=1)

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/termination_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_worst_of_heatmap(
        self,
        paths: np.ndarray,
        times: np.ndarray,
        tickers: list[str],
        initial_spots: np.ndarray,
        figsize: tuple = (12, 6),
    ):
        """Plot worst-of performance heatmap over time.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            times: Time grid
            tickers: Ticker names
            initial_spots: Initial spots
            figsize: Figure size
        """
        n_paths, n_steps, n_assets = paths.shape

        # Calculate performance for each asset at each time
        performances = paths / initial_spots[np.newaxis, np.newaxis, :] * 100

        # Calculate worst-of at each time step across all paths
        worst_of_per_step = np.min(performances, axis=2)  # (n_paths, n_steps)

        # Calculate percentiles for worst-of
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(worst_of_per_step, percentiles, axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Percentile fan chart
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(percentiles)))
        for i, (p, vals) in enumerate(zip(percentiles, percentile_values, strict=False)):
            ax1.plot(times, vals, label=f"P{p}", linewidth=2, color=colors[i])

        ax1.fill_between(
            times, percentile_values[0], percentile_values[-1], alpha=0.2, color="gray"
        )
        ax1.axhline(
            y=100, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Initial Level"
        )
        ax1.set_title("Worst-Of Performance Percentiles", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Time (Years)")
        ax1.set_ylabel("Performance (% of Initial)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Heatmap of worst-of frequency
        bins_time = len(times)
        bins_perf = 20
        perf_range = [60, 140]  # 60% to 140%

        hist, xedges, yedges = np.histogram2d(
            np.tile(times, n_paths),
            worst_of_per_step.flatten(),
            bins=[bins_time, bins_perf],
            range=[[times[0], times[-1]], perf_range],
        )

        im = ax2.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], perf_range[0], perf_range[1]],
            cmap="YlOrRd",
        )
        ax2.axhline(
            y=100, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Initial Level"
        )
        ax2.set_title("Worst-Of Performance Density", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Time (Years)")
        ax2.set_ylabel("Performance (% of Initial)")
        ax2.legend()
        plt.colorbar(im, ax=ax2, label="Frequency")

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/worst_of_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()


def quick_plot_pnl(pnl_history: list[dict], title: str = "P&L History"):
    """Quick one-liner for P&L visualization.

    Args:
        pnl_history: List of P&L history dicts
        title: Plot title
    """
    viz = StructuredProductVisualizer()
    viz.plot_pnl_history(pnl_history)


def quick_plot_greeks(greeks: dict, spots: dict, title: str = "Greeks Dashboard"):
    """Quick one-liner for Greeks visualization.

    Args:
        greeks: Greeks dictionary
        spots: Current spot prices
        title: Plot title
    """
    viz = StructuredProductVisualizer()
    viz.plot_greeks_dashboard(greeks, spots)
