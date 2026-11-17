import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output/figures")


def plot_convergence(objective_func_vals: list[float], title: str = "QAOA Convergence") -> None:
    """Plot and save the QAOA optimizer convergence curve.

    A smoothly decreasing curve that plateaus is the signature of convergence.
    A noisy, non-monotonic curve usually indicates too few shots, poor initial
    parameters, or a circuit dominated by hardware noise.

    Args:
        objective_func_vals: List of cost values recorded at each iteration.
        title:               Title for the plot.
    """
    fig = plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "large_scale_convergence.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {output_path}")

    plt.show()
    plt.close(fig)


def _plot_cdf(objective_values: dict[float, float], ax, color: str) -> None:
    """Draw the cumulative distribution function line on the given axes.

    Args:
        objective_values: Dict mapping objective values to their probability.
        ax:               Matplotlib axes to draw on.
        color:            Line color.
    """
    x_vals = sorted(objective_values.keys(), reverse=True)
    y_vals = np.cumsum([objective_values[x] for x in x_vals])
    ax.plot(x_vals, y_vals, color=color)


def plot_cdf(dist: dict[float, float], ax, title: str) -> None:
    """Plot the cumulative distribution function of the objective values.

    Highlights the best (minimum) cost found with a vertical dashed line.
    Used to visually assess the quality of the QAOA solution distribution
    at utility scale.

    Args:
        dist:  Dict mapping objective values to their probability.
        ax:    Matplotlib axes to draw on.
        title: Title for the subplot (typically the backend name).
    """
    _plot_cdf(dist, ax, "C1")
    ax.vlines(min(list(dist.keys())), 0, 1, "C1", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Objective function value")
    ax.set_ylabel("Cumulative distribution function")
    ax.grid(alpha=0.3)


def save_cdf_figure(dist: dict[float, float], title: str) -> None:
    """Create, display, and save the CDF figure to output/figures/.

    Args:
        dist:  Dict mapping objective values to their probability.
        title: Title for the plot (typically the backend name).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_cdf(dist, ax, title)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "large_scale_cdf.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {output_path}")

    plt.show()
    plt.close(fig)
