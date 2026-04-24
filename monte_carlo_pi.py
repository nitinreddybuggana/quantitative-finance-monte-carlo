"""
Monte Carlo Simulation — Estimating π
======================================
Uses random sampling to estimate the value of π.

Concept:
  - A unit circle (radius=1) sits inside a 2×2 square.
  - Area of circle  = π × r² = π
  - Area of square  = (2r)²  = 4
  - Ratio           = π / 4
  - So: π ≈ 4 × (points inside circle / total points)

Author: [Your Name]
"""

import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ─── Core simulation ──────────────────────────────────────────────────────────

def estimate_pi(n_samples: int, seed: int = None) -> tuple[float, int, int]:
    """
    Run a Monte Carlo simulation to estimate π.

    Parameters
    ----------
    n_samples : int
        Number of random points to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pi_estimate : float
    inside_count : int
    total_count : int
    """
    if seed is not None:
        random.seed(seed)

    inside = 0
    for _ in range(n_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside += 1

    pi_estimate = 4 * inside / n_samples
    return pi_estimate, inside, n_samples


# ─── Convergence analysis ─────────────────────────────────────────────────────

def convergence_analysis(max_samples: int = 100_000, steps: int = 50) -> dict:
    """
    Track how the π estimate improves as sample size increases.

    Parameters
    ----------
    max_samples : int
        Maximum number of points to simulate.
    steps : int
        Number of data points to record along the way.

    Returns
    -------
    dict with 'sample_sizes', 'estimates', and 'errors'
    """
    sample_sizes = np.logspace(1, math.log10(max_samples), steps, dtype=int)
    estimates = []
    errors = []

    for n in sample_sizes:
        pi_est, _, _ = estimate_pi(int(n), seed=42)
        estimates.append(pi_est)
        errors.append(abs(pi_est - math.pi))

    return {
        "sample_sizes": sample_sizes.tolist(),
        "estimates": estimates,
        "errors": errors,
    }


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_simulation(n_samples: int = 5_000, seed: int = 42):
    """
    Visualise the random points inside and outside the unit circle.
    Blue dots landed inside the circle; red dots landed outside.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1, 1, n_samples)
    ys = rng.uniform(-1, 1, n_samples)
    inside = xs**2 + ys**2 <= 1

    pi_est = 4 * inside.sum() / n_samples

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs[inside],  ys[inside],  s=0.8, color="#185FA5", alpha=0.6, label="Inside")
    ax.scatter(xs[~inside], ys[~inside], s=0.8, color="#993C1D", alpha=0.6, label="Outside")

    circle = plt.Circle((0, 0), 1, fill=False, color="#0C447C", linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Monte Carlo π Estimation\nn={n_samples:,}  →  π ≈ {pi_est:.5f}", fontsize=13)
    ax.legend(loc="upper right", markerscale=8, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("monte_carlo_scatter.png", dpi=150)
    plt.show()
    print(f"Saved: monte_carlo_scatter.png")
    return pi_est


def plot_convergence(max_samples: int = 100_000):
    """
    Plot how the estimate converges toward π as n increases.
    """
    data = convergence_analysis(max_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left — estimate vs true π
    axes[0].semilogx(data["sample_sizes"], data["estimates"], color="#185FA5", linewidth=1.5, label="Estimate")
    axes[0].axhline(math.pi, color="#A32D2D", linewidth=1, linestyle="--", label=f"True π = {math.pi:.5f}")
    axes[0].set_title("π Estimate vs Sample Size")
    axes[0].set_xlabel("Number of samples (log scale)")
    axes[0].set_ylabel("Estimated π")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right — absolute error
    axes[1].loglog(data["sample_sizes"], data["errors"], color="#993C1D", linewidth=1.5, label="|error|")
    # Theoretical 1/√n decay line
    n_arr = np.array(data["sample_sizes"])
    ref = data["errors"][0] * np.sqrt(data["sample_sizes"][0]) / np.sqrt(n_arr)
    axes[1].loglog(n_arr, ref, color="#888780", linewidth=1, linestyle="--", label="1/√n reference")
    axes[1].set_title("Absolute Error vs Sample Size")
    axes[1].set_xlabel("Number of samples (log scale)")
    axes[1].set_ylabel("|Estimate − π|  (log scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("monte_carlo_convergence.png", dpi=150)
    plt.show()
    print("Saved: monte_carlo_convergence.png")


# ─── CLI demo ─────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 50)
    print("  Monte Carlo π Estimation — Demo")
    print("=" * 50)

    sample_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    print(f"\n{'Samples':>12}  {'Estimate':>12}  {'Error':>12}  {'Time (ms)':>10}")
    print("-" * 52)

    for n in sample_sizes:
        t0 = time.perf_counter()
        pi_est, inside, total = estimate_pi(n, seed=42)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        error = abs(pi_est - math.pi)
        print(f"{n:>12,}  {pi_est:>12.6f}  {error:>12.6f}  {elapsed_ms:>10.1f}")

    print(f"\n  True π = {math.pi:.10f}")
    print("\nGenerating plots...")
    plot_simulation(n_samples=5_000)
    plot_convergence(max_samples=100_000)
    print("\nDone!")


if __name__ == "__main__":
    run_demo()