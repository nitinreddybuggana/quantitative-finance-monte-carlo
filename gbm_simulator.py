"""
Geometric Brownian Motion — Stock Price Simulator
===================================================
Simulates future stock price paths using GBM, the stochastic
process underlying the Black-Scholes model.

The model:
    dS = μ·S·dt + σ·S·dW

Exact discretisation (no Euler approximation error):
    S(t+dt) = S(t) · exp[(μ - ½σ²)·dt + σ·√dt·Z],  Z ~ N(0,1)

Parameters:
    S0  — initial stock price
    mu  — annualised drift (expected return)
    sigma — annualised volatility
    T   — time horizon in years
    N   — number of simulated paths

Key outputs:
    - Simulated price paths
    - Percentile bands (10th / 50th / 90th)
    - Log-return distribution vs theoretical normal
    - Sensitivity analysis (drift and volatility)

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, lognorm


# ─── GBM Simulator ───────────────────────────────────────────────────────────

class GBMSimulator:
    """
    Simulates stock price paths using Geometric Brownian Motion.

    Under GBM, log-returns are normally distributed:
        ln(S_T / S_0) ~ N[(μ - ½σ²)T,  σ²T]

    Meaning the final price S_T follows a log-normal distribution.
    """

    def __init__(self, S0: float, mu: float, sigma: float,
                 T: float, N: int = 1000, steps_per_year: int = 252,
                 seed: int = None):
        """
        Parameters
        ----------
        S0             : Initial stock price
        mu             : Annualised drift (e.g. 0.08 = 8% per year)
        sigma          : Annualised volatility (e.g. 0.20 = 20%)
        T              : Time horizon in years
        N              : Number of simulation paths
        steps_per_year : Trading days per year (252 standard)
        seed           : Random seed for reproducibility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N
        self.steps = int(steps_per_year * T)
        self.dt = T / self.steps
        if seed is not None:
            np.random.seed(seed)

    def simulate(self) -> np.ndarray:
        """
        Generate N price paths.

        Returns
        -------
        paths : ndarray of shape (N, steps + 1)
            Each row is one simulated price path.
        """
        drift     = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)

        Z         = np.random.standard_normal((self.N, self.steps))
        log_ret   = drift + diffusion * Z
        log_paths = np.hstack([np.zeros((self.N, 1)), np.cumsum(log_ret, axis=1)])

        return self.S0 * np.exp(log_paths)

    def analytical_stats(self) -> dict:
        """
        Closed-form statistics of the final price distribution.
        S_T is log-normally distributed — these are exact, not simulated.
        """
        mu, sigma, S0, T = self.mu, self.sigma, self.S0, self.T

        mean   = S0 * np.exp(mu * T)
        median = S0 * np.exp((mu - 0.5 * sigma**2) * T)
        mode   = S0 * np.exp((mu - 1.5 * sigma**2) * T)
        var    = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)
        std    = np.sqrt(var)

        return {'mean': mean, 'median': median, 'mode': mode,
                'std': std, 'variance': var}


# ─── Analysis functions ───────────────────────────────────────────────────────

def percentile_band(paths: np.ndarray, p: float) -> np.ndarray:
    return np.percentile(paths, p, axis=0)


def prob_above(paths: np.ndarray, target: float) -> float:
    """Probability that the final price exceeds a target."""
    return (paths[:, -1] > target).mean()


def log_returns(paths: np.ndarray) -> np.ndarray:
    """Compute step-by-step log returns across all paths."""
    return np.diff(np.log(paths), axis=1).flatten()


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_simulation(S0=100, mu=0.08, sigma=0.20, T=1.0, N=200, seed=42):
    """
    Four-panel analysis plot:
      1. Simulated price paths with percentile bands
      2. Final price distribution vs log-normal theoretical
      3. Log-return distribution vs theoretical normal
      4. Sensitivity: final price distribution across volatility levels
    """
    sim   = GBMSimulator(S0, mu, sigma, T, N, seed=seed)
    paths = sim.simulate()
    stats = sim.analytical_stats()
    t     = np.linspace(0, T, sim.steps + 1)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('GBM Stock Price Simulator', fontsize=15, y=0.98)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    # ── Panel 1: Price paths ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for path in paths[:80]:
        ax1.plot(t, path, lw=0.5, alpha=0.3, color='#378ADD')

    p10 = percentile_band(paths, 10)
    p50 = percentile_band(paths, 50)
    p90 = percentile_band(paths, 90)

    ax1.fill_between(t, p10, p90, alpha=0.15, color='#BA7517', label='10th–90th pct')
    ax1.plot(t, p50, color='#1D9E75', lw=2,   label='Median path')
    ax1.plot(t, S0 * np.exp(mu * t), color='#A32D2D', lw=1.5,
             linestyle='--', label='Expected (e^μt)')
    ax1.set_title('Simulated GBM paths', fontsize=11)
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Stock price ($)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Final price distribution ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    finals = paths[:, -1]
    ax2.hist(finals, bins=50, density=True, color='#185FA5',
             alpha=0.6, label='Simulated')

    x = np.linspace(finals.min(), finals.max(), 300)
    mu_ln    = np.log(S0) + (mu - 0.5 * sigma**2) * T
    sigma_ln = sigma * np.sqrt(T)
    ax2.plot(x, lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln)),
             color='#A32D2D', lw=2, label='Log-normal (theoretical)')
    ax2.axvline(stats['mean'],   color='#1D9E75', lw=1.5, linestyle='--',
                label=f'Mean ${stats["mean"]:.1f}')
    ax2.axvline(stats['median'], color='#BA7517', lw=1.5, linestyle=':',
                label=f'Median ${stats["median"]:.1f}')
    ax2.set_title('Final price distribution', fontsize=11)
    ax2.set_xlabel('Stock price at T ($)')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Log-return distribution ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    lr = log_returns(paths)
    ax3.hist(lr, bins=80, density=True, color='#534AB7',
             alpha=0.6, label='Simulated log-returns')
    x2 = np.linspace(lr.min(), lr.max(), 300)
    ax3.plot(x2, norm.pdf(x2, loc=(mu - 0.5*sigma**2)*sim.dt,
                          scale=sigma*np.sqrt(sim.dt)),
             color='#A32D2D', lw=2, label='Normal (theoretical)')
    ax3.set_title('Log-return distribution', fontsize=11)
    ax3.set_xlabel('Log return per step')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: Volatility sensitivity ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vols   = [0.10, 0.20, 0.35, 0.55]
    colors = ['#185FA5', '#1D9E75', '#BA7517', '#A32D2D']
    for v, c in zip(vols, colors):
        s2   = GBMSimulator(S0, mu, v, T, N=500, seed=seed)
        f    = s2.simulate()[:, -1]
        ax4.hist(f, bins=50, density=True, alpha=0.45, color=c,
                 label=f'σ={int(v*100)}%')
    ax4.axvline(S0, color='#444441', lw=1.5, linestyle='--', label=f'S₀=${S0}')
    ax4.set_title('Final price by volatility', fontsize=11)
    ax4.set_xlabel('Final stock price ($)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    plt.savefig('gbm_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: gbm_simulation.png')


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(S0=100, mu=0.08, sigma=0.20, T=1.0, N=10_000):
    print('=' * 58)
    print('  GBM Stock Price Simulation Report')
    print('=' * 58)
    print(f'  S₀=${S0}  μ={mu*100:.1f}%  σ={sigma*100:.0f}%  T={T}yr  N={N:,}')
    print('-' * 58)

    sim    = GBMSimulator(S0, mu, sigma, T, N, seed=42)
    paths  = sim.simulate()
    stats  = sim.analytical_stats()
    finals = paths[:, -1]

    print(f'\n  {"":28} {"Simulated":>12} {"Analytical":>12}')
    print(f'  {"-"*54}')
    print(f'  {"Mean final price":28} ${finals.mean():>10.2f} ${stats["mean"]:>10.2f}')
    print(f'  {"Median final price":28} ${np.median(finals):>10.2f} ${stats["median"]:>10.2f}')
    print(f'  {"Std dev of final price":28} ${finals.std():>10.2f} ${stats["std"]:>10.2f}')

    for pct in [5, 10, 25, 75, 90, 95]:
        print(f'  {f"{pct}th percentile":28} ${np.percentile(finals, pct):>10.2f}')

    pop = prob_above(paths, S0) * 100
    print(f'\n  Probability final price > S₀: {pop:.1f}%')
    print(f'  Probability final price > 2×S₀: {prob_above(paths, 2*S0)*100:.1f}%')
    print(f'  Probability of >20% loss:       {prob_above(paths, S0*0.8)*100:.1f}% above 0.8×S₀')
    print('=' * 58)


if __name__ == '__main__':
    print_report()
    print('\nGenerating plots...')
    plot_simulation()