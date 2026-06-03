"""
Monte Carlo Stock Forecaster
=============================
Uses real historical stock data to calibrate a GBM model,
then simulates thousands of possible future price paths.

Workflow:
    1. Fetch real historical price data via yfinance
    2. Calibrate μ (drift) and σ (volatility) from log-returns
    3. Simulate N future price paths using GBM
    4. Analyse the distribution of outcomes
    5. Visualise paths, percentile bands, and return distribution

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─── Data fetching & calibration ─────────────────────────────────────────────

def fetch_data(ticker: str, period: str = '2y') -> tuple:
    """
    Fetch historical closing prices from Yahoo Finance.

    Parameters
    ----------
    ticker : str   e.g. 'AAPL', 'TSLA', 'SPY'
    period : str   e.g. '1y', '2y', '5y'

    Returns
    -------
    prices : ndarray of closing prices
    dates  : DatetimeIndex
    """
    print(f"Fetching {ticker} data ({period} history)...")
    stock = yf.Ticker(ticker)
    df    = stock.history(period = period)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    prices = df['Close'].values
    dates  = df.index
    print(f"  Loaded {len(prices)} trading days  |  "
          f"${prices[0]:.2f} → ${prices[-1]:.2f}")
    return prices, dates


def calibrate(prices: np.ndarray) -> tuple[float, float]:
    """
    Estimate annualised drift (μ) and volatility (σ) from historical prices.

    Method: compute daily log-returns, then annualise.
        μ_daily = mean(log-returns)
        σ_daily = std(log-returns)
        μ_annual = μ_daily × 252  +  ½σ_annual²   (Itô correction)
        σ_annual = σ_daily × √252

    The Itô correction converts the log-return drift (under the log-normal
    measure) back to the arithmetic drift of the price process.

    Returns
    -------
    mu    : annualised drift
    sigma : annualised volatility
    """
    log_returns = np.diff(np.log(prices))
    mu_daily    = log_returns.mean()
    sigma_daily = log_returns.std()

    sigma = sigma_daily * np.sqrt(252)
    mu    = mu_daily * 252 + 0.5 * sigma**2

    print(f"  Calibrated:  μ = {mu*100:.2f}%/yr   σ = {sigma*100:.2f}%/yr")
    return mu, sigma


# ─── Simulation ───────────────────────────────────────────────────────────────

class StockForecaster:
    """
    Monte Carlo forecaster calibrated from real historical data.
    Simulates future price paths using exact GBM discretisation.
    """

    def __init__(self, ticker: str = 'AAPL', forecast_days: int = 30,
                 n_simulations: int = 1000, history_period: str = '2y',
                 seed: int = None):
        """
        Parameters
        ----------
        ticker          : Stock ticker symbol
        forecast_days   : Number of trading days to forecast ahead
        n_simulations   : Number of Monte Carlo paths
        history_period  : Historical data window for calibration
        seed            : Random seed for reproducibility
        """
        self.ticker        = ticker.upper()
        self.forecast_days = forecast_days
        self.n_sims        = n_simulations
        self.history_period = history_period

        if seed is not None:
            np.random.seed(seed)

        # Fetch and calibrate
        self.prices, self.dates = fetch_data(self.ticker, self.history_period)
        self.S0                 = self.prices[-1]
        self.mu, self.sigma     = calibrate(self.prices)
        self.paths              = None

    def simulate(self) -> np.ndarray:
        """
        Simulate N future price paths over forecast_days trading days.

        Returns
        -------
        paths : ndarray of shape (n_simulations, forecast_days + 1)
                paths[:, 0] = S0 (today's price) for all paths
        """
        dt        = 1 / 252                          # one trading day
        drift     = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z         = np.random.standard_normal((self.n_sims, self.forecast_days))
        log_ret   = drift + diffusion * Z
        log_paths = np.hstack([np.zeros((self.n_sims, 1)),
                               np.cumsum(log_ret, axis=1)])

        self.paths = self.S0 * np.exp(log_paths)
        return self.paths

    def summary(self) -> dict:
        """
        Compute summary statistics from simulated final prices.
        """
        if self.paths is None:
            self.simulate()

        finals = self.paths[:, -1]
        T      = self.forecast_days / 252

        return {
            'current_price'  : self.S0,
            'mean'           : finals.mean(),
            'median'         : np.median(finals),
            'std'            : finals.std(),
            'p5'             : np.percentile(finals, 5),
            'p10'            : np.percentile(finals, 10),
            'p25'            : np.percentile(finals, 25),
            'p75'            : np.percentile(finals, 75),
            'p90'            : np.percentile(finals, 90),
            'p95'            : np.percentile(finals, 95),
            'prob_profit'    : (finals > self.S0).mean() * 100,
            'prob_5pct_gain' : (finals > self.S0 * 1.05).mean() * 100,
            'prob_5pct_loss' : (finals < self.S0 * 0.95).mean() * 100,
            'expected_return': (finals.mean() / self.S0 - 1) * 100,
            'analytical_mean': self.S0 * np.exp(self.mu * T),
        }


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_forecast(forecaster: StockForecaster, n_paths_shown: int = 100):
    """
    Four-panel visualisation:
      1. Historical price + simulated forecast paths with percentile bands
      2. Distribution of final simulated prices
      3. Historical log-return distribution vs normal fit
      4. Summary statistics table
    """
    if forecaster.paths is None:
        forecaster.simulate()

    paths  = forecaster.paths
    finals = paths[:, -1]
    stats  = forecaster.summary()

    # Build forecast date axis (trading days only)
    last_date     = forecaster.dates[-1].to_pydatetime()
    forecast_dates = [last_date + timedelta(days=int(i * 365/252))
                      for i in range(forecaster.forecast_days + 1)]

    # Percentile bands across time
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{forecaster.ticker} — Monte Carlo Stock Forecast '
                 f'({forecaster.forecast_days}-day, {forecaster.n_sims:,} simulations)',
                 fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    # ── Panel 1: Historical + forecast ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # full-width top panel

    # Historical (last 60 days for context)
    hist_slice  = min(60, len(forecaster.prices))
    hist_dates  = [d.to_pydatetime() for d in forecaster.dates[-hist_slice:]]
    hist_prices = forecaster.prices[-hist_slice:]
    ax1.plot(hist_dates, hist_prices, color='#444441', lw=2, label='Historical', zorder=5)

    # Simulated paths (sample)
    for path in paths[:n_paths_shown]:
        ax1.plot(forecast_dates, path, lw=0.4, alpha=0.2, color='#378ADD')

    # Percentile bands
    ax1.fill_between(forecast_dates, p10, p90, alpha=0.12,
                     color='#BA7517', label='10th–90th pct')
    ax1.fill_between(forecast_dates, p25, p75, alpha=0.18,
                     color='#1D9E75', label='25th–75th pct')
    ax1.plot(forecast_dates, p50, color='#1D9E75', lw=2,
             linestyle='--', label='Median forecast', zorder=4)
    ax1.axvline(last_date, color='#A32D2D', lw=1.5,
                linestyle=':', label='Today', zorder=6)
    ax1.axhline(forecaster.S0, color='#A32D2D', lw=1,
                linestyle='--', alpha=0.5, label=f'Current ${forecaster.S0:.2f}')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.set_title('Historical price & simulated forecast paths', fontsize=11)
    ax1.set_ylabel('Price ($)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Final price distribution ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    itm = finals[finals >= forecaster.S0]
    otm = finals[finals <  forecaster.S0]
    ax2.hist(otm, bins=50, color='#993C1D', alpha=0.7, label='Below current')
    ax2.hist(itm, bins=50, color='#185FA5', alpha=0.7, label='Above current')
    ax2.axvline(forecaster.S0,    color='#444441', lw=2,   linestyle='--',
                label=f'Current ${forecaster.S0:.2f}')
    ax2.axvline(stats['median'],  color='#1D9E75', lw=1.5, linestyle=':',
                label=f'Median ${stats["median"]:.2f}')
    ax2.axvline(stats['p5'],      color='#BA7517', lw=1,   linestyle=':',
                label=f'5th pct ${stats["p5"]:.2f}')
    ax2.axvline(stats['p95'],     color='#BA7517', lw=1,   linestyle=':',
                label=f'95th pct ${stats["p95"]:.2f}')
    ax2.set_title(f'Final price distribution (day {forecaster.forecast_days})', fontsize=11)
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Log-return distribution ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    log_returns = np.diff(np.log(forecaster.prices))
    ax3.hist(log_returns, bins=60, density=True, color='#534AB7',
             alpha=0.6, label='Historical log-returns')
    x  = np.linspace(log_returns.min(), log_returns.max(), 300)
    ax3.plot(x, norm.pdf(x, log_returns.mean(), log_returns.std()),
             color='#A32D2D', lw=2, label='Normal fit')
    ax3.set_title('Historical log-return distribution', fontsize=11)
    ax3.set_xlabel('Daily log-return')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)
    mu_d  = log_returns.mean()
    sig_d = log_returns.std()
    ax3.text(0.97, 0.95, f'μ={mu_d*100:.3f}%/d\nσ={sig_d*100:.2f}%/d',
             transform=ax3.transAxes, fontsize=9, va='top', ha='right',
             color='#444441',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.savefig(f'{forecaster.ticker}_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {forecaster.ticker}_forecast.png')


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(forecaster: StockForecaster):
    if forecaster.paths is None:
        forecaster.simulate()

    s = forecaster.summary()
    print('=' * 58)
    print(f'  {forecaster.ticker} — {forecaster.forecast_days}-Day Monte Carlo Forecast')
    print('=' * 58)
    print(f'  Simulations   : {forecaster.n_sims:,}')
    print(f'  Current price : ${s["current_price"]:.2f}')
    print(f'  Forecast horizon: {forecaster.forecast_days} trading days\n')
    print(f'  {"Price targets":}')
    print(f'  {"-"*40}')
    print(f'  {"Mean (analytical)":28} ${s["analytical_mean"]:>8.2f}')
    print(f'  {"Mean (simulated)":28} ${s["mean"]:>8.2f}')
    print(f'  {"Median":28} ${s["median"]:>8.2f}')
    print(f'  {"Std deviation":28} ${s["std"]:>8.2f}')
    print(f'\n  {"Percentiles":}')
    print(f'  {"-"*40}')
    for label, key in [('5th', 'p5'), ('10th', 'p10'), ('25th', 'p25'),
                       ('75th', 'p75'), ('90th', 'p90'), ('95th', 'p95')]:
        print(f'  {label+" percentile":28} ${s[key]:>8.2f}')
    print(f'\n  {"Probabilities":}')
    print(f'  {"-"*40}')
    print(f'  {"P(price > current)":28} {s["prob_profit"]:>7.1f}%')
    print(f'  {"P(gain > 5%)":28} {s["prob_5pct_gain"]:>7.1f}%')
    print(f'  {"P(loss > 5%)":28} {s["prob_5pct_loss"]:>7.1f}%')
    print(f'  {"Expected return":28} {s["expected_return"]:>7.2f}%')
    print('=' * 58)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    forecaster = StockForecaster(
        ticker          = 'AAPL',
        forecast_days   = 30,
        n_simulations   = 1000,
        history_period  = '2y',
        seed            = 42
    )

    forecaster.simulate()
    print_report(forecaster)
    print('\nGenerating plots...')
    plot_forecast(forecaster)

