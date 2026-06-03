# Quantitative Finance — Monte Carlo Methods

A portfolio of three projects exploring Monte Carlo simulation from first principles through to real-world stock price forecasting, built as a first-year Computer Engineering student with an interest in quantitative finance.

The projects are designed to be read in order. Each one inherits the mathematical foundation of the last and pushes it into new territory.

---

## The progression

```
Random sampling          Stochastic processes        Real-world forecasting
       │                         │                          │
  Estimate π               Simulate stock             Calibrate from real
  by geometry              price paths                historical data
       │                         │                          │
  Law of Large             Geometric               Probability distributions
  Numbers, 1/√N            Brownian Motion         of future outcomes
  convergence              Itô's lemma             Risk & return analysis
```

All three share the same core engine: generate enough random samples, and the average converges to the true answer. What changes across the projects is what that "true answer" represents — a geometric constant, a theoretical price distribution, or a real forecast built from live market data.

---

## Projects

### 1. Monte Carlo Pi Estimation — `monte_carlo_pi.py`

The entry point. Estimates π using nothing but random points and geometry: throw darts at a square, count how many land inside the inscribed circle, and the ratio converges to π/4.

This project establishes the two ideas everything else builds on. First, the **Law of Large Numbers** — repeated random sampling converges to the true expected value. Second, the **1/√N convergence rate** — to halve your error, you need four times as many samples. Both of these reappear, unchanged, in the stock forecaster.

**Key concepts:** Law of Large Numbers · geometric probability · Monte Carlo convergence · 1/√N error decay

```bash
pip install matplotlib numpy
python monte_carlo_pi.py
```

---

### 2. Monte Carlo Stock Forecaster — `stock_forecaster.py`

The bridge between pure probability and real-world finance. Instead of sampling random points in a square, we now sample random shocks to a real stock price — fetching live historical data, calibrating the model, and simulating thousands of possible futures.

The model is **Geometric Brownian Motion (GBM)**. At each time step, the stock receives a normally distributed random shock scaled by volatility. The shocks are multiplicative (not additive), which means log-returns are normally distributed and prices follow a **log-normal distribution**. The −½σ² correction in the exponent comes from **Itô's lemma**: when you apply a nonlinear function (the exponential) to a random process, you pick up an extra curvature term that must be corrected for.

What makes this project distinct from a purely theoretical simulation is the calibration step: we estimate μ (drift) and σ (volatility) directly from real historical returns fetched via Yahoo Finance. The model is then grounded in actual market behaviour — not assumed parameters.

**Key concepts:** Geometric Brownian Motion · Itô's lemma · log-normal distribution · model calibration from real data · percentile fan charts · probability of profit/loss

```bash
pip install numpy matplotlib scipy yfinance
python stock_forecaster.py
```

---

## How the mathematics connects

Every project reduces to the same equation:

$$\text{Answer} \approx \frac{1}{N} \sum_{i=1}^{N} f(X_i)$$

where X_i is a random sample, and f is some function of interest.

| Project | X_i | f(X_i) | Answer |
|---|---|---|---|
| Pi estimation | Uniform point (x, y) | 1 if x²+y² ≤ 1, else 0 | π/4 |
| Stock forecaster | Random shock Z ~ N(0,1) | GBM price path S(t) | Distribution of future prices |

The error in both cases shrinks at the same rate: **σ/√N**, where σ is the standard deviation of f(X). This is the Central Limit Theorem — the unifying theorem beneath both projects.

---

## Repository structure

```
quantitative-finance-monte-carlo/
│
├── README.md                  ← you are here
├── monte_carlo_pi.py          # Phase 1: Pi estimation
└── stock_forecaster.py        # Phase 2: Stock price forecasting
```

---

## Author

Nitin Reddy Buggana — First-year Computer Engineering student  
Interests: quantitative finance, stochastic modelling, algorithmic systems

---

## Licence

MIT
