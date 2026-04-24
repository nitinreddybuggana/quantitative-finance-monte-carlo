# Quantitative Finance — Monte Carlo Methods

A portfolio of three projects exploring Monte Carlo simulation from first principles through to financial derivatives pricing, built as a first-year Computer Engineering student with an interest in quantitative finance.

The projects are designed to be read in order. Each one inherits the mathematical foundation of the last and pushes it into new territory.

---

## The progression

```
Random sampling          Stochastic processes        Derivatives pricing
       │                         │                          │
  Estimate π               Simulate stock             Price options &
  by geometry              price futures              compute Greeks
       │                         │                          │
  Law of Large             Geometric               Risk-neutral measure
  Numbers, 1/√N            Brownian Motion         Black-Scholes benchmark
  convergence              Itô's lemma             Confidence intervals
```

All three share the same core engine: generate enough random samples, and the average converges to the true answer. What changes across the projects is what that "true answer" represents — a geometric constant, a price distribution, or a fair value for a financial contract.

---

## Projects

### 1. Monte Carlo Pi Estimation — `monte_carlo_pi/`

The entry point. Estimates π using nothing but random points and geometry: throw darts at a square, count how many land inside the inscribed circle, and the ratio converges to π/4.

This project establishes the two ideas everything else builds on. First, the **Law of Large Numbers** — repeated random sampling converges to the true expected value. Second, the **1/√N convergence rate** — to halve your error, you need four times as many samples. Both of these reappear, unchanged, in the options pricer.

**Key concepts:** Law of Large Numbers · geometric probability · Monte Carlo convergence · 1/√N error decay

```bash
cd monte_carlo_pi
pip install matplotlib numpy
python monte_carlo.py
```

---

### 2. GBM Stock Price Simulator — `gbm_simulator/`

The bridge between pure probability and finance. Instead of sampling points in a square, we now sample random shocks to a stock price — and watch how uncertainty compounds over time.

The model is Geometric Brownian Motion. At each time step, the stock receives a normally distributed random shock scaled by volatility. Crucially, the shocks are multiplicative (not additive), which means log-returns are normal and prices are **log-normally distributed** — a fact that becomes essential in the next project.

The −½σ² correction in the exponent is the first appearance of **Itô's lemma**: when you apply a nonlinear function (the exponential) to a random process, you pick up an extra curvature term. This is not an approximation — it is exact.

**Key concepts:** Geometric Brownian Motion · Itô's lemma · log-normal distribution · drift vs volatility · percentile fan charts

```bash
cd gbm_simulator
pip install numpy matplotlib scipy
python gbm_simulator.py
```

---

### 3. Monte Carlo Options Pricing — `mc_options_pricing/`

The destination. A European option pays max(S_T − K, 0) at expiry — a nonlinear function of a random variable. There is no simple formula for most options, so we price them by simulation: generate thousands of GBM paths from project 2, compute the payoff at the end of each, discount back to today, and average. That average is the fair price.

This project introduces two new ideas on top of the GBM engine. The **risk-neutral measure** — we replace the real-world drift μ with the risk-free rate r before simulating, which removes the need to estimate expected returns and gives a unique, arbitrage-free price. And the **Black-Scholes analytical solution**, which we use as a benchmark: when the number of simulations is large, our Monte Carlo price converges to it, validating the entire chain of reasoning from project 1 through to here.

The five **Greeks** — Delta, Gamma, Vega, Theta, Rho — are then computed analytically from Black-Scholes, and measure how sensitive the option price is to each input parameter.

**Key concepts:** Risk-neutral pricing · discounted expected payoff · Black-Scholes · put-call parity · option Greeks · confidence intervals

```bash
cd mc_options_pricing
pip install numpy scipy matplotlib
python monte_carlo_options.py
```

---

## How the mathematics connects

Every project reduces to the same equation:

$$\text{Answer} \approx \frac{1}{N} \sum_{i=1}^{N} f(X_i)$$

where X_i is a random sample and f is some function of interest.

| Project | X_i | f(X_i) | Answer |
|---|---|---|---|
| Pi estimation | Uniform point (x, y) | 1 if x²+y² ≤ 1, else 0 | π/4 |
| GBM simulator | Random shock Z ~ N(0,1) | Stock price path S(t) | Distribution of S_T |
| Options pricing | GBM path ending at S_T | e^(−rT) · max(S_T − K, 0) | Fair option price C |

The error in all three cases shrinks at the same rate: **σ/√N**, where σ is the standard deviation of f(X). This is the Central Limit Theorem — the unifying theorem beneath all three projects.

---

## Repository structure

```
quantitative-finance-monte-carlo/
│
├── README.md                            ← you are here
│
├── monte_carlo_pi.py
├── gbm_simulator.py
└── options_pricing.py
```

---

## Author

Nitin Reddy Buggana — First-year Computer Engineering student  
Interests: quantitative finance, stochastic modelling, algorithmic systems

---

## Licence

MIT
