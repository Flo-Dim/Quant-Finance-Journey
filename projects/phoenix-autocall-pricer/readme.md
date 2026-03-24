# Phoenix Autocall Pricer — Monte Carlo (Python)

> **Work in progress** — this project is actively being developed. Structure and results may change.

---
## Overview

This project implements a **Phoenix Autocall (no memory)** pricer with Monte Carlo simulation of the underlying (GBM), path-dependent payoff logic, early redemption, conditional coupon payments, and a capital-at-risk feature at maturity.

The objective is to replicate how a **structuring desk would prototype a product**. This project is meant for learning purposes and not a production-grade library.

---

## Data

| Parameter | Description |
|---|---|
| Underlying | EURO STOXX 50 (`^STOXX50E`) |
| Source | Yahoo Finance (`yfinance`) |
| Volatility | Historical, annualized |
| Spot | Latest available closing price |
| Trading days | 252 |

---

## Methodology

### Model

The underlying follows **Geometric Brownian Motion** under risk-neutral dynamics. Paths are simulated using a fully vectorised NumPy implementation.

### Product Logic — Phoenix Autocall (no memory)

At each monthly observation date:

- **Autocall** — if $S_t \geq$ autocall barrier: product terminates, investor receives notional + coupon
- **Coupon** — if coupon barrier $\leq S_t <$ autocall barrier: coupon is paid
- **No coupon** — if $S_t <$ coupon barrier: coupon is lost (no memory)

At maturity, if never autocalled:

- $S_T \geq$ protection barrier: full notional returned
- $S_T <$ protection barrier: capital loss proportional to underlying performance ($N \times S_T / S_0$)

All cash flows are discounted continuously from their respective payment dates.

### Assumptions

| Parameter | Value |
|---|---|
| Model | Black-Scholes GBM |
| Volatility | Flat (constant) |
| Interest rate | Flat |
| Dividends | Not modelled |
| Observation dates | 12 monthly |
| Coupon memory | None |
| Discounting | Continuous |

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Entry point — runs pricing with live market data |
| `pricer/model.py` | Vectorised GBM path simulation |
| `pricer/product.py` | Phoenix payoff logic |
| `pricer/pricer.py` | Monte Carlo engine |
| `utils/market.py` | Market data retrieval via yfinance |
| `playground.py` | Interactive Gradio UI — shareable live demo |

---

## Interactive Playground

`playground.py` launches a local Gradio interface with a public shareable link (valid 72h). Sliders expose all key parameters in real time: spot, volatility, rate, maturity, notional, coupon rate, and all three barrier levels.

```bash
python playground.py
```

---

## Takeaways

**What works**
- Realistic path-dependent payoff with correct per-date discounting
- Clean model / product / pricer separation
- Vectorised simulation (~100x faster than a Python loop)

**What is simplified**
- Flat volatility (no surface)
- No dividends, no stochastic rates
- Single underlying only

**Next steps**
- Coupon memory feature
- Greeks via finite differencing (Delta, Vega)
- Variance reduction (antithetic variates, control variate)
- Worst-of / basket autocallables
- Implied volatility calibration

---

## Reproduction

```bash
pip install numpy pandas yfinance matplotlib gradio
python main.py
```
