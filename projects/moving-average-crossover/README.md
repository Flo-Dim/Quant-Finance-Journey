# 📈 SMA / EMA Crossover Strategy

> **Goal:**  
> Test a simple moving average (SMA) and exponential moving average (EMA) crossover strategy on Apple (AAPL) to assess whether a trend-following approach can outperform a passive buy-and-hold strategy.

---

## 🧠 Overview

This project implements and compares **two versions** of a moving-average crossover system:

1. **Manual SMA version** – built from scratch using `pandas` and `matplotlib`.  
2. **Framework EMA version** – using the `backtesting.py` library for structured simulation and performance reporting.

Both aim to capture medium-term trends while avoiding large drawdowns during market reversals.

---

## 📊 Data

| Parameter | Description |
|------------|--------------|
| **Universe / Ticker** | AAPL (Apple Inc.) |
| **Date Range** | ~5 years (2020 – 2025) |
| **Source** | Yahoo Finance (`yfinance`) |
| **Data Fields** | Open, High, Low, Close, Volume |
| **Storage** | Cached locally under `data/` (not committed) |

---

## ⚙️ Methodology

### 💡 Strategy Logic
- Compute two moving averages:
  - **Fast MA:** 20 days  
  - **Slow MA:** 50 days  
- **Buy** when fast MA crosses **above** slow MA  
- **Exit** when fast MA crosses **below** slow MA  
- Framework version uses **EMA** for faster response.  
- Manual version uses **SMA** with a 2-day confirmation filter to avoid noise.

### 🧾 Assumptions
| Parameter | Value / Description |
|------------|--------------------|
| **Rebalancing** | Daily on confirmed signals |
| **Commission** | 0.1% per trade (`0.001`) |
| **Signal Shift** | Trades executed at close after signal |
| **Initial Cash** | \$10,000 |
| **Exposure** | ~60% of total market days |

---

## 📈 Results

| Metric | Manual SMA | Framework EMA |
|:--------|:------------|:--------------|
| **Total Return** | ~8% | ~45% |
| **Buy & Hold Return** | ~100% | ~100% |
| **Sharpe Ratio** | 0.20 | 0.38 |
| **Max Drawdown** | -30% | -25% |
| **Market Exposure** | ~55–60% | ~60% |

---

## 🖼️ Visuals

| Figure | Description |
|:--------|:-------------|
| **Figure 1** | AAPL with 20-day & 50-day SMAs |
| **Figure 2** | Cumulative Performance: Buy & Hold vs. SMA Strategy |
| **Figure 3** | Framework EMA 20/50 Crossover with trade markers |

🗂️ Saved under: `figures/`

---

## 💬 Takeaways

### ✅ What Worked
- The crossover effectively captured medium-term trends.  
- EMA version reduced lag and improved responsiveness.  
- Lower drawdowns versus passive exposure.

### ❌ What Didn’t
- Frequent **false signals** in sideways markets.  
- Missed large parts of rallies in strong bull trends (e.g., Apple 2020–2025).  
- Sensitivity to parameter choice (20/50 too short for trending assets).

### 🔄 Next Steps
- Test longer MA pairs (e.g., **50/200-day**).  
- Apply **market regime filters** (trade only when index > 200 MA).  
- Introduce **volatility or momentum filters**.  
- Expand to **multi-asset portfolios** (FX, commodities).

---

## 🧪 Reproduction

### 🗂️ Scripts
| File | Description |
|------|--------------|
| [`SMA_CrosOver_Strategy.py`](./SMA_CrosOver_Strategy.py) | Manual implementation with SMA & cost model |
| [`SMA_CrossOver_using_Framework.py`](./SMA_CrossOver_using_Framework.py) | Framework-based EMA crossover using `backtesting.py` |
| [`SMA Cross-Over Strategy.docx`](./SMA%20Cross-Over%20Strategy.docx) | Full documentation & performance discussion |

### 🧮 Notebook
Saved under 'notebooks'

### 🔧 Parameters to Tweak
- `fast` and `slow` MA windows (default: 20 / 50)  
- `commission` and trading cost assumptions  
- Signal confirmation period  
- MA type (`SMA` or `EMA`)  
- Target ticker, date range, and interval

---

## 📚 Requirements
- install the requirements.txt 
