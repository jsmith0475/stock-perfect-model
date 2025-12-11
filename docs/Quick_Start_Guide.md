# Stock Perfect Model - Quick Start Guide

## Overview

The Stock Perfect Model identifies stock mispricings using **algebraic topology** and **graph signal processing**, enhanced with **AWS Bedrock LLM** integration for interpretable analysis.

---

## Installation

### Option 1: Use Pre-configured Environment

If you cloned the repo with the `venv/` folder:

```bash
cd stock-perfect-model
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Option 2: Fresh Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-perfect-model.git
cd stock-perfect-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from stock_perfect import StockPerfectModel; print('✅ Ready!')"
```

---

## Basic Usage

### Run from Command Line

```bash
python stock_perfect.py
```

### Run from Python

```python
from stock_perfect import StockPerfectModel

model = StockPerfectModel(
    tickers=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA'],
    start_date='2024-01-01',
    end_date='2024-12-01'
)

model.run_quantitative_pipeline()
```

---

## Understanding the Output

### Sample Output

```
======================================================================
RESIDUAL RANKINGS (with z-scores and signals)
======================================================================
ticker  residual   z_score       signal  abs_residual
  TSLA -0.008618 -2.297064 STRONG_SHORT      0.008618
  MSFT  0.005332  1.450915         LONG      0.005332
  NVDA  0.003985  1.088867         LONG      0.003985
  META -0.004140 -1.094031        SHORT      0.004140

======================================================================
TRADING SIGNALS
======================================================================
LONG candidates (z > 1):  ['MSFT', 'NVDA']
SHORT candidates (z < -1): ['TSLA', 'META']

======================================================================
REGIME STATUS
======================================================================
Regime:              STABLE
Position Sizing:     100% of normal
```

### Signal Interpretation

| Signal | Z-Score | Meaning | Recommendation |
|--------|---------|---------|----------------|
| **STRONG_LONG** | > +2 | Significantly undervalued | ✅ High conviction BUY |
| **LONG** | +1 to +2 | Undervalued vs peers | ✅ BUY opportunity |
| **NEUTRAL** | -1 to +1 | Fairly priced | ⏸️ No action |
| **SHORT** | -1 to -2 | Overvalued vs peers | ⚠️ SELL / avoid buying |
| **STRONG_SHORT** | < -2 | Significantly overvalued | ⚠️ High conviction SELL |

### What the Signals Mean

- **Positive residual (BUY):** Stock moved *less* than its correlated peers predicted → potentially undervalued → price may catch up
  
- **Negative residual (SELL):** Stock moved *more* than its correlated peers predicted → potentially overvalued → price may pull back

### Regime-Based Position Sizing

| Regime | H1 Persistence | Position Size | Market Condition |
|--------|----------------|---------------|------------------|
| **STABLE** | Low (< 0.1) | 100% | Simple correlation structure |
| **TRANSITIONING** | Medium (0.1-0.5) | 50% | Structure shifting |
| **FRAGMENTED** | High (> 0.5) | 25% | Complex/unstable market |

---

## Key Methods

### `get_residual_rankings()`

Returns DataFrame with all stocks ranked by mispricing:

```python
rankings = model.get_residual_rankings()
print(rankings)
```

| Column | Description |
|--------|-------------|
| `ticker` | Stock symbol |
| `residual` | Raw deviation from expected |
| `z_score` | Standardized score |
| `signal` | STRONG_LONG, LONG, NEUTRAL, SHORT, STRONG_SHORT |
| `abs_residual` | Absolute value for ranking |

### `get_trading_signals(min_z_score=1.0)`

Returns actionable BUY/SELL lists:

```python
signals = model.get_trading_signals(min_z_score=1.0)
print(f"BUY:  {signals['long']}")
print(f"SELL: {signals['short']}")
```

### `get_regime_status()`

Returns current market regime and position sizing:

```python
regime = model.get_regime_status()
print(f"Regime: {regime['regime']}")
print(f"Sizing: {regime['recommended_sizing']}")
```

---

## Full Pipeline with LLM

### Setup AWS Credentials

```bash
cp env.example.txt .env
# Edit .env with your AWS credentials:
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AWS_REGION=us-east-1
```

### Run Full Pipeline

```python
model.run_full_pipeline(
    interpret_regime=True,    # LLM regime analysis
    generate_report=True,     # LLM report generation
    save_report_path='daily_report.html'
)
```

This generates an HTML report with:
- Executive summary
- Market regime analysis
- Top opportunities with recommendations
- Risk warnings
- Position sizing guidance

---

## Daily Workflow Example

```python
from stock_perfect import StockPerfectModel
from datetime import datetime, timedelta

# Define universe
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'AMD', 'INTC', 'CRM'
]

# Set date range (trailing 1 year)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Initialize and run
model = StockPerfectModel(tickers, start_date, end_date)
model.run_quantitative_pipeline()

# Get signals
signals = model.get_trading_signals(min_z_score=1.0)
regime = model.get_regime_status()

# Print daily summary
print("="*50)
print(f"DAILY SIGNALS - {end_date}")
print("="*50)
print(f"✅ BUY:  {signals['long']}")
print(f"⚠️ SELL: {signals['short']}")
print(f"📊 Regime: {regime['regime']}")
print(f"⚖️ Position Sizing: {regime['recommended_sizing']}")
```

---

## Trading Strategy Guidelines

### Entry Rules

1. **BUY** when z-score > +1 (stronger signal at +2)
2. **SELL/SHORT** when z-score < -1 (stronger signal at -2)
3. Scale position size by regime multiplier

### Exit Rules

1. Exit when z-score returns to ~0 (mean reversion complete)
2. Stop-loss at 2% adverse move
3. Time stop: exit after 5 days if no movement

### Risk Management

| Rule | Value |
|------|-------|
| Max position size | 3% of portfolio |
| Max sector exposure | 15% of portfolio |
| Stop-loss | 2% per position |
| Reduce size when | H1 persistence high |

---

## Troubleshooting

### "ModuleNotFoundError"

```bash
source venv/bin/activate  # Make sure venv is active
pip install -r requirements.txt
```

### "No data fetched"

- Verify tickers are valid Yahoo Finance symbols
- Check date range contains trading days
- Some tickers may have missing data

### AWS Bedrock Errors

- Verify AWS credentials in `.env`
- Check your region supports Bedrock
- Ensure Claude models are enabled in your AWS account

---

## Cost Estimates (AWS Bedrock)

| Feature | Model | Cost per Run |
|---------|-------|--------------|
| Sentiment (per stock) | Haiku | ~$0.001 |
| Regime interpretation | Sonnet | ~$0.01 |
| Report generation | Sonnet | ~$0.02 |
| **Full pipeline (10 stocks)** | Mixed | **~$0.05** |

The quantitative pipeline (without LLM) is **free** and runs locally.

---

## Next Steps

1. **Read the Technical Paper** → [docs/Stock_Perfect_Model_Technical_Paper.md](Stock_Perfect_Model_Technical_Paper.md)
2. **Explore Architecture** → [docs/Architecture_Diagram.md](Architecture_Diagram.md)
3. **Customize your universe** → Edit tickers for your focus area
4. **Paper trade first** → Test signals before using real money

---

*Last updated: December 2024*
