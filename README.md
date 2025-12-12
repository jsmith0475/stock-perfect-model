# Stock Perfect Model

**A quantitative trading algorithm using algebraic topology, graph signal processing, and time-series analysis**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is this?

Stock Perfect Model is a **geometric trading framework** that identifies stock mispricings using advanced mathematics:

- **Graph Signal Processing** — Models the market as a correlation manifold
- **Laplacian Diffusion** — Separates market-wide movements from stock-specific deviations  
- **Graph Fourier Transform (GFT)** — Analyzes market coherence and spectral entropy
- **Walsh-Hadamard Transform** — Classifies signals as "elastic" (mean-reverting) vs "drifting" (trending)
- **Persistent Homology** — Detects regime changes in correlation structure via topology
- **LLM Integration** — AWS Bedrock for sentiment analysis and narrative reports

**No neural networks. No black boxes. Pure mathematics with geometric intuition.**

---

## 📊 Sample Output

```
======================================================================
TRADING SIGNALS (Regime Adjusted)
======================================================================
Regime Multiplier: 1.00
  → This is the GLOBAL position sizing based on market topology/spectral risk
  → Factors: Stable Market Regime

✅ BUY  (Count: 2)
   Format: [Ticker] (Z-score) | Walsh Class (Score) | Final Size
   Legend: Z-score = How unusual (>1 = tradeable) | Walsh = Mean-reversion quality (>0.3 = Elastic)
   - MSFT  (z= 1.4) | Walsh: Elastic  (0.58) | Size: 100%
   - NVDA  (z= 1.1) | Walsh: Elastic  (0.54) | Size: 100%

⚠️  SELL (Count: 2)
   - TSLA  (z=-2.3) | Walsh: Elastic  (0.51) | Size: 100%
   - META  (z=-1.1) | Walsh: Elastic  (0.47) | Size: 100%

======================================================================
REGIME STATUS
======================================================================
Regime:              STABLE
Description:         Stable Market Regime | High Market Coherence | High Elasticity (Mean Reverting)
H1 Persistence:      0.0000  (Low < 0.1 = Stable | High > 0.5 = Fragmented)
Market Coherence:    0.5385  (High > 0.7 = Beta-driven | Low < 0.3 = Stock-picking)
Spectral Entropy:    0.5268  (High > 0.8 = Chaotic | Low < 0.5 = Orderly)
Avg Elasticity:      0.5127  (High > 0.4 = Mean-reverting market)
Position Sizing:     100% of normal
```

### Interpreting the Metrics

| Metric | Range | What it means | Good for trading |\n|--------|-------|---------------|------------------|\n| **Z-Score** | -∞ to +∞ | How unusual the residual is | > ±1.0 |\n| **Walsh Score** | 0.0 - 1.0 | How "elastic" (mean-reverting) | > 0.3 |\n| **Market Coherence** | 0.0 - 1.0 | How much market moves as one block | 0.3 - 0.6 |\n| **Spectral Entropy** | 0.0 - 1.0 | Market chaos/disorder | < 0.6 |\n| **H1 Persistence** | 0.0 - ∞ | Topological complexity | < 0.1 |\n| **Regime Multiplier** | 0.0 - 1.0 | Global position sizing | 1.0 |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jsmith0475/stock-perfect-model.git
cd stock-perfect-model
```

### 2. Set Up Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Run the Model

```bash
python stock_perfect.py
```

The model will:
1. Fetch stock data (1 year historical)
2. Build correlation graph
3. Compute Laplacian residuals
4. Perform spectral analysis (GFT)
5. Analyze Walsh stability (time-series)
6. Compute persistent homology (topology)
7. Output trading signals with regime-adjusted sizing

---

## 📖 Usage

### Basic Analysis (No AWS Required)

```python
from stock_perfect import StockPerfectModel

model = StockPerfectModel(
    tickers=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA'],
    start_date='2024-01-01',
    end_date='2024-12-01'
)

# Run quantitative pipeline (Graph + Spectral + Walsh + Topology)
model.run_quantitative_pipeline()

# Get regime-adjusted trading signals
signals = model.get_trading_signals(min_z_score=1.0)
print(signals)

# Get regime status
regime = model.get_regime_status()
print(regime)
```

### Understanding Position Sizing

```python
# Position size is calculated as:
# Final Size = Regime Multiplier × Walsh Multiplier

# Example scenarios:
# - 100% = Stable regime (H1 low) + Elastic stock (Walsh > 0.3)
# - 50%  = Transitioning regime OR Mixed elasticity
# - 0%   = Fragmented regime OR Drifting stock (AVOID)
```

---

## 🧮 The Mathematics

### 1. Market Graph Construction

Stocks are nodes in a **correlation manifold**:

```
Correlation Distance:  d_ij = √(2(1 - ρ_ij))
Gaussian Kernel:       W_ij = exp(-d²/2σ²)
Graph Laplacian:       L = I - D^(-½) W D^(-½)
```

### 2. Laplacian Diffusion (Spatial Analysis)

Heat kernel separates market-wide from idiosyncratic returns:

```
Heat Kernel:  H = exp(-tL)
Smoothed:     r_smooth = H × r_actual
Residual:     r_residual = r_actual - r_smooth
```

**Interpretation:** Residual = Stock's deviation from what the graph structure predicts

### 3. Graph Fourier Transform (Frequency Analysis)

Decomposes returns into frequency modes:

```
GFT:              f̂ = U^T × r  (U = Laplacian eigenvectors)
Power Spectrum:   p = f̂² / sum(f̂²)
Market Coherence: p[0]  (energy in λ₀ mode)
Spectral Entropy: -Σ(p log p) / log(N)
```

**Interpretation:**
- High Coherence (>0.7) → Market moving as one block (beta-driven)
- High Entropy (>0.8) → Chaotic, no clear driver

### 4. Walsh-Hadamard Transform (Time Analysis)

Classifies residual behavior using sign-flips:

```
Sequency Score = (Number of Sign Flips) / (Total Days - 1)

Elastic (>0.3):   Stock oscillates (mean-reverting) → TRADE
Drifting (<0.15): Stock trends (fundamental shift) → AVOID
```

**Interpretation:** Prevents "catching falling knives"

### 5. Signal Generation

Multi-factor position sizing:

```
Z-Score:  z_i = (residual_i - μ) / σ
Signal:   |z| > 1 = Tradeable
Walsh:    Score > 0.3 = Elastic (good for mean-reversion)
Regime:   H1 < 0.1 = Stable (full size)
          H1 > 0.5 = Fragmented (reduce size)
```

### 6. Regime Detection (Topology)

Persistent homology ($H_1$ loops) detects structural fragmentation:

```
H1 Persistence = Σ(death - birth) for all correlation loops

Low  (<0.1):  Tree-like structure → Stable → 100% sizing
High (>0.5):  Circular dependencies → Fragmented → 25% sizing
```

---

## 📁 Project Structure

```
stock-perfect-model/
├── stock_perfect.py            # Main model (GFT + Walsh + Topology)
├── tasks_phase_2_spectral.md   # Spectral analysis task plan
├── tasks_phase_3_walsh.md      # Walsh transform task plan
├── bedrock_client.py           # AWS Bedrock LLM client
├── news_sentiment.py           # News sentiment analysis
├── regime_interpreter.py       # Topology regime interpretation
├── report_generator.py         # Report generation
├── requirements.txt            # Dependencies
├── env.example.txt             # AWS credentials template
├── .gitignore                  # Excludes docs/ from repo
└── README.md                   # This file
```

**Note:** `docs/` directory (Medium articles, technical papers) is excluded from the public repo.

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp env.example.txt .env
```

Required for LLM features (optional):
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffusion_time` | 1.0 | Heat kernel smoothing (0.5-2.0 recommended) |
| `min_z_score` | 1.0 | Minimum z-score for trading signal |
| `sigma` | auto | Gaussian kernel bandwidth |

---

## 🎓 Key Concepts

### Why Geometry?

Traditional factor models assume markets are **flat** (linear space). But:
- Correlations change (the space curves)
- Loops form during regime shifts (the manifold tears)
- Linear regression can't see structural breaks

**Solution:** Treat the market as a **geometric manifold** and use topology to detect when the shape breaks.

### The Three-Layer Framework

1. **Graph Layer (WHERE?)** — Laplacian finds stocks deviating from local equilibrium
2. **Spectral Layer (WHEN?)** — GFT + Topology detect regime fragmentation
3. **Time Layer (HOW?)** — Walsh classifies signals as elastic vs drifting

---

## ⚠️ Disclaimer

**This is a research project, not financial advice.**

- Past performance does not guarantee future results
- The model uses historical correlations which may not hold in the future
- Always do your own research before trading
- The authors are not responsible for any financial losses
- Use at your own risk

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Rolling window topology analysis
- Multi-scale Walsh transforms
- Alternative graph kernels
- Backtesting frameworks

Process:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ripser](https://github.com/scikit-tda/ripser.py) for persistent homology computation
- [yfinance](https://github.com/ranaroussi/yfinance) for financial data
- [AWS Bedrock](https://aws.amazon.com/bedrock/) for LLM integration
- The algebraic topology and topological data analysis research community
- Graph signal processing literature (Shuman et al., 2013)

---

## 📚 Further Reading

- **Medium Article:** "The Market Has a Shape. And It's Broken." (Coming soon)
- **Technical Concepts:**
  - [Graph Signal Processing on Wikipedia](https://en.wikipedia.org/wiki/Graph_signal_processing)
  - [Persistent Homology Tutorial](https://www.math.upenn.edu/~ghrist/preprints/barcodes.pdf)
  - [Spectral Graph Theory](https://mathworld.wolfram.com/SpectralGraphTheory.html)

---

## 📬 Contact

Questions? Ideas? 
- Open an issue on GitHub
- Connect on [LinkedIn](https://linkedin.com/in/yourprofile)

**Star ⭐ this repo if you find it useful!**
