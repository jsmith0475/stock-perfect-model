# Stock Perfect Model

**A quantitative trading algorithm using algebraic topology and graph signal processing**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is this?

Stock Perfect Model is a **mathematical trading framework** that identifies stock mispricings using:

- **Graph Signal Processing** — Models the market as a correlation graph
- **Laplacian Diffusion** — Separates market-wide movements from stock-specific deviations  
- **Persistent Homology** — Detects regime changes in correlation structure
- **LLM Integration** — AWS Bedrock for sentiment analysis and report generation

No neural networks. No black boxes. **Pure mathematics.**

---

## 📊 Sample Output

```
======================================================================
TRADING SIGNALS
======================================================================
Ticker   Signal    Z-Score   Recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TSLA     SELL      -2.30     ⚠️ Overextended - avoid buying
MSFT     BUY       +1.45     ✅ Undervalued vs peers
NVDA     BUY       +1.09     ✅ Undervalued vs peers
META     SELL      -1.09     ⚠️ Overextended - avoid buying

======================================================================
REGIME STATUS
======================================================================
Regime:              STABLE
H1 Persistence:      0.0000
Position Sizing:     100% of normal
```

### How to Read Signals

| Signal | Z-Score | Meaning | Action |
|--------|---------|---------|--------|
| **BUY** | > +1 | Stock lagging peers → undervalued | Good entry point |
| **STRONG BUY** | > +2 | Significantly undervalued | High conviction entry |
| **SELL** | < -1 | Stock ahead of peers → overextended | Avoid or short |
| **STRONG SELL** | < -2 | Significantly overextended | High conviction short |
| **NEUTRAL** | -1 to +1 | Fairly priced | No action |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-perfect-model.git
cd stock-perfect-model
```

### 2. Set Up Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Model

```bash
python stock_perfect.py
```

### 4. (Optional) Configure AWS for LLM Features

```bash
cp env.example.txt .env
# Edit .env with your AWS credentials
```

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

# Run quantitative pipeline
model.run_quantitative_pipeline()

# Get trading signals
print(model.get_residual_rankings())
print(model.get_trading_signals())
print(model.get_regime_status())
```

### Full Pipeline with LLM Analysis

```python
model.run_full_pipeline(
    interpret_regime=True,
    generate_report=True,
    save_report_path='daily_report.html'
)
```

---

## 🧮 The Mathematics

### 1. Market Graph Construction

Stocks are nodes, correlations define edge weights:

```
Correlation Distance:  d_ij = √(2(1 - ρ_ij))
Gaussian Kernel:       W_ij = exp(-d²/2σ²)
```

### 2. Laplacian Diffusion

Separate market-wide movements from stock-specific deviations:

```
Normalized Laplacian:  L = I - D^(-½) W D^(-½)
Heat Kernel:           H = exp(-tL)
Residual:              r = signal - H × signal
```

### 3. Signal Generation

Z-score normalization identifies mispricings:

```
z_i = (residual_i - μ) / σ

BUY signal:   z > +1  (stock undervalued vs peers)
SELL signal:  z < -1  (stock overvalued vs peers)
```

### 4. Regime Detection

Persistent homology detects structural changes:

```
H1 Persistence low  → Stable market    → Full position sizing
H1 Persistence high → Fragmented market → Reduce positions
```

---

## 📁 Project Structure

```
stock-perfect-model/
├── stock_perfect.py       # Main model
├── bedrock_client.py      # AWS Bedrock LLM client
├── news_sentiment.py      # News sentiment analysis
├── regime_interpreter.py  # Topology regime interpretation
├── report_generator.py    # Report generation
├── requirements.txt       # Dependencies
├── env.example.txt        # AWS credentials template
├── README.md              # This file
└── docs/
    ├── Stock_Perfect_Model_Technical_Paper.md
    ├── Quick_Start_Guide.md
    └── Architecture_Diagram.md
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp env.example.txt .env
```

Required for LLM features:
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

## 📚 Documentation

- **[Technical Paper](docs/Stock_Perfect_Model_Technical_Paper.md)** — Full mathematical derivations and proofs
- **[Quick Start Guide](docs/Quick_Start_Guide.md)** — Detailed usage instructions
- **[Architecture Diagram](docs/Architecture_Diagram.md)** — System design and data flow

---

## ⚠️ Disclaimer

**This is a research project, not financial advice.**

- Past performance does not guarantee future results
- Always do your own research before trading
- The authors are not responsible for any financial losses
- Use at your own risk

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ripser](https://github.com/scikit-tda/ripser.py) for persistent homology computation
- [yfinance](https://github.com/ranaroussi/yfinance) for financial data
- [AWS Bedrock](https://aws.amazon.com/bedrock/) for LLM integration
- The algebraic topology and TDA research community

---

## 📬 Contact

Questions? Ideas? Open an issue or reach out on LinkedIn.

**Star ⭐ this repo if you find it useful!**

