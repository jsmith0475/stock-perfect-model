# Stock Perfect Model - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STOCK PERFECT MODEL                                │
│                    Topological Trading Framework                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │    DATA     │ │  ANALYSIS   │ │   OUTPUT    │
            │   LAYER     │ │   LAYER     │ │   LAYER     │
            └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Detailed Data Flow

```
                              ┌──────────────────┐
                              │   Yahoo Finance  │
                              │       API        │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA ACQUISITION                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │   Tickers   │───▶│   Prices    │───▶│ Log Returns │                   │
│  │   List      │    │  DataFrame  │    │  DataFrame  │                   │
│  └─────────────┘    └─────────────┘    └─────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         GRAPH CONSTRUCTION                                │
│                                                                          │
│  Returns ───▶ Correlation ───▶ Distance ───▶ Gaussian ───▶ Adjacency    │
│    (N×T)       Matrix         Matrix        Kernel        Matrix        │
│                (N×N)          (N×N)         (N×N)         (N×N)         │
│                                                                          │
│              ρᵢⱼ = corr(rᵢ,rⱼ)   dᵢⱼ = √(2(1-ρ))   Wᵢⱼ = e^(-d²/2σ²)   │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────┐
│      GRAPH SIGNAL PROCESSING    │  │       ALGEBRAIC TOPOLOGY            │
│                                 │  │                                     │
│  ┌─────────────────────────┐    │  │  ┌─────────────────────────────┐    │
│  │   Degree Matrix (D)     │    │  │  │   Vietoris-Rips Complex     │    │
│  │   D = diag(∑W)          │    │  │  │   from Distance Matrix      │    │
│  └───────────┬─────────────┘    │  │  └───────────┬─────────────────┘    │
│              ▼                  │  │              ▼                      │
│  ┌─────────────────────────┐    │  │  ┌─────────────────────────────┐    │
│  │   Normalized Laplacian  │    │  │  │   Persistent Homology       │    │
│  │   L = I - D⁻¹/²WD⁻¹/²   │    │  │  │   via Ripser               │    │
│  └───────────┬─────────────┘    │  │  └───────────┬─────────────────┘    │
│              ▼                  │  │              ▼                      │
│  ┌─────────────────────────┐    │  │  ┌─────────────────────────────┐    │
│  │   Heat Kernel           │    │  │  │   Persistence Diagrams      │    │
│  │   H = exp(-tL)          │    │  │  │   H₀: Components            │    │
│  └───────────┬─────────────┘    │  │  │   H₁: Loops                 │    │
│              ▼                  │  │  └───────────┬─────────────────┘    │
│  ┌─────────────────────────┐    │  │              ▼                      │
│  │   Residuals             │    │  │  ┌─────────────────────────────┐    │
│  │   r_res = r - H·r       │    │  │  │   Topological Features      │    │
│  └─────────────────────────┘    │  │  │   • H1 Total Persistence    │    │
│                                 │  │  │   • H1 Max Persistence      │    │
│  Output: Mispricing signals     │  │  │   • Feature Counts          │    │
│                                 │  │  └─────────────────────────────┘    │
└─────────────────────────────────┘  │                                     │
                │                    │  Output: Regime features            │
                │                    └─────────────────────────────────────┘
                │                                    │
                └──────────────┬─────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         LLM INTEGRATION (AWS BEDROCK)                     │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  │
│  │  NEWS SENTIMENT    │  │ REGIME INTERPRETER │  │  REPORT GENERATOR  │  │
│  │  (Claude Haiku)    │  │  (Claude Sonnet)   │  │  (Claude Sonnet)   │  │
│  │                    │  │                    │  │                    │  │
│  │  • Fetch news      │  │  • Analyze H0/H1   │  │  • Executive sum   │  │
│  │  • Score sentiment │  │  • Classify regime │  │  • Opportunities   │  │
│  │  • Extract themes  │  │  • Risk assessment │  │  • Risk warnings   │  │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘  │
│           │                       │                       │              │
└───────────┼───────────────────────┼───────────────────────┼──────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                       │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐  │
│  │   Residual     │  │   Regime       │  │      Trading Report        │  │
│  │   Rankings     │  │   State        │  │      (HTML/Markdown)       │  │
│  │                │  │                │  │                            │  │
│  │  NVDA: +1.2%   │  │  Type: Stable  │  │  • Market analysis         │  │
│  │  AAPL: -0.8%   │  │  Risk: Low     │  │  • Top opportunities       │  │
│  │  MSFT: +0.5%   │  │  Action: Full  │  │  • Position recommendations│  │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                           ┌─────────────────┐
                           │ stock_perfect.py│
                           │   (Main Model)  │
                           └────────┬────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│news_sentiment.py│      │regime_interpreter│     │report_generator │
│                 │      │       .py        │     │      .py        │
└────────┬────────┘      └────────┬────────┘     └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │bedrock_client.py│
                        │  (AWS Bedrock)  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │      boto3      │
                        │   (AWS SDK)     │
                        └─────────────────┘
```

---

## External Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │Yahoo Finance │    │  AWS Bedrock │    │ Google News  │       │
│  │     API      │    │   (Claude)   │    │    (RSS)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │ Price Data        │ LLM Inference     │ News Articles  │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  STOCK PERFECT MODEL                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Core Data Flow

```
Input:
  tickers: List[str]           # ['AAPL', 'MSFT', ...]
  start_date: str              # '2024-01-01'
  end_date: str                # '2024-12-01'

Intermediate:
  data: DataFrame              # (T × N) price matrix
  returns: DataFrame           # (T-1 × N) log returns
  correlation_matrix: ndarray  # (N × N) correlations
  adj_matrix: ndarray          # (N × N) graph weights
  residuals: ndarray           # (N × T-1) mispricing signals

Topology:
  diagrams: List[ndarray]      # Persistence diagrams
  h1_total_persistence: float  # Sum of H1 lifetimes
  h1_max_persistence: float    # Max H1 lifetime
  h1_feature_count: int        # Number of H1 features

Output:
  rankings: DataFrame          # Stocks sorted by |residual|
  regime: RegimeInterpretation # LLM regime analysis
  report: TradingReport        # Full trading report
```

---

## Sequence Diagram

```
User          StockPerfect    yfinance    scipy    ripser    Bedrock
  │                │             │          │         │          │
  │  run_pipeline  │             │          │         │          │
  │───────────────▶│             │          │         │          │
  │                │ download    │          │         │          │
  │                │────────────▶│          │         │          │
  │                │◀────────────│          │         │          │
  │                │             │          │         │          │
  │                │ expm(L)     │          │         │          │
  │                │─────────────────────▶  │         │          │
  │                │◀─────────────────────  │         │          │
  │                │             │          │         │          │
  │                │ ripser(dist)│          │         │          │
  │                │────────────────────────────────▶ │          │
  │                │◀──────────────────────────────── │          │
  │                │             │          │         │          │
  │                │ interpret   │          │         │          │
  │                │───────────────────────────────────────────▶ │
  │                │◀─────────────────────────────────────────── │
  │                │             │          │         │          │
  │  report.html   │             │          │         │          │
  │◀───────────────│             │          │         │          │
  │                │             │          │         │          │
```

---

## File Structure

```
Stock Perfect Model/
│
├── stock_perfect.py        # Main model class
│   ├── StockPerfectModel
│   │   ├── __init__()
│   │   ├── fetch_data()
│   │   ├── compute_returns()
│   │   ├── build_graph()
│   │   ├── compute_laplacian_residuals()
│   │   ├── compute_topology()
│   │   ├── analyze_sentiment()
│   │   ├── interpret_regime()
│   │   ├── generate_report()
│   │   ├── run_quantitative_pipeline()
│   │   └── run_full_pipeline()
│   └── main()
│
├── bedrock_client.py       # AWS Bedrock wrapper
│   ├── ClaudeModel (Enum)
│   ├── BedrockClient
│   │   ├── invoke()
│   │   └── invoke_for_json()
│   └── analyze_with_haiku/sonnet()
│
├── news_sentiment.py       # News analysis
│   ├── SentimentResult (dataclass)
│   ├── NewsSentimentAnalyzer
│   │   ├── fetch_news()
│   │   ├── analyze_sentiment()
│   │   └── analyze_batch()
│   └── get_sentiment()
│
├── regime_interpreter.py   # Topology interpretation
│   ├── RegimeInterpretation (dataclass)
│   ├── RegimeInterpreter
│   │   ├── interpret()
│   │   └── quick_regime_check()
│   └── interpret_regime()
│
├── report_generator.py     # Report generation
│   ├── TradingReport (dataclass)
│   │   ├── to_markdown()
│   │   └── to_html()
│   ├── ReportGenerator
│   │   ├── generate_daily_report()
│   │   └── generate_alert()
│   └── generate_report()
│
├── requirements.txt        # Dependencies
├── env.example.txt         # Credentials template
│
└── docs/
    ├── Stock_Perfect_Model_Technical_Paper.md
    ├── Quick_Start_Guide.md
    └── Architecture_Diagram.md
```

---

*Architecture documentation for Stock Perfect Model v1.0*

