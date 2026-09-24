cd ~/24DIT065

cat > README.md <<'EOF'
# 24DIT065 — Research Models

Research repository containing two stages of an equity-market prediction study:

- **Model 1** — Baseline stock prediction using technical, factor, momentum, TCN, GNN, and ensemble approaches.
- **Model 2** — A larger multimodal research framework combining market technicals, related assets, financial news, and deep temporal models.

The project is designed as an experimental research pipeline with an emphasis on reproducibility, temporal validation, and out-of-sample evaluation.

---

## Model 1 — Multi-Model Stock Prediction Baseline

### Overview

Model 1 is the initial stock-prediction framework developed on a 49-stock Indian equity universe.

It evaluates multiple machine-learning and deep-learning approaches for predicting short-horizon stock returns and ranking stocks for portfolio selection.

### Main model families

- LightGBM
- Factor-based model
- Momentum model
- Temporal Convolutional Network (TCN)
- Graph Neural Network (GNN)
- Model ensemble

### Main idea

The models use historical market information to estimate future stock behavior and evaluate whether predicted rankings can be converted into portfolio-selection signals.

### Evaluation

The experiments include:

- Return prediction
- Cross-sectional stock ranking
- Top-1 / Top-5 selection
- CAGR
- Sharpe ratio
- Total return

Model 1 serves as the baseline research stage and provides a benchmark for the more comprehensive Model 2 framework.

---

## Model 2 — Multimodal Daily Stock Prediction

### Overview

Model 2 extends the research pipeline by combining several information sources into a unified daily dataset.

The current framework contains:

- Stock OHLCV market data
- Technical indicators
- Related-asset information
- NIFTY 50 information
- Global market indices
- Commodities
- FX information
- Cryptocurrency market information
- GDELT financial/news data
- News sentiment features
- News impact modelling
- Temporal deep-learning models

### Data period

The daily market pipeline covers approximately:

**2019-01-01 to 2026-08-25**

for a 49-stock Indian equity universe.

### News component

Historical news was collected using GDELT and transformed into daily stock-level features such as:

- News volume
- Mean sentiment
- Positive/negative news counts
- Rolling news intensity
- Sentiment pressure
- Predicted continuous news-impact score

### Deep-learning architecture

The main experimental architecture combines:

**TCN + LSTM + multi-task prediction heads**

The model predicts:

- 1-day return
- 3-day return
- 5-day return
- 1-day direction
- 5-day direction

### Validation methodology

The dataset uses chronological temporal splits:

- Training period
- Validation period
- Test period

The objective is to avoid random temporal leakage and evaluate performance on genuinely later observations.

### Current research finding

The current daily multimodal experiment demonstrates that the complete data and modelling pipeline can be constructed successfully, but the first TCN-LSTM experiment showed substantial overfitting and weak out-of-sample predictive performance.

Therefore, the current Model 2 results are treated as an **experimental baseline/failure analysis**, not as evidence of a successful trading predictor.

---

## Research Direction

The next stage of the project is intended to investigate **short-horizon intraday prediction** using a point-in-time framework.

The planned research direction includes:

- 5-minute market data
- 15-minute market data
- Intraday technical features
- Liquidity and market-microstructure proxies
- Quantitative price-action/market-structure features
- Timestamp-aware financial news
- Dynamic relationships between stocks
- Multimodal temporal modelling
- Strict out-of-sample evaluation
- Transaction-cost and slippage analysis

A key research objective is to determine whether these information sources provide **incremental predictive information** when aligned strictly according to what was observable at the prediction time.

---

## Repository Structure

```text
24DIT065/
│
├── model1/
│   └── Initial multi-model stock-prediction experiments
│
├── model2/
│   ├── 01_raw/
│   ├── 02_interim/
│   ├── 03_features/
│   ├── 04_model/
│   └── scripts/
│
├── README.md
└── .gitignore
