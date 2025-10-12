# Binance Quant Project (Backtest + Live)

Event-driven Python scaffold for **backtesting** and **live trading** on Binance with a unified data pipeline.

## Features
- Event-driven engine shared by backtester and live trader.
- Strategy interface (`Strategy`) with example **SMA Crossover**.
- Data pipeline:
  - Historical bars via REST
  - Real-time via WebSocket (klines)
  - Local storage in **Parquet** (fast, columnar)
- Paper broker (slippage, fees) and live Binance broker.
- Risk checks (max position, notional limits, circuit breaker).
- Simple config via `config.yaml` and environment variables.

> This is a starter kit meant to be read and extended. Productionizing (latency, reliability, reconnection, robust error handling, etc.) is left to you.

## Quickstart

### 0) Python
Use Python 3.10+ in a virtualenv or Conda.

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Configure
Copy `.env.example` to `.env` and fill in keys if you want live trading or testnet:
```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_USE_TESTNET=true
```

Edit `config.yaml` for symbols and params.

### 3) Backtest SMA example
```bash
python -m src.scripts.backtest_sma --symbol BTCUSDT --interval 1h --lookback_days 120
```

### 4) Live trade SMA example (USE TESTNET FIRST)
```bash
python -m src.scripts.live_trade_sma --symbol BTCUSDT --interval 1m
```

### Notes
- **Timezone**: Uses `Asia/Taipei` where relevant.
- **Storage**: Historical bars cached in `./data/parquet/{symbol}_{interval}.parquet`.
- **Fees & Slippage** in backtest are configurable in `config.yaml`.
- **Testnet**: Strongly recommended while testing live trading.
- **Disclaimer**: For educational use only. No financial advice.
