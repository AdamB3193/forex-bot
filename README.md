# Forex Bot

Deep reinforcement learning forex trading bot trained on a support/resistance zone detection strategy.

## Project Structure

```
Forex Bot/
├── data/
│   ├── raw/          # Raw downloaded CSV files from Dukascopy (backup copies)
│   ├── processed/    # Cleaned, timezone-normalized files
│   └── db/           # SQLite database (forex_bot.db)
├── src/
│   ├── data_pipeline/ # Step 1: historical data pipeline
│   ├── zone_engine/   # Step 2: support/resistance zone detection
│   └── rl_agent/      # Step 3: RL training
├── tests/
├── logs/
└── notebooks/
```

## Steps

1. **Data Pipeline** — Download, clean, normalize, and store 24 forex pairs (2005–2026) in SQLite
2. **Zone Engine** — Detect support/resistance zones from processed OHLC data
3. **RL Agent** — Train a DRL agent to trade based on zone signals

## Data

- 24 pairs, daily OHLC, sourced from Dukascopy
- NY close (17:00 EST) candle alignment
- Train: 2005-01-01 – 2019-12-31
- Validation: 2020-01-01 – 2022-12-31
- Test: 2023-01-01 – 2026-04-30 (**never touch during development**)

## Setup

```bash
pip install -r requirements.txt
```
