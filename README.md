# Forex Bot

Deep reinforcement learning forex trading bot implementing Nick Shawn's S&R zone-based trading strategy.

## Project Status

**Steps 1–6 complete. 51/51 tests passing.**

| Step | Component | Status |
|------|-----------|--------|
| 1 | Data Pipeline | ✅ Complete |
| 2 | Zone Engine (sr_engine.py) | ✅ Complete |
| 3 | Gymnasium Trading Environment | ✅ Complete |
| 4 | Risk Guard Wrapper (5 rules) | ✅ Complete |
| 5 | PPO Trainer + LSTM Feature Extractor | ✅ Complete |
| 6 | Hyperparameter Optimization (Optuna) | ✅ Complete |
| 7 | MT5 Live Execution | Pending |
| 8 | Paper Trading | Pending |

## Project Structure
```
Forex Bot/
├── data/
│   ├── raw/          # Raw Dukascopy CSVs (gitignored)
│   ├── processed/    # Cleaned files (gitignored)
│   └── db/           # SQLite DB: forex_bot.db + optuna_study.db (gitignored)
├── models/
│   └── best_params.json   # Best hyperparams from Step 6 Optuna sweep
├── src/
│   ├── data_pipeline/     # Step 1: download, clean, validate, store
│   ├── zone_engine/       # Step 2: LuxAlgo S&R port + Nick extensions
│   └── rl_agent/          # Steps 3–6: environment, risk guard, trainer, hyperopt
├── tests/                 # 51 pytest tests (all passing)
├── logs/                  # Validation reports (gitignored)
└── Past Chats/            # Full session logs
```

## Data

- 24 forex pairs, daily OHLC (NY close 17:00 EST), sourced from Dukascopy
- 126,736 bars total
- Train: 2005-01-01 – 2019-12-31
- Validation: 2020-01-01 – 2022-12-31
- **Test: 2023-01-01 – present (SEALED — never used during development)**

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest tests\ -v
```

## Run Hyperparameter Optimization

```bash
python src\rl_agent\hyperopt.py --trials 75
python src\rl_agent\hyperopt.py --show-best
```
