# Adaptive Systematic Trading

A practical trading-research project with two steps:

- `Step 2`: build events, sweep strategy settings, and simulate dual-asset behavior.
- `Step 3`: train and walk-forward test ML models that decide when to trade safer vs more aggressive.

Main research pair is **SPY + QQQ**.  
Side pairs (like IBIT + ETHA) live under `other-pair/`.

## Quick Results

### Main Pair (SPY + QQQ)

These charts are generated from the latest run (`generated_utc=2026-02-19`), starting with `$10,000`.

Step 2 simulation chart (avg monthly PnL ≈ `$22`, avg trades/month ≈ `4.4`, end equity ≈ `$12,679`):

![Step2 SPY QQQ](final%20output/charts/01_step2_ml_simulation_spy_qqq.png)

Step 3 real ML chart (avg monthly PnL ≈ `$126`, avg trades/month ≈ `6.7`, end equity ≈ `$22,191`):

- Step 3 uses a walk-forward ML stack with two core predictors per symbol:
  - a classifier for win probability (`P(win)` after costs)
  - a regressor/quantiles for expected return distribution (used for expected value + sizing)
- We also run cost stress tests and a block bootstrap to reduce the chance we are just fitting noise.

![Step3 SPY QQQ](final%20output/charts/02_step3_real_ml_spy_qqq.png)

### Side Pair Example (IBIT + ETHA)

This is a side experiment. The overlap history is much shorter, which makes ML less stable.

Step 2 simulation chart (avg monthly PnL ≈ `$25`, end equity ≈ `$10,632`):

![Step2 IBIT ETHA](other-pair/ibit_etha/final%20output/charts/01_step2_ml_simulation_ibit_etha.png)

Step 3 real ML chart (avg monthly PnL ≈ `$19`, end equity ≈ `$10,244`):

![Step3 IBIT ETHA](other-pair/ibit_etha/final%20output/charts/02_step3_real_ml_ibit_etha.png)

## How It Works (Simple Version)

1. Download and clean bar data.
2. Create event-level features (trend, volatility, cost, etc).
3. Test many parameter sets and keep robust candidates.
4. Train ML in walk-forward style (train on past, test on future).
5. Build a dual-portfolio curve and stress test with higher costs.

Some quant terms used:
- `drawdown`: how much the strategy drops from a peak.
- `calmar`: return divided by drawdown (risk-adjusted).
- `walk-forward`: repeated train/test through time to reduce overfitting.

## How It Works (Slightly More Quant)

### Step 1: Data (IBKR)

- Data source is IBKR historical bars (Interactive Brokers Gateway/TWS).
- Main pair downloader: `data/scripts/download_history_spy_qqq.py`
- Side pair downloader example: `other-pair/ibit_etha/scripts/download_history_ibit_etha.py`
- Expected format is OHLCV bars with a UTC timestamp (`date, open, high, low, close, volume`), saved as `.csv` and/or `.parquet`.

### Step 2: Event Research + Candidate Selection

- We convert bars into an event dataset (profit target / stop loss style outcomes, plus features like trend/volatility/cost proxies).
- Then we sweep many parameter variations and keep candidates that look good on out-of-sample slices.
- Output is a dual-portfolio simulation that combines SPY and QQQ (including dynamic weighting rules in the allocator).

Key scripts:
- Build dataset: `step2_build_events_dataset.py`
- Diagnostics: `step2_5_analyze_events.py`
- Sweep + select: `step2_compare_and_select.py`
- Dual portfolio: `step2_dual_symbol_portfolio_test.py`

### Step 3: Walk-Forward ML + Risk Mode

- Step 3 is “real ML” (not a single backtest fit). It trains and tests repeatedly through time:
  - train on a rolling lookback window
  - embargo/purge around split boundaries to reduce leakage
  - test on the next time segment
- The model stack is intentionally conservative:
  - probability calibration (so `P(win)` is closer to reality)
  - expected value gating (don’t trade unless there is edge after costs)
  - sizing rules that switch between “safe” and “aggressive” modes
- Optional: a cross-asset “pattern aid” that uses SPY+QQQ context to nudge probabilities/returns, but it is capped to avoid overfitting.

Key scripts:
- Build Step 3 dataset: `step3_build_training_dataset.py`
- Optimize configs: `step3_optimize_model.py`
- Train + backtest: `step3_train_and_backtest.py`
- Pattern experiment: `step3_pattern_aid_experiment.py`

## Project Layout

- `data/`, `data_clean/`: main SPY+QQQ data.
- `step2_*.py`: Step 2 pipeline scripts.
- `step3_*.py`: Step 3 ML scripts.
- `run_step2_step3_final.py`: end-to-end final export runner.
- `final output/`: main pair charts + summary JSON.
- `other-pair/`: side experiments (IBIT+ETHA and future pairs).

## Run Commands

Install deps:

```bash
pip install -r requirements.txt
```

Run main Step 2:

```bash
python step2_run_all.py
```

Run main Step 3:

```bash
python step3_run_all.py
```

Run full main export (recommended):

```bash
python run_step2_step3_final.py --run-pattern-experiment
```

Run side-pair example (IBIT + ETHA):

```bash
python other-pair/ibit_etha/scripts/run_ibit_etha_pipeline.py --skip-download
```

## Important Notes

- Main pair is SPY+QQQ and is the primary benchmark.
- Side pairs are exploratory and may have shorter history / lower reliability.
- Use `final output/reports/final_output_summary.json` for automation and comparisons.
