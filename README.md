# Adaptive Systematic Trading

Research codebase for two pair-trading workflows:

1. `SPY + QQQ` (equity)
2. `BTC + ETH` (crypto, via side-pair pipeline in `other-pair/btc_eth/`)

The core pipeline is:

1. `Step 2`: event construction + strategy simulation.
2. `Step 3`: walk-forward ML training/backtesting with stress/bootstrap diagnostics.

## Current Snapshot (Latest Saved Outputs)

Starting capital: `$10,000`

### SPY + QQQ

Source: `final output/reports/final_output_summary.json`  
Generated: `2026-02-25T00:28:15.110102+00:00`

- Step 2 avg monthly PnL: **$41.93**
- Step 3 avg monthly PnL: **$74.18**
- Step 3 max drawdown: **12.04%**
- Step 3 calmar: **0.58**
- Step 3 avg monthly trades: **7.13**
- Step 3 total trades: **692**

![SPY QQQ Step2](final%20output/charts/01_step2_ml_simulation_spy_qqq.png)
![SPY QQQ Step3](final%20output/charts/02_step3_real_ml_spy_qqq.png)

### BTC + ETH

Source: `other-pair/btc_eth/final output/reports/final_output_summary.json`  
Generated: `2026-02-25T17:35:32.886017+00:00`

- Step 2 avg monthly PnL: **$61.82**
- Step 3 avg monthly PnL: **$539.80**
- Step 3 max drawdown: **32.24%**
- Step 3 calmar: **0.81**
- Step 3 avg monthly trades: **16.31**
- Step 3 total trades: **1,549**
- Hybrid overlay status: **not promoted** (`selected_candidate=active_baseline`)

![BTC ETH Step2](other-pair/btc_eth/final%20output/charts/01_step2_ml_simulation_btc_eth.png)
![BTC ETH Step3](other-pair/btc_eth/final%20output/charts/02_step3_real_ml_btc_eth.png)

Note: BTC/ETH currently has high return and high drawdown. Treat as research-only until drawdown constraints are tightened.

## Recent Integrity Fixes

- Enforced hard cross-asset alignment checks in Step 3 dataset build.
- Standardized `same_bar_policy=worst` in active runs.
- Fixed BTC/ETH hybrid baseline selection to reject stale/degenerate tilt baselines.
- Fixed BTC/ETH hybrid no-promotion path to restore active baseline backtest files.
- Expanded final snapshot JSON to include drawdown/calmar/trade totals directly.

See `docs/analysis.md` section `15. Focused Remediation Pass (2026-02-25 UTC)` for details.

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run SPY/QQQ:

```bash
python run_step2_step3_final.py --same-bar-policy worst
```

Run BTC/ETH (skip data download if local data already exists):

```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --skip-download
```

## Key Files

- `step2_build_events_dataset.py`: feature/event/label construction.
- `step3_build_training_dataset.py`: step3 dataset assembly + cross-alignment validation.
- `step3_train_and_backtest.py`: walk-forward training + policy simulation.
- `step3_optimize_model.py`: candidate search and promotion logic.
- `other-pair/btc_eth/scripts/run_btc_eth_pipeline.py`: BTC/ETH end-to-end runner.
- `other-pair/btc_eth/scripts/optimize_step3_btc_eth.py`: BTC/ETH active tuning.
- `other-pair/btc_eth/scripts/optimize_step3_hybrid_btc_eth.py`: BTC/ETH hybrid overlay optimizer.
