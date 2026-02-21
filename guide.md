# Project Guide

## What This Repo Is
Adaptive systematic trading research with two main phases:
- `Step 2`: event generation + parameter sweeps + dual-portfolio simulation.
- `Step 3`: walk-forward ML training/backtesting with robustness checks.

Main benchmark pair is `SPY + QQQ`.
Side-pair experiments live in `other-pair/` (for example `btc_eth`, `ibit_etha`).

## Quick Start

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Run Main Pair (SPY+QQQ)
```bash
python step2_run_all.py
python step3_run_all.py
python run_step2_step3_final.py --run-pattern-experiment
```

### 3) Run BTC+ETH Side Pair
```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py
```

Skip long stages if already done:
```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --skip-download --skip-prepare
```

## Core Structure
- `step2_*.py`: Step 2 pipeline scripts.
- `step3_*.py`: Step 3 model training/backtesting/optimization.
- `run_step2_step3_final.py`: chart/report exporter from existing `step2_out` + `step3_out`.
- `final output/`: main-pair final charts and summary json.
- `other-pair/<pair>/scripts/`: side-pair orchestration scripts.

## Data + Output Conventions
- Raw and heavy generated files are intentionally excluded by `.gitignore`.
- Commit only:
  - scripts
  - docs
  - compact final outputs (charts + summary json) where useful
- For side pairs, keep everything self-contained under `other-pair/<pair>/`.

## Recommended Workflow

### A) Research Loop
1. Download/refresh data.
2. Run Step 2 sweep and inspect `step2_out/selection`.
3. Run Step 3 optimization.
4. Export final summary/charts.
5. Compare against baseline metrics before promoting.

### B) Robustness Checklist
- Check drawdown and calmar.
- Check cost stress tests (`1.25x`, `1.5x`).
- Check bootstrap lower tail (`p10` monthly pnl).
- Verify trade frequency is realistic (not too sparse).

### C) When Trying New Pair
1. Create new folder under `other-pair/<new_pair>/`.
2. Add downloader + prepare + run scripts.
3. Keep outputs isolated in that folder.
4. Do not overwrite main SPY/QQQ outputs.

## BTC+ETH Notes
- Key files:
  - `other-pair/btc_eth/IMPLEMENTATION_PLAN.md`
  - `other-pair/btc_eth/MOONSHOT_IMPLEMENTATION_PLAN.md`
  - `other-pair/btc_eth/ML_GAP_IMPLEMENTATION_PLAN.md`
  - `other-pair/btc_eth/scripts/optimize_step3_btc_eth.py`
- Current best active Step 3 is promoted via BTC+ETH-specific tuning.
- Pattern-aid is currently not recommended for BTC+ETH (until it passes robust OOS gates).

## Git Workflow

### Daily
```bash
git pull
# work
git add .
git commit -m "your message"
git push
```

### Fresh Machine
```bash
git clone <repo-url>
cd adaptive-systematic-trading
pip install -r requirements.txt
```

Then run either main pipeline or a side-pair pipeline.

## Where To Read Results
- Main pair summary:
  - `final output/reports/final_output_summary.json`
- BTC+ETH summary:
  - `other-pair/btc_eth/final output/reports/final_output_summary.json`

These JSONs are the primary machine-readable outputs for comparing runs.
