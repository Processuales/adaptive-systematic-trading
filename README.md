# Adaptive Systematic Trading

A practical trading-research project with two steps:

- `Step 2`: build events, sweep strategy settings, and simulate dual-asset behavior.
- `Step 3`: train and walk-forward test ML models that decide when to trade safer vs more aggressive.

Main research pair is **SPY + QQQ**.  
Side pairs (like IBIT + ETHA) live under `other-pair/`.

## Quick Results

### Main Pair (SPY + QQQ)

Step 2 simulation chart:

![Step2 SPY QQQ](final%20output/charts/01_step2_ml_simulation_spy_qqq.png)

Step 3 real ML chart:

![Step3 SPY QQQ](final%20output/charts/02_step3_real_ml_spy_qqq.png)

### Side Pair Example (IBIT + ETHA)

Step 2 simulation chart:

![Step2 IBIT ETHA](other-pair/ibit_etha/final%20output/charts/01_step2_ml_simulation_ibit_etha.png)

Step 3 real ML chart:

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
