# Adaptive Systematic Trading

Practical systematic trading research with three production layers:

1. `Step 2`: event dataset construction + parameter sweep + dual-symbol simulation.
2. `Step 3`: walk-forward ML training/backtesting for each pair.
3. `Step 4`: portfolio allocator across SPY/QQQ and BTC/ETH sleeves.

## Main Pairs

This repo now treats both pairs as core:

- **SPY + QQQ** (equities)
- **BTC + ETH** (crypto)

### SPY + QQQ Snapshot

From `final output/reports/final_output_summary.json`:

- Avg monthly PnL: `$125.68`
- End equity (from `$10,000`): `$22,190.93`
- Stress test (1.25x costs): `$106.87` avg monthly PnL

Charts:

- `final output/charts/01_step2_ml_simulation_spy_qqq.png`
- `final output/charts/02_step3_real_ml_spy_qqq.png`

### BTC + ETH Snapshot

From `other-pair/btc_eth/final output/reports/final_output_summary.json`:

- Avg monthly PnL (Step 3 selected): `$120.99`
- End equity (from `$10,000`): `$21,009.91`
- Bootstrap p10 avg monthly PnL: `$25.75`

Charts:

- `other-pair/btc_eth/final output/charts/01_step2_ml_simulation_btc_eth.png`
- `other-pair/btc_eth/final output/charts/02_step3_real_ml_btc_eth.png`

## SPYQQQ + BTCETH Combined (Short)

Combined outputs are organized under:

- `final output/combined/charts/`
- `final output/combined/reports/`
- `final output/combined/data/`

Step 4 promoted outputs:

- Absolute best (return-first): `final output/charts/03_step4_portfolio_allocator_spyqqq_btceth.png`
- Dynamic best (risk-balanced): `final output/charts/03_step4_portfolio_allocator_spyqqq_btceth_dynamic.png`

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run SPY+QQQ pipeline:

```bash
python run_step2_step3_final.py --run-pattern-experiment
```

Run BTC+ETH pipeline:

```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --skip-download
```

Run Step 4 combined allocator:

```bash
python step4_run_all.py \
  --spyqqq-monthly combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet \
  --btceth-monthly other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet \
  --out-dir step4_out
```

## Project Layout

- `data/`, `data_clean/`: SPY+QQQ data and cleaning outputs.
- `other-pair/btc_eth/`: BTC+ETH pipeline, reports, and scripts.
- `step2_*.py`: Step 2 event research and simulation.
- `step3_*.py`: Step 3 ML optimization and backtesting.
- `step4_optimize_allocator.py`, `step4_run_all.py`: cross-pair allocator.
- `final output/`: consolidated charts/reports.

## Notes

- Use `final output/reports/final_output_summary.json` for SPY+QQQ automation.
- Use `other-pair/btc_eth/final output/reports/final_output_summary.json` for BTC+ETH automation.
- Use `step4_out/final/backtest/step4_summary.json` and `step4_out/final_dynamic/backtest/step4_summary.json` for combined allocator results.
