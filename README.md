# Adaptive Systematic Trading

This is my research project for systematic trading with two main pairs:

- **SPY + QQQ** (equity pair)
- **BTC + ETH** (crypto pair)

I kept the structure simple:

1. `Step 2` builds event data and tests strategy settings.
2. `Step 3` trains ML models with walk-forward testing.
3. `Step 4` combines the two sleeves into one allocator.

## Quick Results (From Latest Saved Outputs)

Starting capital in these runs is `$10,000`.

### SPY + QQQ

- Avg monthly PnL: **$125.68**
- End equity: **$22,190.93**
- Stress (1.25x costs): **$106.87** avg monthly PnL

![SPY QQQ Step2](final%20output/charts/01_step2_ml_simulation_spy_qqq.png)
![SPY QQQ Step3](final%20output/charts/02_step3_real_ml_spy_qqq.png)

Source: `final output/reports/final_output_summary.json`

### BTC + ETH

- Avg monthly PnL: **$163.67**
- End equity: **$24,730.00**
- Max drawdown: **19.65%**
- Calmar: **0.66**
- Stress (1.25x costs): **$161.65** avg monthly PnL
- Bootstrap p10 avg monthly PnL: **$56.81**

![BTC ETH Step2](other-pair/btc_eth/final%20output/charts/01_step2_ml_simulation_btc_eth.png)
![BTC ETH Step3](other-pair/btc_eth/final%20output/charts/02_step3_real_ml_btc_eth.png)

Source: `other-pair/btc_eth/final output/reports/final_output_summary.json`

## SPYQQQ + BTCETH Combined (Short)

Combined artifacts live in `final output/combined/`.

Two useful Step 4 views:

- Return-first allocator output
- Dynamic/risk-balanced allocator output

![Combined Allocator](final%20output/charts/03_step4_portfolio_allocator_spyqqq_btceth.png)
![Combined Allocator Dynamic](final%20output/charts/03_step4_portfolio_allocator_spyqqq_btceth_dynamic.png)

Also available:

- `final output/combined/charts/SPYQQQ_BTCETH_combined_chart.png`
- `final output/combined/charts/SPYQQQ_BTCETH_combined_chart_calmar.png`

## Overfit Check (BTC + ETH)

From the latest diagnostics:

- promoted hybrid model risk label: **low**
- active baseline risk label: **high**
- promoted model passed all current checks (time split, concentration, significance, bootstrap tail)

Source: `other-pair/btc_eth/step3_out/optimization/overfit_diagnostics_report.json`

## How The Code Works

### Step 2 (event building + simulation)

- Builds event-level rows from bar data.
- Computes features, labels outcomes, and sweeps many knob combinations.
- Produces a tradable simulation summary for each pair.

Main files:

- `step2_build_events_dataset.py`
- `step2_compare_and_select.py`
- `step2_dual_symbol_portfolio_test.py`

### Step 3 (walk-forward ML)

- Uses rolling train/test splits instead of one static split.
- Trains probability + return models and applies cost-aware gating.
- Runs stress tests and bootstrap diagnostics.

Main files:

- `step3_build_training_dataset.py`
- `step3_optimize_model.py`
- `step3_train_and_backtest.py`

### Step 4 (cross-pair allocator)

- Takes monthly sleeve returns from SPY/QQQ and BTC/ETH.
- Tests static, heuristic, and ML allocation policies.
- Produces two promoted outputs:
  - best absolute return profile
  - best truly dynamic profile

Main files:

- `step4_optimize_allocator.py`
- `step4_run_all.py`

## Interesting things I added

These are the parts I think are interesting:

- **Drift guard** logic that soft-kills exposure when behavior changes.
- **Drought relief** logic that relaxes thresholds a bit when trade count dries up.
- **Pattern aid** (cluster-based context) to nudge probabilities and return estimates.
- **Dual promotion in Step 4** (absolute winner + dynamic winner) so we do not force one objective.
- **Cost stress + bootstrap tail checks** in selection, not just raw return ranking.

## Run Commands

Install deps:

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

Run Step 4 allocator:

```bash
python step4_run_all.py \
  --spyqqq-monthly combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet \
  --btceth-monthly other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet \
  --out-dir step4_out
```

## Project Layout

- `data/`, `data_clean/`: SPY+QQQ data and cleaning outputs.
- `other-pair/btc_eth/`: BTC+ETH pipeline and reports.
- `step2_*.py`: event building and simulation.
- `step3_*.py`: ML training and walk-forward backtests.
- `step4_*.py`: combined allocator.
- `final output/`: final charts and summaries.
