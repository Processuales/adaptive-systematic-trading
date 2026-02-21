# BTC + ETH Pipeline Implementation Plan

## Goal

Run the full SPY+QQQ-style research pipeline for BTC+ETH in an isolated side-pair workspace, with crypto-aware settings and robust evaluation.

## Scope

- Data download from IBKR
- Data cleaning + pipeline-compatible formatting
- Step 2 simulation and parameter search
- Step 3 ML training + optimization
- Iterative tuning if results are weak
- Clean final outputs with BTC/ETH naming only

## Constraints

- Do not modify or overwrite primary SPY+QQQ artifacts.
- Keep all BTC+ETH artifacts under `other-pair/btc_eth/`.
- Long-running tasks must print heartbeat/progress logs.
- Avoid obvious overfitting by using walk-forward evaluation, stress tests, and bootstrap diagnostics.

## Phase 1: Data Ingestion (IBKR)

1. Create robust downloader:
   - Path: `other-pair/btc_eth/scripts/download_history_btc_eth.py`
   - Contracts: IBKR crypto contracts for BTC and ETH.
   - Bar interval: `1 hour`.
   - `useRTH=False` (crypto is 24/7).
   - Chunked backfill with retries and pacing-safe sleeps.
   - Heartbeats: elapsed time, request counts, ETA estimate, symbol progress.
2. Output files:
   - `other-pair/btc_eth/data/raw/btc_1h_all.{csv,parquet}`
   - `other-pair/btc_eth/data/raw/eth_1h_all.{csv,parquet}`

## Phase 2: Clean + Prepare

1. Create prep script:
   - Path: `other-pair/btc_eth/scripts/prepare_btc_eth_data.py`
2. Clean bars:
   - Enforce UTC timestamps.
   - Validate OHLCV integrity.
   - Remove duplicates and malformed rows.
3. Build pipeline alias files for core scripts:
   - `qqq_1h_rth_clean.parquet` <- BTC
   - `spy_1h_rth_clean.parquet` <- ETH
4. Write mapping metadata for traceability:
   - `other-pair/btc_eth/data_clean_alias/pair_alias_map.json`

## Phase 3: Step 2 Simulation + Search

1. Create orchestrator:
   - Path: `other-pair/btc_eth/scripts/run_btc_eth_pipeline.py`
2. Run:
   - `step2_build_events_dataset.py`
   - `step2_5_analyze_events.py`
   - `step2_compare_and_select.py` with fallback constraints
   - `step2_capital_curve.py`
   - `step2_dual_symbol_portfolio_test.py` for no-ML and ML-sim
3. Crypto-specific choices:
   - No explicit overnight suppression assumptions for interpretation.
   - Keep high event throughput but constrain with out-of-sample filters.
4. Heartbeats:
   - Print stage start/end, elapsed seconds, and fallback attempts.

## Phase 4: Step 3 ML Training + Optimization

1. Run:
   - `step3_build_training_dataset.py`
   - `step3_optimize_model.py`
   - fallback to `step3_train_and_backtest.py` if optimization fails
2. Keep overfit controls:
   - Purge/embargo
   - Min train/val/test event guards
   - Cost stress multipliers
   - Bootstrap distribution checks
3. Optional pattern-aid experiment if optimizer output is stable.

## Phase 5: Iteration Loop (Quality Gate)

Repeat until one of two endpoints:

- Endpoint A (success): Improve monthly PnL and maintain acceptable robustness.
- Endpoint B (stop): Additional tuning fails to improve robust metrics.

Primary metrics:

- Average monthly PnL
- End equity (from 10k start)
- Max drawdown / calmar
- Stress `1.25x` and `1.5x` cost sensitivity
- Bootstrap P10 monthly PnL

## Phase 6: Final Output and Cleanup

1. Export clean outputs only:
   - `other-pair/btc_eth/final output/charts/01_step2_ml_simulation_btc_eth.png`
   - `other-pair/btc_eth/final output/charts/02_step3_real_ml_btc_eth.png`
   - `other-pair/btc_eth/final output/reports/final_output_summary.json`
   - `other-pair/btc_eth/final output/README.txt`
2. Remove stale temp folders:
   - `data_clean/`, `data_clean_alias/`, `step2_out/`, `step3_out/`
3. Validate final JSON contains BTC/ETH naming and no SPY/QQQ filename artifacts.

## Run Commands (Planned)

1. Download:
   - `python other-pair/btc_eth/scripts/download_history_btc_eth.py`
2. Full pipeline:
   - `python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --start-capital 10000`

## Risks

- IBKR crypto history depth may be shorter than equities.
- Regime instability can produce fragile backtests.
- Alias-based compatibility may still create internal qqq/spy keys in intermediate artifacts; final published outputs will be mapped/cleaned to BTC/ETH.
