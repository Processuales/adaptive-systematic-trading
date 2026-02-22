# BTC/ETH Executed Improvement Plan (2026-02-21)

## Objective
- Improve BTC/ETH Step 3 real-ML performance and robustness without touching SPY/QQQ production pipeline logic.

## Constraints Applied
- All changes scoped to `other-pair/btc_eth/*`.
- Pattern-aid kept disabled by default for BTC/ETH.
- Step 3 promotion based on robustness-aware gates and explicit tolerances.

## Implementation Steps Executed
1. Data availability fix
- IBKR downloader was unavailable locally (no TWS/Gateway connection).
- Filled `other-pair/btc_eth/data/raw/{btc,eth}_1h_all.{parquet,csv}` using Binance 1h historical API in pipeline-compatible OHLCV schema.

2. Pipeline execution and baseline
- Ran BTC/ETH pipeline end-to-end (`Step2 + Step3`), then reran Step 3 after installing missing ML dependencies (`scikit-learn`, `lightgbm`).
- Baseline tuned active-only Step 3 result:
  - avg monthly PnL: `267.56`
  - end equity: `35418.58`
  - calmar: `0.7966`
  - max drawdown: `0.2184`
  - stress x1.25 avg monthly PnL: `191.34`
  - stress x1.50 avg monthly PnL: `130.71`
  - bootstrap p10 avg monthly PnL: `94.07`

3. BTC/ETH-specific tuning layer
- Kept and used `optimize_step3_btc_eth.py` tilt search (active allocator strongly de-weights weak BTC active leg).
- Promoted active config centered on:
  - `mix_struct_weight=0.75`
  - `portfolio_allocator=dynamic_regime_forced`
  - `portfolio_min_weight=0.0`
  - `portfolio_max_weight=0.005`

4. Hybrid core-satellite overlay
- Implemented and executed `optimize_step3_hybrid_btc_eth.py`.
- First full-grid run found strong hybrid candidates but conservative promotion settings kept active baseline.
- Added baseline-anchor guard so hybrid optimization always references the tuned active-only BTC/ETH run (prevents recursive re-optimization on already-hybrid outputs).
- Added stable backup behavior for active baseline snapshots.
- Ran strict low-drawdown pass and promoted:
  - candidate: `hybrid_core0_15_btc0_85_active1_00_fixed`
  - interpretation: 15% passive core (85% BTC / 15% ETH), 85% active ML sleeve.

5. Overfitting diagnostics
- Added reusable diagnostics script:
  - `other-pair/btc_eth/scripts/build_overfit_diagnostics.py`
- Generated report:
  - `other-pair/btc_eth/step3_out/optimization/overfit_diagnostics_report.json`
- Checks included:
  - stress/bootstrapped tails
  - first-half vs second-half stability
  - last-24-month regime retention
  - concentration (`top5` / `top10` positive PnL share)
  - sign-flip significance test on monthly PnL

6. Final export
- Re-ran final export and compact report generation under `other-pair/btc_eth/final output`.
- Added BTC/ETH-specific Step 3 chart rewrite in side pipeline to remove alias labels like `QQQ weight`.

## Current Promoted Result
- Source: `other-pair/btc_eth/step3_out/backtest/step3_summary.json`
- Real-ML (after hybrid promotion):
  - avg monthly PnL: `402.11`
  - end equity: `48200.85`
  - calmar: `0.9790`
  - max drawdown: `0.2270`
  - stress x1.25 avg monthly PnL: `170.85`
  - stress x1.50 avg monthly PnL: `64.43`
  - bootstrap p10 avg monthly PnL: `165.10`
  - bootstrap p90 max drawdown: `0.3326`

## Why This Was Promoted
- Chosen explicitly for lower risk with limited profit give-up:
  - prior balanced hybrid: avg pnl `446.70`, dd `0.2842`, calmar `0.8282`
  - current risk-off hybrid: avg pnl `402.11`, dd `0.2270`, calmar `0.9790`
- Against tuned active-only baseline:
  - baseline: avg pnl `267.56`, dd `0.2184`, calmar `0.7966`
  - promoted: avg pnl `402.11`, dd `0.2270`, calmar `0.9790`
- Overfit diagnostics moved from medium risk (active baseline) to low risk (promoted hybrid), based on current checks.

## Next Iteration Plan
1. Offer two promotion profiles
- Risk-off (current): max drawdown target near `0.23` with avg monthly PnL around `400`.
- Balanced alternative: allow drawdown up to `0.29` to target avg monthly PnL around `440+`.

2. Tighten tail-cost robustness for risk-off profile
- Require (for dd<=0.25 promotion path):
  - stress x1.50 avg monthly PnL >= `60`
  - bootstrap p10 avg monthly PnL >= `140`

3. Add drawdown-aware objective variant
- Include score variant that penalizes bootstrap p90 drawdown more aggressively when dd exceeds `0.27`.
- Keep current objective as default and compare both in report.

## Repro Commands
- Full side-pair pipeline (existing data):
```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --skip-download --start-capital 10000
```

- Focused low-drawdown hybrid re-optimization and promotion:
```bash
python other-pair/btc_eth/scripts/optimize_step3_hybrid_btc_eth.py ^
  --step3-out-dir other-pair/btc_eth/step3_out ^
  --alias-dir other-pair/btc_eth/data_clean_alias ^
  --start-capital 10000 ^
  --core-fractions 0.10,0.15,0.20 ^
  --core-btc-shares 0.65,0.75,0.85 ^
  --active-scales 1.00,0.90,0.80 ^
  --core-modes fixed,vol_parity_6m ^
  --min-stress125-avg-monthly-pnl 120 ^
  --min-stress150-avg-monthly-pnl 60 ^
  --min-bootstrap-p10-monthly-pnl 120 ^
  --max-drawdown-cap 0.25 ^
  --promote-drawdown-tolerance 0.04
```

- Build overfit diagnostics:
```bash
python other-pair/btc_eth/scripts/build_overfit_diagnostics.py --step3-out-dir other-pair/btc_eth/step3_out
```
