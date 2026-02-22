# Step 4 Allocator

## Purpose
Step 4 is the formal portfolio layer above Step 3 sleeves:
- Sleeve A: SPY+QQQ Step 3 monthly returns
- Sleeve B: BTC+ETH Step 3 monthly returns

It optimizes allocation and leverage with walk-forward testing and robustness checks.

## Scripts
- `step4_optimize_allocator.py`
  - runs candidate search for one objective (`return`, `balanced`, `calmar`, `robust`)
  - includes baseline, heuristic, regime, and ML candidates
  - writes backtest outputs and optimization report

- `step4_run_all.py`
  - runs Step 4 optimizer for multiple objectives
  - promotes both:
    - absolute best run (subject to promotion DD cap)
    - best truly dynamic run (non-static, with minimum activity/diversification)
  - writes artifacts under `step4_out/final/` and `step4_out/final_dynamic/`

## Core Models
Step 4 candidate families:
- `static`: fixed sleeve weights (including levered BTC-only baseline)
- `heuristic`: rolling sharpe-minus-drawdown score allocator
- `regime_guard`: threshold-based BTC risk-off guardrail that rotates toward SPY+QQQ
- `regime_rf`: random-forest binary regime classifier (BTC-favored vs SPY-favored month)
- `ml_lr`: logistic-regression utility-bin allocator
- `ml_rf`: random-forest utility-bin allocator
- `ml_dual_ridge`: dual expected-return regressors with mean-variance portfolio mapping

## Typical Run
```bash
python step4_run_all.py ^
  --spyqqq-monthly combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet ^
  --btceth-monthly other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet ^
  --out-dir step4_out ^
  --start-capital 10000 ^
  --train-ratio 0.6 ^
  --min-history-months 24 ^
  --bootstrap-samples 1200 ^
  --bootstrap-block-months 6 ^
  --promote-dd-cap 0.30
```

## Outputs
- `step4_out/final/backtest/step4_portfolio_curve.png`
- `step4_out/final/backtest/step4_monthly_table.parquet`
- `step4_out/final/backtest/step4_summary.json`
- `step4_out/final/optimization/step4_optimization_report.json`
- `step4_out/final/step4_run_report.json`
- `step4_out/final_dynamic/backtest/step4_portfolio_curve.png` (if a dynamic candidate passes)
- `step4_out/final_dynamic/backtest/step4_monthly_table.parquet` (if available)
- `step4_out/final_dynamic/backtest/step4_summary.json` (if available)

## Notes
- Step 4 compares sleeves on aligned overlapping months only.
- Reported benchmark metrics in Step 4 summary are for the same Step 4 backtest window.
- Robustness gates include drawdown caps, stress positivity, and bootstrap tail checks.
