# Plan (Step 2 -> Step 3)

## Current Status
- Step 2 event generation is stable and cost-aware.
- Step 2.5 filter diagnostics are implemented.
- Step 2b sweep/backtest and candidate selection are implemented.
- Best candidates are now stored in `step2_out/selection/`.

## Immediate Next Actions
1. Freeze one non-ML candidate and one ML-sim candidate from `step2_out/selection/`.
2. Run the dual-symbol portfolio test script for a single optimized SPY/QQQ curve.
3. Pick deployment preference:
   - return-first (higher CAGR),
   - or risk-first (higher Calmar / lower drawdown).
4. Move into Step 3 model training workflow from `docs/step3_plan.md`.

## Risk Controls to Keep During Transition
- Keep modeled costs/slippage enabled in all tests.
- Keep no-overlap event selection in backtests.
- Keep chronological splits (no random shuffle).
- Reject strategies that only work with tiny test trade counts.

## Targeting Higher Monthly PnL Without Overfitting
- Increase trades gradually by tuning spacing and regime bands, not by dropping all filters.
- Validate every increase under cost stress (`1.25x` and `1.5x` costs).
- Use an explicit drawdown guardrail before accepting frequency increases.

## Deliverables for Step 3 Start
- Chosen candidate config JSON(s).
- Walk-forward split spec (purge + embargo).
- Feature list freeze (versioned).
- Baseline model + calibrated gate acceptance metrics.
