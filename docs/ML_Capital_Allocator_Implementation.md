# ML Capital Allocation Model (SPY vs QQQ) - Implementation Plan

## 1. Goal
Build a second ML layer that decides daily portfolio allocation weights:
- `w_qqq` in `[0.0, 1.0]`
- `w_spy = 1.0 - w_qqq`

Primary objective:
- maximize risk-adjusted net growth
- keep drawdown and allocation turnover controlled

This model sits above existing Step 3 per-symbol trade engines.

## 2. Inputs and Data Pipeline
Data source:
- Step 3 outputs (`step3_out/backtest/*.parquet`, fold summaries, scored events)
- market features already available in event datasets

Training table granularity:
- 1 row per day (or per rebalance timestamp)

Core feature blocks:
- recent realized returns of Step 3 QQQ/SPY strategy legs (`1d/5d/21d/63d`)
- realized vol and downside vol (`5d/21d/63d`)
- rolling max drawdown and recovery speed
- rolling hit-rate / payoff ratio of each leg
- model confidence stats from Step 3 (probability calibration output and uncertainty quantiles)
- drift flags / drift intensity from fold diagnostics
- macro regime proxies from prices (trend, slope, correlation spread between QQQ and SPY)

Data hygiene:
- strict timestamp alignment, UTC
- no forward leakage (features only from information available before rebalance)
- purge+embargo around fold boundaries

## 3. Labeling Strategy
Use a direct utility label, not raw return:
- For each day `t`, evaluate candidate allocations over horizon `H` (e.g., 5 to 20 trading days):
  - `w in {0.00, 0.05, ..., 1.00}`
  - compute realized utility:
    - `U = future_return - lambda_dd * future_drawdown - lambda_tc * turnover_cost`
- choose best `w*` as target for supervised learning

Two target options:
1. Regression target: continuous `w*`
2. Classification target: bucketed weights (e.g., 11 bins)

Recommendation:
- start with bucketed classification for stability and easier calibration.

## 4. Model Architecture
Baseline model:
- LightGBM multiclass classifier (weight buckets)

Upgrade model:
- ensemble:
  - LightGBM multiclass
  - ElasticNet / Ridge classifier on normalized features
  - optional CatBoost multiclass (if dependency acceptable)
- blend probabilities with validation-weighted averaging
- temperature scaling or isotonic calibration on final probabilities

Output policy:
- expected utility per bucket from calibrated probabilities
- choose bucket with highest expected utility
- apply smoothing and turnover cap:
  - `w_t = alpha * w_{t-1} + (1-alpha) * w_raw`
  - max daily change cap (e.g., `<= 0.10`)

## 5. Training and Validation Protocol
Use nested walk-forward:
- outer folds: chronological train/test
- inner folds: tune hyperparameters and risk penalties

Each fold:
- train on past only
- test on next out-of-sample block
- log fold-level:
  - PnL, CAGR, max DD, Calmar
  - allocation turnover
  - monthly hit-rate and tail-loss stats

Selection criterion:
- robust objective, not single-run peak:
  - maximize median fold Calmar + avg monthly PnL
  - penalize high DD, high turnover, unstable fold tails

## 6. Backtest Integration
Integration order per day:
1. Step 3 symbol models produce candidate returns/confidence
2. Allocator model predicts `w_qqq`
3. Apply risk envelope:
  - DD guard rails
  - volatility scaling
  - drift safety throttle
4. Execute combined portfolio return

Costs:
- include fee + slippage + turnover impact in allocator training utility and backtest.

## 7. Overfitting Controls
- hard split discipline (no leakage)
- minimum feature importance stability checks across folds
- monotonic sanity tests on key risk features
- cap model complexity (`num_leaves`, depth, regularization)
- reject configs that fail robustness gates:
  - drawdown cap
  - high negative-fold rate
  - weak p25 fold Calmar

## 8. Deliverables
New scripts:
1. `step3b_build_allocator_dataset.py`
2. `step3b_train_allocator.py`
3. `step3b_optimize_allocator.py`
4. `step3b_backtest_with_allocator.py`

Artifacts:
- `step3_out/allocator/dataset/*.parquet`
- `step3_out/allocator/models/*.txt|*.json`
- `step3_out/allocator/backtest/*`
- `final output/step3_allocator/*`

## 9. Acceptance Criteria
Allocator is accepted only if out-of-sample results beat current Step 3 baseline on both:
- `avg_monthly_pnl`
- risk-adjusted metric (Calmar)

And stays within risk envelope:
- max drawdown not worse than baseline by more than configured tolerance
- turnover below cap
- robustness gates pass on fold tails

## 10. Rollout Plan
1. Build dataset and confirm no leakage.
2. Train baseline multiclass allocator.
3. Add calibration + smoothing + turnover controls.
4. Run nested walk-forward optimization.
5. Compare against current Step 3 in one report.
6. Promote only robust pass config to final output.
