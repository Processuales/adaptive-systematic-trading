# BTC/ETH Regime Classifier + Model Audit Implementation Plan (2026-02-22)

## Scope
- Pair in scope: BTC/ETH side pipeline only (`other-pair/btc_eth`).
- SPY/QQQ default production behavior must remain unchanged.
- Goal: add a third ML model (regime classifier), improve risk-adjusted performance, and audit model/formula correctness for crypto.

## 1. Current Step 3 Model Architecture (as implemented)

### Model 1: Direction Probability Model
- Stack: ridge logistic-style head + LightGBM binary classifier.
- Ensemble probability:
  - `p_raw = w_p * p_tree + (1 - w_p) * p_ridge`
- Calibration:
  - isotonic (fallback platt, fallback identity).

### Model 2: Return Magnitude Model
- Stack: ridge regression + LightGBM regression + LightGBM quantiles (q10/q90).
- Ensemble return:
  - `r_raw = w_r * r_tree + (1 - w_r) * r_ridge`
- Uncertainty:
  - `uncertainty = clip(q90 - q10)`.

### Existing decision formula
- Structural EV:
  - `ev_struct = p * a_tp - (1 - p) * b_sl - cost_rt`
- Final EV:
  - `ev_final = mix_struct_weight * ev_struct + (1 - mix_struct_weight) * r_raw`
- Trade gating:
  - safe/aggressive thresholds on `p` and `ev_final`.
- Position sizing:
  - confidence-aware scaling over safe/aggressive base sizes.

## 2. Crypto Correctness Audit (existing two-model setup)

### Correct / acceptable
- Crypto friction profile is now wired (spread/slippage/commission) in Step 2/3 builders.
- 24/7 mode disables overnight friction and adds `is_weekend`.
- Walk-forward chronology and embargo are structurally sound.

### Major issue found (P0)
- **Feature leakage in Step 3 feature set**:
  - `hold_bars`, `touch_delay_bars`, `same_bar_ambiguous`, `truncated_horizon` are post-outcome fields.
  - These can leak label resolution info and inflate backtest quality.
- Fix strategy:
  - Add crypto-safe feature exclusion path (`--exclude-leaky-features`) and enable it for BTC/ETH runs.
  - Keep default behavior unchanged for non-crypto/SPYQQQ runs.

### Medium issues for crypto fit
- `entry_overnight` is always 0 in 24/7 mode; low utility for crypto after override.
- No explicit intraday hour/session seasonality feature in Step 3 (only weekend flag).
- Symbol naming in generic Step 3 still references QQQ/SPY labels internally (cosmetic + maintainability issue; not core math).

## 3. Third Model Design: Regime Classifier

### Objective
Predict whether the **near-future trading regime** is favorable before taking risk.

### Label design
- Build forward regime score from next `L` events:
  - forward mean net return (`fwd_ret`)
  - forward win rate (`fwd_win`)
  - `regime_score = 10000 * fwd_ret + 20 * (fwd_win - 0.5)`
- Binary target:
  - `regime_y = 1{regime_score >= quantile(label_q)}`

### Regime model
- LightGBM binary classifier over regime-context features.
- Probability calibration (same calibrator stack pattern as main classifier).

### Integration into policy
- Regime EV adjustment:
  - `ev_final_adj = ev_final + regime_ev_scale * regime_center * cost_rt`
  - where `regime_center = 2 * (p_regime - 0.5)` clipped to [-1, 1]
- Regime gate:
  - safe trades require `p_regime >= regime_p_cut`
  - aggressive trades require `p_regime >= regime_agg_p_cut`
- Regime size scaling:
  - `size *= clip(1 + regime_size_scale * regime_center, regime_size_min_mult, regime_size_max_mult)`

## 4. Implementation Steps

### Step A: Safety + Feature Hygiene
1. Add leaky-feature exclusion option in `step3_train_and_backtest.py`.
2. Pass this option only from BTC/ETH flow.

### Step B: Regime Model in Step 3
1. Add regime-feature selector and regime-label construction.
2. Train calibrated LightGBM regime classifier per fold.
3. Inject regime gate + EV/size modulation into `simulate_policy`.
4. Persist regime artifacts per fold in model metadata.

### Step C: BTC/ETH Optimizer Updates
1. Extend BTC/ETH Step 3 tuner to evaluate regime-on and regime-off candidates.
2. Always run BTC/ETH Step 3 candidates with leaky-feature exclusion.
3. Promote only if score/robustness improves.

### Step D: Hybrid Overlay Re-optimization
1. Re-run hybrid core/active search after new active sleeve is produced.
2. Re-balance for drawdown cap with stress and bootstrap constraints.

## 5. PnL + Drawdown Improvement Levers (post-regime)

### Priority levers
1. Regime gating strictness (`regime_p_cut`, `regime_agg_p_cut`) for drawdown control.
2. Regime size asymmetry (`regime_size_scale`, min/max multipliers).
3. Mix weighting (`mix_struct_weight`) vs pure return forecast exposure.
4. Active sleeve fraction in hybrid overlay (already highly effective).

### Secondary algorithmic upgrades
1. Add hour-of-day cyclic features (`sin/cos(hour_utc)`) for crypto micro-regime effects.
2. Add rolling downside semivariance / tail pressure feature for risk-aware gating.
3. Add correlation-aware active sleeve throttling in hybrid overlay.
4. Add objective term penalizing downside concentration (monthly tail CVaR proxy).

## 6. Formula and Parameter Final Checks

### Checked formulas
- Structural EV and blended EV: coherent and dimensionally consistent.
- Confidence sizing and uncertainty calibration: coherent.
- Portfolio dynamic allocator + tilt: coherent.
- Cost stress and bootstrap diagnostics: coherent.

### High-priority parameter checks
- Ensure `regime_agg_p_cut >= regime_p_cut`.
- Keep `regime_label_quantile` in (0,1) and avoid target imbalance.
- Keep `max_aggressive_size` and regime size multipliers within bounded caps.
- Keep friction profile + 24/7 mode propagated to all BTC/ETH stages.

### Known major risks after this audit
1. Leakage risk if leaky-feature exclusion is not enabled for BTC/ETH Step 3.
2. Candidate overfitting risk if regime thresholds are over-tuned without stress/bootstrap constraints.

## 7. Execution Protocol
1. Implement code changes for Steps A-C.
2. Run BTC/ETH Step 3 training/tuning with regime candidates.
3. Run hybrid overlay optimization with drawdown-focused constraints.
4. Re-run overfit diagnostics.
5. Compare vs current BTC/ETH promoted baseline on:
   - avg monthly PnL
   - max drawdown
   - calmar
   - stress x1.25/x1.50 monthly PnL
   - bootstrap p10 monthly PnL
6. Keep promoted result only if risk/return improvement is real and robust.
