# Step 3 Plan (Technical)

## Scope
Step 3 moves from deterministic filtering to walk-forward ML gating while preserving:
- strict cost realism (fees/slippage/overnight penalties),
- anti-overfit split discipline,
- controlled trade frequency,
- explicit risk limits.

---

## 1. ML Training Plan

### 1.1 Data Contract
- Input dataset: Step 2 events parquet with `label_end_idx`, `net_logret`, `y`, feature columns, and candidate family tags.
- Label targets:
  - classification: `y` (net-positive event),
  - regression: `net_logret` (for expected-value ranking).
- Keep costs in labels (already done) to avoid fake alpha from gross-only training.

### 1.2 Train/Validation/Test Protocol
- Use rolling walk-forward folds (time-ordered):
  - train window: trailing `L` months,
  - validation window: next 1 month,
  - optional final untouched holdout segment for acceptance.
- Purge/embargo:
  - purge events whose label window overlaps validation,
  - embargo at least max horizon bars.
- Never random shuffle.

### 1.3 Feature Sets
- Start with a compact stable set only (roughly 15-30 features):
  - signal geometry: `trend_score`, `pullback_z`, `ema_fast_slope`, `dist_to_hi`,
  - volatility/range: `sigma`, `u_atr`, `sigma_prank`, `u_atr_prank`, `range_ratio`,
  - tails/gaps: `intraday_tail_frac`, `gap_sd`, `gap_tail`,
  - execution-aware: `tp_to_cost`, `entry_overnight`.
- Cross-asset features (`spy_*` or `qqq_*`) should be toggled as an experiment, not always-on default.

### 1.4 Model Stack
- Baseline: regularized logistic regression for transparent calibration.
- Main model: gradient boosting trees (e.g., XGBoost/LightGBM equivalent behavior).
- Optional regression head for `net_logret`.

### 1.5 Calibration + Gate
- Calibrate probability per fold (Platt or isotonic style).
- Convert predictions to decision gate:
  - `EV_proxy = p * a_tp - (1-p) * b_sl - cost_rt`.
  - Trade only when `EV_proxy > margin`.
- Add separate overnight policy:
  - stricter margin or full disable for overnight entries.

### 1.6 Acceptance Criteria (must pass all)
- Validation and test both positive net after costs.
- Cost stress test survives at least `1.25x` friction.
- Max drawdown under predefined threshold.
- Trade count above minimum statistical floor (avoid tiny-sample winners).
- No strong train-test instability in edge metrics.

---

## 2. Make It More Short-Term (Higher Monthly Trade Count)

Goal example: around `+$200/month` on `$10,000` (about `+2%/month`) is aggressive and non-guaranteed.
We should target this via controlled frequency increase, not by loosening all filters blindly.

### 2.1 Frequency Levers (ordered)
1. Reduce `min_event_spacing_bars` toward 1.
2. Slightly widen accepted volatility band around current winners:
   - from `0.50-0.85` to nearby alternatives (`0.45-0.85`, `0.50-0.90`) tested by walk-forward.
3. Keep trend moderation (avoid extreme trend deciles).
4. Use lower `tp_to_cost` floor only if cost stress remains robust.

### 2.2 Bar/Horizon Tuning for Shorter Holds
- Test smaller horizons (`H`) and tighter barriers while preserving `tp_to_cost` quality.
- Add a max holding cut to recycle capital faster.
- Keep same-bar policy conservative (`worst`).

### 2.3 Trade Scheduler + Throttle
- Impose per-day/per-week caps to avoid burst overtrading.
- Add cool-down after losses to reduce regime-whipsaw churn.
- Optional session-only windows (exclude low-quality time buckets).

### 2.4 Dynamic Regime Sizing
- Size up in historically stable mid-vol regimes.
- Size down in extremes (both low-vol noise and high-vol chaos).
- Keep allocation cap per symbol and portfolio exposure cap.

### 2.5 Portfolio Construction (QQQ/SPY)
- For higher return: overweight QQQ candidate.
- For better risk-adjusted profile: blend with SPY candidate.
- Optimize weight by walk-forward objective:
  - max return subject to drawdown cap, or
  - max Calmar.

---

## 3. Step 3 Execution Checklist

1. Freeze Step 2 candidate policies and dataset version.
2. Implement fold builder with purge/embargo using `label_end_idx`.
3. Train baseline logistic + calibrated probabilities.
4. Train boosted tree model and compare.
5. Build EV gate + cost stress harness.
6. Run walk-forward report:
   - return, drawdown, calibration, trade count, turnover.
7. Select final model/policy only after untouched holdout pass.
8. Move to Step 3.5 paper harness with fill logs for cost calibration loop.
