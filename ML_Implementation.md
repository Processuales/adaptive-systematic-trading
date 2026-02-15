# ML Implementation Plan (Step 3)

## 1) Goal
Build a real ML trading layer (not mock gating) that improves **risk-adjusted return** and keeps trade frequency usable for paper/live validation.

## 2) Current Baseline (from current project artifacts)
Date reference: **February 15, 2026**.

- Step 2 event universe (`step2_out/step2_5/step2_5_summary.json`):
  - Events: **10,019**
  - Mean net edge (all events): **-2.26 bps**
  - Mean modeled cost: **9.14 bps**
  - Weekly event rate: **19.1**
- Best filters are still selective (edge exists only after filtering).
- Current dual portfolio after this upgrade (`step2_out/selection/dual_portfolio_ml/dual_symbol_portfolio_summary.json`):
  - End equity: **$14,602** (from $10,000)
  - CAGR: **3.87%**
  - Max drawdown: **8.60%**
  - Avg monthly PnL: **$38.03**
  - Avg monthly trades: **6.98**

This is better than the earlier low-frequency setup (roughly ~1.4 trades/month), but still below your `$200/month on $10k` target.

## 3) Step 3 Target Specification
Primary target:
- Increase expected monthly PnL while keeping drawdown controlled.

Quantitative acceptance gates (first production candidate):
- Out-of-sample Calmar >= current mock ML dual baseline.
- Max drawdown <= 10-12%.
- Avg monthly trades >= 5.
- No catastrophic fold (no walk-forward month with model instability or extreme turnover).

## 4) Data and Labels
Use existing Step 2 outputs as the base training table:
- `step2_out/events/*.parquet`
- `step2_out/bar_features/*.parquet`

Labels to train simultaneously:
- Classification: `y = I(net_logret > 0)`
- Regression: `net_logret`
- Optional quantiles: `q10/q50/q90` of `net_logret` (quantile models)

Keep event metadata:
- `t_idx`, `label_end_idx`, decision/entry/exit timestamps
- `entry_overnight`, costs, TP/SL geometry

## 5) Model Stack
## 5.1 Baselines
- `scikit-learn` Logistic Regression (L2)
- `scikit-learn` Ridge / ElasticNet regression for `net_logret`

## 5.2 Main Models
- `LightGBM` classifier for `P(y=1)`
- `LightGBM` quantile regressors for `net_logret` tail control

## 5.3 Calibration
- Isotonic regression or Platt scaling on validation folds.
- Store calibration model per walk-forward fold.

## 5.4 Final Decision Score
For each event:
- `p = calibrated P(y=1)`
- `ev_struct = p*a_tp - (1-p)*b_sl - cost_rt`
- `ev_ml = predicted net_logret`
- `ev_final = w1*ev_struct + w2*ev_ml` (weights tuned on train only)

Trade gate:
- `ev_final > k * cost_rt`
- `tp_to_cost >= phi`
- Optional stricter overnight gate.

## 6) Validation Protocol (critical)
Use walk-forward time splits with purge/embargo:
- Train window: trailing 24-36 months
- Validation/Test: next 1 month
- Purge overlapping labels by `label_end_idx`
- Embargo: `H` bars

Track per fold:
- End equity, CAGR, Max DD, Calmar
- Trade count / month
- Net bps mean, turnover
- Calibration error (Brier + reliability bins)

## 7) Hyperparameter Search
Use `Optuna` over:
- Model hyperparams (depth/leaves/min_data/learning_rate)
- Gate params (`k`, `phi`, overnight multiplier)
- Portfolio params (allocation/risk caps)

Objective:
- Maximize median fold Calmar
- Penalize drawdown > cap and low trade count

## 8) Execution and Risk Layer
Keep deterministic risk controls in Step 3:
- Drawdown kill switch
- Daily/session loss stop
- Min holding / cooldown
- Turnover governor

Position sizing:
- Probability/EV-based capped sizing
- Volatility targeting cap
- Overnight size haircut

## 9) Implementation Phases
Phase 0:
- Freeze dataset schema + feature list.
- Add dataset versioning and fold manifests.

Phase 1:
- Build `step3_train_dataset.py` to assemble ML table with purge-safe folds.

Phase 2:
- Build `step3_train_models.py` (logistic/ridge baselines + calibration).

Phase 3:
- Add LightGBM models + Optuna (`step3_optuna_search.py`).

Phase 4:
- Build `step3_walkforward_backtest.py` using calibrated outputs and EV gates.

Phase 5:
- Integrate with dual portfolio allocator and risk engine.

Phase 6:
- Paper-trading shadow deployment and weekly recalibration loop.

## 10) Required Libraries
Add to requirements for Step 3:
- `scikit-learn`
- `lightgbm`
- `optuna`
- `scipy`
- (optional) `shap` for diagnostics

## 11) Expected Improvement vs Current Mock ML
Current mock ML is linear + threshold tuning. Step 3 should improve by:
- Better nonlinear regime separation (trees)
- Better calibration (probabilities usable for sizing)
- Better tail-risk control (quantile predictions)

Practical expectation (realistic):
- 20-60% relative uplift in Calmar over mock ML in stable folds
- Trade count maintained or slightly increased without large DD jump
- More stable month-to-month behavior than fixed-threshold mock gating

## 12) Deliverables
- `step3_train_dataset.py`
- `step3_train_models.py`
- `step3_walkforward_backtest.py`
- `step3_optuna_search.py`
- `step3_model_registry.json`
- `docs/step3_report.md`

