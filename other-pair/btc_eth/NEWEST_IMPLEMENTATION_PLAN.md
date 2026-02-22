# BTC/ETH Comprehensive Analysis & Implementation Plan

## Executive Summary

**Promoted result**: avg monthly PnL **$402**, end equity **$48,201**, calmar **0.98**, max drawdown **22.7%**, starting from $10k.
This is the hybrid core-satellite overlay (15% passive BTC core + 85% active ML sleeve), promoted over the tuned active-only baseline (avg monthly PnL $268).

After deep analysis of the full codebase (~8,000 lines across 12 scripts), final reports, and overfit diagnostics, here are findings organized by the user's 5 questions.

---

## 1. Are There Major Bugs That Inflated Results?

### 🔴 Critical: Friction Model Uses Equity-Market Defaults for Crypto

The `default_knobs()` in [step2_build_events_dataset.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/step2_build_events_dataset.py#L112-L120) defines spread costs only for SPY, QQQ, and SMH:

```python
k.spread_half_bps = {
    "SPY": 0.8,
    "QQQ": 1.2,
    "SMH": 1.8,
}
```

When BTC/ETH data is aliased as QQQ/SPY, BTC gets QQQ's spread of **1.2 bps** and ETH gets SPY's spread of **0.8 bps**. In reality:
- **BTC** spot/perp spreads: typically **2-8 bps** on major exchanges (Binance/Coinbase), much wider during vol spikes
- **ETH** spot/perp spreads: typically **3-12 bps**, worse liquidity than BTC

> [!CAUTION]
> The friction model is **undercharging crypto spreads by ~3-6x**. This alone could inflate PnL significantly. The total round-trip cost (`cost_rt`) is approximately `2*(spread_half + slip_half) + commission`. With equity-tier spreads, the model is far too optimistic for crypto execution.

**Base slippage** (`slip_base_bps = 0.8`) and **commission** (`commission_round_trip_bps = 1.0`) are also equity-calibrated. Crypto exchanges (even maker-taker) charge **5-10 bps round-trip** for takers.

### 🟡 Medium: `entry_overnight` Semantics Are Wrong for 24/7 Markets

The `compute_gap_stats_event_window()` function detects "overnight" entries by checking if the next bar crosses a New York session boundary:

```python
sdate = session_date_ny(ts_utc)
sdate_next = sdate.shift(-1)
out["entry_overnight"] = (sdate_next.notna()) & (sdate_next != sdate)
```

Crypto trades 24/7 with no session breaks. This flag still fires whenever a bar crosses the NY midnight boundary, which:
- Triggers wider barriers (`tp_mult_overn=2.2` vs `tp_mult_intra=2.0`, `sl_mult_overn=1.6` vs `sl_mult_intra=1.5`)
- Adds overnight slippage (`overnight_slip_add_bps=1.5`)
- Uses shorter horizon (`horizon_overn=10` vs `horizon_intra=12`)

This isn't necessarily a *bug* (it still adds conservatism), but the logic is semantically wrong for crypto and the barriers/horizons should be rethought.

### 🟢 No Future-Leak Found in Labeling

The triple-barrier labeling in `label_event_long()` is correctly implemented:
- Decision at bar `t` close, entry at bar `t+1` open
- Exit at next-bar open after barrier touch (not at touch price itself)
- `same_bar_policy="worst"` conservatively resolves ambiguous bars as stop-loss
- Gap statistics are shifted by 1 event to prevent leakage
- Cross-asset features use `merge_asof(direction="backward")` — no lookahead

### 🟢 Walk-Forward Is Properly Structured

`train_symbol_walkforward()` uses expanding-window monthly folds with embargo separation. Train/val/test splits are chronologically ordered. No data from the test period leaks into training.

---

## 2. Are We Properly Simulating Fees, Slippage, and Combatting Overfitting?

### Fee & Slippage Assessment

| Component | Current Value | Realistic Crypto Value | Gap |
|---|---|---|---|
| Spread half-side (BTC alias QQQ) | 1.2 bps | 3-6 bps | **~3-5x too low** |
| Spread half-side (ETH alias SPY) | 0.8 bps | 4-8 bps | **~5-10x too low** |
| Base slippage half-side | 0.8 bps | 2-5 bps | **~2-6x too low** |
| Vol-spike slippage add | 2.0 bps | 5-15 bps | **~2-7x too low** |
| Commission round-trip | 1.0 bps | 5-10 bps (taker) or 2-4 bps (maker) | **~2-10x too low** |
| Vol-dependent slippage mult | 0.05 * sigma | Reasonable concept | Needs recalibration |

**Bottom line**: Total round-trip cost in the current model is roughly **5-8 bps**. Realistic crypto costs are **20-50 bps** depending on venue, liquidity, and volatility regime. This is the most concerning finding.

### Cost Stress Testing

The pipeline does run cost stress tests at 1.25x and 1.50x multipliers, which is good. The promoted result shows:
- Stress x1.25: avg monthly PnL = $171 (still positive)
- Stress x1.50: avg monthly PnL = $64 (barely positive)

But if base costs are 3-5x too low, even the 1.5x stress is still only testing at ~50-60% of realistic crypto costs.

### Overfitting Defenses (Generally Strong)

The system has multiple layers of overfitting protection:

1. **Walk-forward training** — monthly folds, no look-ahead ✅
2. **Nested cross-validation** — inner splits for hyperparameter tuning ✅
3. **Embargo periods** — 10-day embargo between train/test ✅
4. **Block bootstrap** — 800 samples with 6-month blocks for tail statistics ✅
5. **Cost stress tests** — 1.25x and 1.50x multipliers ✅
6. **Sign-flip significance test** — randomization test on monthly PnL ✅
7. **Concentration checks** — top-5/top-10 positive PnL share ✅
8. **Time-split stability** — first-half vs second-half comparison ✅
9. **Robustness gates** — multiple conditions required for promotion ✅

> [!IMPORTANT]
> The overfitting framework is genuinely well-designed. The weakness is **not** in the ML pipeline structure, but in the **friction model calibration** being inherited from equities.

---

## 3. Can We Benefit from a Third ML Model?

### Current Model Architecture

The system uses **2 ML models** per symbol (QQQ-leg and SPY-leg), each containing:
- **LightGBM classifier** — predicts P(profit) for each event
- **LightGBM regressor** — predicts expected return magnitude
- **Ridge linear model** — structural expected value estimator
- **Probability calibrator** — Platt scaling on raw probabilities
- **Confidence calibrator** — uncertainty-adjusted confidence scores

The final trade decision combines: `ev_final = mix_struct_weight * ev_struct + (1 - mix_struct_weight) * ret_pred`

### Recommendation: A Third "Meta-Model" Could Help, But Isn't Priority #1

A potential third model role:

| Option | Description | Value | Risk |
|---|---|---|---|
| **Regime classifier** | Predicts market regime (trend/mean-revert/chop) to gate trades | High — crypto regimes shift violently | Medium — must not overfit to regime labels |
| **Ensemble meta-learner** | Stacks predictions from both symbol models | Medium — could improve signal | High — adds complexity with limited OOS months |
| **Volatility forecaster** | Predicts next-period vol for better position sizing | Medium — helps risk management | Low — straightforward to implement |

**Verdict**: A **regime classifier** would be the highest-value third model, specifically to reduce trading in chop/ranging markets where the trend-following features are noisy. However, fix the friction model first — no model improvement matters if costs are wrong.

---

## 4. Should We Add a Third Asset (e.g., SOL)?

### Analysis

Adding a third asset creates combinatorial complexity:
- 3 symbol legs × 2 models each = 6 ML models
- Dynamic weight allocation across 3 assets
- Cross-feature matrix grows from 2×2 to 3×3

| Factor | Pro | Con |
|---|---|---|
| Diversification | SOL has different momentum patterns | Correlation with ETH is high (0.7-0.85) |
| Signal quality | More trading opportunities | Worse liquidity = higher costs |
| Complexity | Richer feature space | Much harder to avoid overfitting |
| Data length | SOL has shorter reliable history | Walk-forward needs years of data |

**Verdict: Not yet.** BTC/ETH is the right starting pair because:
1. They have the deepest liquidity (lowest execution costs)
2. They have the longest reliable history for walk-forward testing
3. The correlation (~0.6-0.8) still provides some diversification benefit
4. Adding SOL before fixing the friction model would just amplify the cost problem

If you do add one later, **SOL** is the best candidate due to its distinct momentum profile. But treat it as a Phase 2 after the crypto friction model is validated with real fills.

---

## 5. Algorithmic/Math Improvements and Execution Issues

### 🔴 P0: Fix Crypto Friction Model (Critical Before Any Live Trading)

**Changes needed in [step2_build_events_dataset.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/step2_build_events_dataset.py):**

1. Add crypto-specific spread defaults:
```python
k.spread_half_bps = {
    "SPY": 0.8,  "QQQ": 1.2,  "SMH": 1.8,   # equities
    "BTC": 3.0,  "ETH": 4.5,                  # crypto (conservative)
}
```

2. Add crypto commission tier:
```python
commission_round_trip_bps_crypto: float = 7.0  # taker on major exchange
```

3. Increase base slippage for crypto:
```python
slip_base_bps_crypto: float = 2.5
slip_spike_add_bps_crypto: float = 8.0
```

4. In `prepare_btc_eth_data.py`, override the alias friction when the pipeline sees crypto data, or create a `crypto_knobs()` factory.

### 🔴 P1: Remove/Rethink `entry_overnight` for 24/7 Markets

For crypto, replace the overnight detection with either:
- **UTC time-of-day based session splits** (e.g., Asian/European/US trading hours which have different liquidity profiles)
- **Weekend flag** (Saturday/Sunday have noticeably lower volume)
- Simply disable `entry_overnight` and use a single horizon/barrier config

### 🟡 P2: Crypto-Specific Feature Engineering

The current feature set was designed for US equity markets. Several features to add or modify:

| Feature | Description | Why It Helps |
|---|---|---|
| **Funding rate** | Perp funding rate (if trading futures) | Captures long/short crowding |
| **Weekend/hour seasonality** | UTC hour + day-of-week effects | Crypto has strong intraday patterns |
| **Vol-of-vol** | Rolling std of `sigma` | Captures regime instability |
| **Cross-pair spread momentum** | BTC/ETH ratio momentum at multiple horizons | More crypto-native than equity-style `rs_log` |
| **Order book imbalance** | Bid-ask depth ratio (if available) | Predicts short-term direction |

### 🟡 P3: Better Hybrid Overlay — Adaptive Core Rebalancing

Current hybrid uses fixed monthly core rebalancing. Improvements:
- **Correlation-adaptive core weight**: increase core when BTC-ETH correlation is high (less alpha from active), decrease when correlation is low
- **Drawdown-triggered sleeve reduction**: automatically reduce active sleeve when portfolio enters drawdown >15%
- **Momentum filter on core**: shift core weights based on 3-month price momentum (already partially done with `vol_parity_6m`, but could be extended)

### 🟡 P4: Walk-Forward Stress Expansion

Current: single expanding-window walk-forward. Improvements:
- Add **anchored + rolling** window comparison to detect if strategy edge is consistent or time-decaying
- Add **per-regime decomposition** in diagnostics (high-vol months vs low-vol months performance)
- Add **calendar-aware metrics** (crypto has known seasonal patterns, notably Q1 strength)

### 🟢 P5: Execution Considerations for Live Trading

When moving to live trading, these execution issues need attention:

1. **Latency**: Entry at "next bar open" assumes you can execute at the open price. In practice, you need a few seconds to detect the signal and place the order. Budget 5-10 seconds of market drift.
2. **Partial fills**: The model assumes full fills. For larger position sizes on ETH, this may not hold.
3. **Barrier monitoring**: The model checks barriers at 1-hour intervals. In prod, you'll want continuous monitoring or at least 1-minute checks to avoid missing stop-losses between bars.
4. **Exchange connectivity**: Need reliable websocket feeds + order management for 24/7 operation. Consider using a robust framework like CCXT or exchange-native APIs.

---

## Proposed Changes (Priority Order)

### Phase 1: Fix Critical Cost Issues (Must-Do Before Live)

#### [MODIFY] [step2_build_events_dataset.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/step2_build_events_dataset.py)
- Add `crypto_knobs()` factory with realistic crypto spreads (3-6 bps), slippage (2.5+ bps base), and commissions (5-10 bps RT)
- Add logic to detect crypto symbols and apply crypto cost parameters

#### [MODIFY] [prepare_btc_eth_data.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/other-pair/btc_eth/scripts/prepare_btc_eth_data.py)
- Write a `crypto_friction_overrides.json` file into the alias directory
- Pipeline reads this to apply crypto-specific costs

#### [MODIFY] [run_btc_eth_pipeline.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/other-pair/btc_eth/scripts/run_btc_eth_pipeline.py)
- Pass crypto friction overrides to step2/step3 pipeline calls

### Phase 2: Crypto Feature Refinement

#### [MODIFY] [step2_build_events_dataset.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/step2_build_events_dataset.py)
- Rework `entry_overnight` logic: for crypto, use UTC-hour session windows instead of NY-date crossings
- Add weekend flag as feature
- Add vol-of-vol feature (`sigma.rolling(N).std()`)

### Phase 3: Enhanced Hybrid Optimizer

#### [MODIFY] [optimize_step3_hybrid_btc_eth.py](file:///c:/dev/Coding/Trader/adaptive-systematic-trading/other-pair/btc_eth/scripts/optimize_step3_hybrid_btc_eth.py)
- Add correlation-adaptive core weighting mode
- Add drawdown-triggered sleeve reduction logic
- Add multi-horizon momentum to core allocation

### Phase 4: Live Trading Infrastructure

#### [NEW] `live_trading/` directory
- Exchange connector (CCXT-based)
- Real-time bar aggregation
- Signal generation + order management
- Position monitoring + barrier checking (continuous, not hourly)
- Risk limits + circuit breakers

---

## Verification Plan

### Automated Tests
1. **Re-run pipeline with crypto-realistic costs**:
```bash
python other-pair/btc_eth/scripts/run_btc_eth_pipeline.py --skip-download --start-capital 10000
```
Verify that with corrected friction, the system still produces positive monthly PnL (even if lower).

2. **Run overfit diagnostics on new results**:
```bash
python other-pair/btc_eth/scripts/build_overfit_diagnostics.py --step3-out-dir other-pair/btc_eth/step3_out
```
Compare risk scores before/after friction fix.

3. **Stress test at higher multipliers**: Run hybrid optimizer with `--min-stress150-avg-monthly-pnl 0` to see how the strategy degrades under realistic costs.

### Manual Verification
- Compare final `avg_monthly_pnl` and `calmar` before/after friction model correction
- If avg monthly PnL drops below ~$50 with realistic costs, the active ML sleeve may need fundamental rethinking (which would be an honest finding)
- Cross-reference current pipeline costs vs actual Binance/Coinbase fee schedules

---

## Summary of Verdicts

| Question | Verdict |
|---|---|
| Major bugs inflating results? | **Yes — friction model is ~3-5x too cheap for crypto.** Everything else is clean. |
| Fees/slippage/overfitting? | **Overfitting defenses are strong. Fee calibration is the weak link.** |
| Add a third ML model? | **Not priority. A regime classifier would be highest value, but fix costs first.** |
| Add a third trading pair? | **Not yet. BTC/ETH is the right start. SOL could be Phase 2.** |
| Algo/execution improvements? | **Multiple improvements identified. See P0-P5 above.** |
