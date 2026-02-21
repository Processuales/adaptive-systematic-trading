# BTC + ETH Moonshot Implementation Plan

## 1) Current Findings (What the data says)
- Step 2 ML simulation is extremely strong in-sample style:
  - `end_equity`: `23506.91`
  - `avg_monthly_pnl`: `245.58`
- Step 3 real ML (strict walk-forward) is much lower:
  - current tuned best `end_equity`: `12031.65`
  - `avg_monthly_pnl`: `44.17`
  - bootstrap `p10 avg_monthly_pnl`: `-26.45` (tail risk still weak)
- Pattern-aid on tuned config is harmful:
  - tuned baseline: `avg_monthly_pnl=44.17`, `end_equity=12031.65`
  - tuned + pattern balanced: `avg_monthly_pnl=0.27`, `end_equity=10012.34`
- Buy-and-hold benchmark over Step 3 window (2022-04-19 to 2026-01-29):
  - BTC hold: `21763.63`
  - ETH hold: `9813.62`
  - 50/50 hold: `15788.62`
  - Step 3 tuned ML: `12031.65`

## 2) Root-Cause Hypothesis
- Step 2 selection is likely optimistic for crypto (single split + broad search + candidate selection on limited test trades).
- Step 3 is stricter and reveals weak robustness (especially BTC leg instability).
- Current trading-only design misses large passive trend gains in BTC (model often avoids BTC, while hold BTC was much stronger in this window).
- Pattern clustering is currently overfitting/unstable in BTC/ETH regime shifts.

## 3) Immediate Direction
- Keep pattern-aid OFF for BTC/ETH until it proves robust in independent folds.
- Keep BTC ultra-capped in active allocator for now (`portfolio_max_weight=0.005`) in Step 3 active sleeve.
- Add passive + active hybrid architecture so model does not miss structural crypto trend.

## 4) High-Value Upgrades (Priority Order)

### P0: Hybrid Core-Satellite Portfolio (must-have)
- Add a passive core allocation + active ML overlay:
  - Core: long-only BTC/ETH benchmark weights (e.g., 60/40, 70/30, dynamic vol-parity).
  - Satellite: existing Step 3 active strategy on remaining capital.
- Optimize over:
  - core weight in `[0.3, 0.9]`
  - BTC/ETH core split grid
  - active sleeve risk cap
- Success criterion:
  - beat tuned active-only Step 3 in `end_equity` and `calmar`
  - non-worse cost stress (1.25x, 1.5x)

### P1: Crypto-Native Feature Set
- Replace equity-style assumptions:
  - remove/neutralize session-based semantics that are weak in 24/7 markets.
- Add crypto-native context:
  - rolling trend persistence (multiple horizons)
  - volatility-of-volatility
  - weekend/weekday and UTC-hour seasonality
  - cross-asset relative momentum / spread dynamics
- Success criterion:
  - improves fold stability (`negative_test_calmar_rate`)
  - improves bootstrap `p10 avg_monthly_pnl`

### P2: Robustness-First Optimization Gate
- Candidate must pass all:
  - `bootstrap p10 avg_monthly_pnl > 0`
  - `stress 1.25 avg_monthly_pnl > 0`
  - `max_drawdown <= configured hard cap`
  - minimum monthly trade floor
- If no candidate passes, fallback to safest benchmark/hybrid mode.

### P3: Walk-Forward Stress Expansion
- Add multi-window OOS tests:
  - rolling windows with different train/test spans
  - anchored + rolling combination
- Add diagnostics:
  - metrics per regime bucket (high vol, low vol, trend up/down)
  - concentration metrics (top months/day dependence)

### P4: Cost Realism Upgrade
- Expand fee/slippage assumptions with market-impact tiers for crypto:
  - normal, stressed, extreme.
- Require positive expectancy under stressed tier before promotion.

### P5: BTC/ETH Strategy Decomposition
- Evaluate symbol legs separately as products:
  - promote ETH active if strong
  - only enable BTC active under explicit confidence/regime conditions
- Keep allocator capable of near-zero active BTC when edge absent.

## 5) Test Matrix for Next Iteration

### Track A: Hybrid overlay
- A1: `core=50%`, active=50%
- A2: `core=70%`, active=30%
- A3: `core dynamic vol parity`, active fixed sleeve
- Rank by:
  - `end_equity`, `calmar`, `max_drawdown`
  - stress 1.25/1.5 monthly PnL
  - bootstrap p10 monthly PnL

### Track B: Feature upgrades
- B1: add seasonality + vol-of-vol only
- B2: add spread-relative momentum block
- B3: full crypto-native feature bundle
- Keep same training protocol; compare fold stability deltas.

### Track C: Safety/production rules
- C1: baseline hard risk gates
- C2: stricter robust gate
- C3: strict gate + fallback to core-only mode

## 6) Promotion Rules
- Promote only if BOTH:
  - `avg_monthly_pnl` improves by meaningful margin over current tuned best
  - robustness improves or stays flat (`dd`, stress tests, bootstrap p10)
- Hard reject if pattern-aid variant degrades stress tests or bootstrap tail.

## 7) What to keep right now
- Keep current tuned Step 3 BTC/ETH config as active baseline:
  - `mix_struct_weight=0.65`
  - `max_aggressive_size=1.0`
  - `portfolio_allocator=dynamic_regime_forced`
  - `portfolio_min_weight=0.0`
  - `portfolio_max_weight=0.005`
- Keep pattern ML disabled for BTC/ETH in production path until re-validated.
