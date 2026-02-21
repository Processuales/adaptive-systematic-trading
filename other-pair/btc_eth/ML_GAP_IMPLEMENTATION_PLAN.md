# BTC + ETH ML Gap Implementation Plan

## Goal
Explain why BTC+ETH Step 3 ML underperformed vs Step 2 simulation and define the exact roadmap to improve it without relying on noisy overfit wins.

## Current Snapshot
- Step 2 ML simulation:
  - `avg_monthly_pnl`: `245.58`
  - `end_equity`: `23506.91`
- Step 3 tuned real ML:
  - `avg_monthly_pnl`: `44.17`
  - `end_equity`: `12031.65`
  - `stress_1.25_avg_monthly_pnl`: `24.11`
  - `stress_1.50_avg_monthly_pnl`: `5.59`
  - `bootstrap_p10_avg_monthly_pnl`: `-26.45`

## Why ML Was Worse Than Original Simulation
1. Step 2 and Step 3 are different statistical regimes.
- Step 2 is a lighter candidate search/simulation framework and tends to be optimistic for crypto.
- Step 3 is stricter walk-forward ML with embargo, drift checks, stress costs, and bootstrap tails.

2. Symbol asymmetry is strong in BTC+ETH.
- Active BTC leg was unstable in walk-forward tests.
- ETH leg carried most positive edge.
- Best Step 3 result came from forcing BTC active allocation near zero.

3. Pattern-aid ML is currently unstable for this pair.
- Pattern-aid variants degraded tuned baseline substantially in direct tests.
- Conclusion: keep pattern-aid disabled for BTC+ETH until it passes robust OOS gates.

4. Passive trend capture is missing in active-only design.
- In the same Step 3 date range, buy-and-hold BTC was far stronger than active BTC model behavior.

## Decisions Locked In
- Keep BTC+ETH pattern-aid disabled by default.
- Keep current tuned active baseline:
  - `mix_struct_weight=0.65`
  - `max_aggressive_size=1.0`
  - `portfolio_allocator=dynamic_regime_forced`
  - `portfolio_min_weight=0.0`
  - `portfolio_max_weight=0.005`

## Phase Plan

### Phase 1: Hybrid Core-Satellite Portfolio
- Add passive core + active sleeve:
  - Core: BTC/ETH long-only benchmark weights.
  - Active: current Step 3 ML overlay.
- Search:
  - core fraction in `[0.3, 0.9]`
  - core BTC share in `[0.3, 0.9]`
  - active sleeve caps
- Accept only if robust metrics also improve.

### Phase 2: Crypto-Native Feature Upgrade
- Add features specific to 24/7 markets:
  - weekend/weekday + UTC-hour context
  - volatility-of-volatility
  - multi-horizon relative momentum
  - cross-asset spread dynamics
- Keep same walk-forward validation protocol.

### Phase 3: Robustness-First Promotion
- Candidate must pass all hard gates:
  - `stress_1.25_avg_monthly_pnl > 0`
  - `stress_1.50_avg_monthly_pnl >= 0`
  - `bootstrap_p10_avg_monthly_pnl > 0`
  - drawdown hard cap
- If no pass: fallback to safer baseline/hybrid.

### Phase 4: Cost and Slippage Stress Expansion
- Add realistic crypto cost tiers:
  - base
  - stressed
  - extreme
- Promote only if performance stays acceptable under stressed tier.

## Validation Matrix
- Track A: Active-only baseline vs hybrid variants.
- Track B: Current features vs crypto-native features.
- Track C: Pattern-aid off vs limited pattern-aid (strict gate).

Primary ranking metrics:
- `avg_monthly_pnl`
- `end_equity`
- `calmar`
- `max_drawdown`
- `stress_1.25/1.50 avg_monthly_pnl`
- `bootstrap p10 avg_monthly_pnl`

## Promotion Rule
Promote new config only if:
- return improves meaningfully, and
- risk robustness is non-worse (or better) on stress and bootstrap tails.
