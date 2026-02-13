# Technical Research Plan (v3)

## 0. High-Level Objective and Success Criteria
You are not trying to build a price oracle. You are trying to build a decision system that:

* Proposes trade opportunities with a deterministic signal engine.
* Uses a probabilistic model to estimate whether each opportunity has positive net expected value after friction.
* Expresses risk explicitly, especially overnight gap risk.
* Trades infrequently enough that costs, account constraints, and model drift do not dominate.

A **"good" research result** is not a "high win rate." It is defined by:
* Stable performance under realistic friction and slippage stress tests.
* Controlled drawdowns.
* Good calibration and no catastrophic regime blowups.
* A trade frequency that is high enough to learn, but low enough to keep costs from eating you.

---

## 1. Ontario + IBKR Constraints (First-Class Inputs)
You are in Ontario using IBKR and trading US-listed ETFs. These constraints must be treated as design constraints, even in research:

### 1.1 Regulatory and Account Mechanics Constraints
Depending on account type (cash vs. margin) and the rules applied to your account, you may be constrained in how often you can complete same-day round trips in US equities. Your research policy should avoid relying on frequent same-day round trips for performance. 

The technical spec includes frequency control and holding minimums:
* Minimum holding time unless a stop barrier is hit.
* Cooldown between exits and re-entries.
* No "flip-flop" behavior where the bot enters and exits repeatedly intraday.

This directly improves mathematical realism because it prevents tiny edge chasing, reduces churn and fee drag, and makes results more robust to execution variance.

### 1.2 Data Entitlements and IBKR API Limitations
Your live system will use the IBKR TWS or IB Gateway API for market data that your account is entitled to, as well as order placement and fill reports. Research should not assume unlimited free, high-quality historical intraday bars.

* Live trading will use IBKR data and fills.
* Research uses a primary historical source and at least one validation source when possible.
* "Data integrity" becomes part of the research spec, not an afterthought.

Your friction model is intentionally conservative and then calibrated using paper-trading fill logs from IBKR.

---

## 2. Universe Selection
### 2.1 Universe Choice
You start with one of:
* **QQQ**
* **SPY**

*(Optionally add SMH later.)*

These are liquid ETFs. This matters because backtests are less likely to be invalidated by fill assumptions, spreads are tighter on average, survivorship bias and corporate actions are simpler than single names, and you can keep the research pipeline focused on model validity rather than data chaos.

### 2.2 The Real Problem with "Boring"
The short-horizon edge is often small. That is why this plan has explicit anti-scalping rules:
* Do not trade unless the expected move dwarfs modeled costs.
* Do not trade marginal EV.

You do not "fix" low volatility by cranking trade frequency. You fix it by selecting only the rare moments when the conditional distribution is favorable.

---

## 3. Time Model, Bars, and Sessions (Fix B)
### 3.1 Bars and Timeframes
Let bar size be $\Delta$ (1h now, 4h later). All objects are defined in bar units so the same architecture works. Each bar $t$ has timestamp $T_t$.

### 3.2 Session Awareness
Define a trading session per day $d$:
$S_d = [t_{open}, t_{close}]$
Do not hard-code these. Parameterize them by exchange calendar.

### 3.3 Overnight Transition Indicator
The key variable:
$g_t = \mathbb{1}[\text{next tradable bar after } t \text{ opens in } S_{d+1}]$
This detects "close to next day open" discontinuities. The overnight transition is not treated as a normal bar.

### 3.4 Mathematical Significance
Intraday returns and overnight gap returns come from different distributions:
* Intraday returns are path-continuous in a loose sense.
* Overnight returns include jump-like behavior and latent information flow.

Model both separately: intraday volatility $\sigma_t$ and gap volatility $\sigma^{gap}_t$.

---

## 4. Core Data Objects and Derived Statistics (Bar-Agnostic)
### 4.1 OHLCV
For each bar:
$(O_t, H_t, L_t, C_t, V_t)$

### 4.2 Log Returns
Close-to-close:
$r_t = \ln(C_t / C_{t-1})$
Gap return:
$r^{gap}_t = \ln(O_{t+1} / C_t)$

### 4.3 Volatility Model (EWMA)
$v_t = \lambda \cdot r_t^2 + (1-\lambda) \cdot v_{t-1}$
$\sigma_t = \sqrt{v_t}$
This is robust, cheap, and naturally adjusts across 1h and 4h.

### 4.4 Range Proxy (ATR)
True range:
$TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$
ATR:
$ATR_t = \text{EMA}_n(TR_t)$
Normalized range:
$u_t = ATR_t / C_t$

### 4.5 Gap Risk Features (Explicit Block)
Compute gap distribution statistics from historical gaps only:
$\mu^{gap}_t = \text{mean}(r^{gap}_l)$
$\sigma^{gap}_t = \text{std}(r^{gap}_l)$
where $l$ indexes the last $M$ overnight events.

Also track gap percentile rank over a trailing window and tail frequency of gaps beyond a threshold. Example tail count:
$gapTail_t = \sum_{l \in M} \mathbb{1}(|r^{gap}_l| > k \cdot \sigma^{gap}_t)$
This turns "overnight-aware" into a measurable conditioning variable.

---

## 5. Execution and Friction Model
This system's PnL realism lives or dies by cost modeling. Treat friction as a random variable and calibrate it with IBKR paper fills.

### 5.1 Commission Model
Model per fill:
$fee = \max(minFee, perShareFee \cdot shares)$
Convert to return units by dividing by notional and approximating as a small additive cost in log-return space.

### 5.2 Spread Proxy
If lacking historical bid-ask data, use a conservative proxy tied to range:
$s_t = \max(s_{floor}, k_s \cdot u_t)$
*Interpretation:* In more volatile bars, assume worse effective spread and fill quality.

### 5.3 Slippage Proxy
$\iota_t = k_0 + k_1 \cdot \sigma_t + k_2 \cdot \mathbb{1}[\sigma_t > q_{0.9}]$
Injects regime sensitivity. When volatility spikes, assume slippage spikes.

### 5.4 Overnight Uncertainty Penalty
$\iota^*_t = \iota_t + k_{gap} \cdot g_t$
Acknowledges that holding across market closure introduces uncontrollable jump risk.

### 5.5 Execution Reference and Fill Modeling
Decision is made at bar close t. You execute at the next tradable bar open O_{t+1}. Maps directly to an IBKR API order placed after decision time.

Buy fill:
$\ln P^{buy}_{t+1} = \ln O_{t+1} + s_t + \iota^*_t$
Sell fill:
$\ln P^{sell}_{t+1} = \ln O_{t+1} - s_t - \iota^*_t$

### 5.6 Total Friction Term for Gating and Labeling
$c_t \approx c^{comm}_t + 2 \cdot s_t + 2 \cdot \iota^*_t$

### 5.7 Calibration Loop (Research Spec)
Estimate (k_0, k_1, k_2, k_gap, k_s, s_floor) by matching the distribution of modeled slippage to IBKR paper-trading fills, conditioned on time of day bucket, volatility regime bucket, and overnight vs. intraday transition.

---

## 6. Candidate Generation (Signal Engine)
The engine should be simple, interpretable, and generate enough opportunities to learn while the ML gate decides what to trade.

### 6.1 Trend Score (Vol-Normalized)
$E_f(t) = \text{EMA}_f(C_t)$
$E_s(t) = \text{EMA}_s(C_t)$
$m_t = (E_f(t) - E_s(t)) / C_t$
$T_t = m_t / \sigma_t$

Candidate trend entry:
$T_t > \tau_{in}$
Candidate trend exit condition:
$T_t < \tau_{out}$

### 6.2 Pullback-in-Trend
Trend regime:
$R_t = \mathbb{1}[T_t > \tau_{trend}]$
Pullback z-score:
$x_t = \ln C_t$
$\mu_t = \ln E_f(t)$
$Z^{pb}_t = (x_t - \mu_t) / \sigma^x_t$

Candidate pullback entry:
$R_t = 1 \text{ AND } Z^{pb}_t < -\tau_{pb}$

### 6.3 Candidate Families as Separate "Alphas"
Explicitly label each candidate with a family ID:
* **Family A:** Trend continuation
* **Family B:** Pullback in trend

This lets you learn which family works in which regime and avoids blending them into a single fragile model.

### 6.4 Frequency Controls
To prevent cost drag and account constraint pain:
* Minimum holding time H_min bars unless stop barrier hit.
* Cooldown K bars after exit.
* Limit on total round trips per week (optional governor).

---

## 7. Trade Lifecycle Labeling (Triple Barrier, Overnight-Aware)
### 7.1 Barrier Scaling
Let: $u_t = ATR_t / C_t$

Define two parameter sets.
**Intraday:**
$a^{intra}_t = \alpha_{intra} \cdot u_t$
$b^{intra}_t = \beta_{intra} \cdot u_t$
$H_{intra}$

**Overnight:**
$a^{overn}_t = \alpha_{overn} \cdot u_t$
$b^{overn}_t = \beta_{overn} \cdot u_t$
$H_{overn}$

Select by $g_t$:
$(a_t, b_t, H) = (a^{intra}_t, b^{intra}_t, H_{intra})$ if $g_t = 0$
$(a_t, b_t, H) = (a^{overn}_t, b^{overn}_t, H_{overn})$ if $g_t = 1$

### 7.2 Path Return and Barrier Hitting
Entry at $P^{buy}_{t+1}$. For $k \in [1, H]$:
$\Delta_{t,k} = \ln(C_{t+k} / P^{buy}_{t+1})$

Hit times:
$\tau^+ = \inf \{k \le H : \Delta_{t,k} \ge a_t\}$
$\tau^- = \inf \{k \le H : \Delta_{t,k} \le -b_t\}$

Exit:
$\tau = \min(\tau^+, \tau^-, H)$

Net return:
$\Delta^{net}_t = \Delta_{t,\tau} - c_t$

Binary label:
$y_t = \mathbb{1}[\Delta^{net}_t > 0]$

### 7.3 Regression Labels Upgrade
Store classification label y_t, regression target $\Delta^{net}_t$, and optionally quantile targets ($Q_{0.1}$, $Q_{0.5}$, $Q_{0.9}$). This enables EV or quantile prediction later without rewriting the dataset.

---

## 8. Feature Design
### 8.1 Feature Blocks
* **Block 1: Signal Geometry:** T_t, Z^pb_t, EMA slopes (s_f(t) = (E_f(t) - E_f(t-1)) / C_t), distance to rolling high (d^hi_t = (C_t - max(C_{t-n..t-1})) / C_t).
* **Block 2: Volatility and Tails:** σ_t, volatility percentile rank, u_t, ATR percentile rank, tail count on intraday returns.
* **Block 3: Gap-Risk Block:** g_t, μ^gap_t, σ^gap_t, gapTail_t, recent gap magnitude percentiles.
* **Block 4: Liquidity Proxies:** Volume z-score, range ratio ((H_t - L_t) / C_t).
* **Block 5: Multi-Timeframe Context:** Compute 4h trend score from 4h bars and align as a feature on 1h candidates. Compute daily regime variable and align it.

### 8.2 Feature Discipline Rules
* Hard cap features for MVP (e.g., 30 to 60 total).
* No "feature mining" from thousands of indicators.
* No news ingestion in MVP.
* No cross-asset feature explosion (at most one context series early, like SPY when trading QQQ).

---

## 9. Modeling Choices
### 9.1 Model Families
* **Baseline:** Regularized logistic regression: $p_t = \sigma(\beta^T \cdot x_t)$
* **Main:** Boosted trees (XGBoost or LightGBM). Avoid neural networks in MVP.

### 9.2 Training Objective Options
Run both classification for y_t and regression for $\Delta^{net}_t$ or quantiles. 

### 9.3 Walk-Forward Training
Training is monthly walk-forward: train on trailing L months, validate on next month, roll forward.

### 9.4 Purge and Embargo
Because labels depend on future bars up to H, enforce:
* **Purge:** Any training examples whose label window overlaps the validation window.
* **Embargo:** H bars between train and validation/test.

### 9.5 Calibration
You size and gate using predicted probabilities, so calibration is mandatory per fold (Platt scaling or isotonic regression). Track Brier score and calibration drift.

---

## 10. Decision Policy: "Trade Only When Paid"
### 10.1 Expected Value Gate
Given $p_t$, barriers $(a_t, b_t)$, and cost $c_t$:
$\hat{EV}_t = p_t \cdot a_t - (1 - p_t) \cdot b_t - c_t$

Trade only if:
$\hat{EV}_t > \kappa \cdot c_t$
*(where κ is a safety multiplier)*

### 10.2 Anti-Scalping Rule
To avoid microscopic hourly edges: $a_t \ge \phi \cdot c_t$
*(with φ typically 5 to 10)*

### 10.3 Overnight Tightening Rule
If g_t = 1, enforce an additional margin: $\hat{EV}_t > \kappa_{overn} \cdot c_t$
*(where κ_overn > κ)*

### 10.4 Sizing (Bounded, Robust)
Base sizing from probability:
$w_t = w_{max} \cdot \text{clip}((p_t - p_0) / (p_1 - p_0), 0, 1)$

Overnight penalty:
$w_t \leftarrow w_t \cdot (1 - \delta \cdot g_t)$

Optional volatility targeting:
$w_t \leftarrow w_t \cdot \text{clip}(\sigma_{target} / \sigma_t, 0, 1)$

---

## 11. Risk Control Spec
### 11.1 Drawdown Kill Switch
Let equity curve be $E_t$:
$DD_t = 1 - E_t / \max_{u \le t}(E_u)$
If $DD_t > DD_{max}$, go flat for cooldown $K_{dd}$.

### 11.2 Session Loss Stop
If realized session PnL drops below $-L_{day}$, stop entering new trades for the remainder of the session.

### 11.3 Turnover Governor
* Do not resize if $|\Delta w| < \varepsilon_w$.
* Do not re-enter immediately after exit.
* Enforce minimum holding time except stop.

---

## 12. Shadow Book vs. Live Book
Formalize two books to generate enough data to learn without overtrading live.

### 12.1 Shadow Book
Paper trade all candidates with a minimal filter (e.g., p_t > 0.5). Yields many resolved outcomes per month and fast feedback on calibration.

### 12.2 Live Book
Only trade strict passes (EV gate, anti-scalping gate, overnight tightening, risk controls). Yields fewer, higher-quality trades.

---

## 13. Timeframe Portability to 4H
This plan is bar-agnostic because barriers scale with u_t, volatility is per-bar, horizon is in bars, and overnight logic uses timestamps. 

When moving to 4h evaluation:
* Convert EMA spans and horizons from hours to bars.
* Recompute features and labels at 4h resolution.
* Retrain with the same walk-forward discipline.
* Compare trade frequency, cost sensitivity, drawdown behavior, and calibration stability.

---

## 14. Performance Levers
For this setup, gains increase from:
* Better filtering (EV, quantiles, conservative gates).
* Better regime awareness (gap risk and volatility regimes).
* Better turnover control.
* Better calibration and conservative sizing.

If you want a single "smarter algorithm" upgrade later:
* Quantile regression with boosted trees.
* Gate on $Q_{0.1}(\Delta^{net} | x) > 0$.