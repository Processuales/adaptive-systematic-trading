#!/usr/bin/env python3
"""
Dual-symbol (SPY+QQQ) portfolio test using Step 2 candidates.

Builds a walk-forward, regime-aware allocation between QQQ and SPY strategies.

Outputs:
  - <out-dir>/dual_symbol_portfolio_curve.png
  - <out-dir>/dual_symbol_portfolio_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_build_events_dataset as s2
import step2b_knob_sweep_backtest as s2b

SCRIPT_VERSION = "1.1.0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_obj(obj, ndigits: int = 6):
    if isinstance(obj, dict):
        return {k: round_obj(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_obj(v, ndigits) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, ndigits)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def trial_metric(
    trial: Dict,
    mode: str,
    objective: str,
    min_test_trades: int,
    target_test_trades: int,
    trade_weight: float,
    edge_weight: float,
    dd_cap: float,
    dd_penalty: float,
    min_net_bps: float,
    min_cagr: float,
) -> float:
    test = trial[mode]["test"]
    n = int(test.get("n") or 0)
    if n < min_test_trades:
        return -1e18
    end_eq = float(test.get("end_equity") or 0.0)
    dd = float(test.get("max_drawdown")) if test.get("max_drawdown") is not None else 1.0
    net_bps = float(test.get("net_bps_mean")) if test.get("net_bps_mean") is not None else -1e9
    cagr = float(test.get("cagr")) if test.get("cagr") is not None else None
    if cagr is not None and cagr < min_cagr:
        return -1e18
    if net_bps < min_net_bps:
        return -1e18

    if objective == "end_equity":
        base = (end_eq / 10000.0) - 1.0
    else:
        if cagr is None:
            return -1e18
        if dd <= 0:
            base = cagr
        else:
            base = cagr / dd
    trade_bonus = trade_weight * (np.sqrt(max(n, 1)) + 0.25 * max(0, n - target_test_trades))
    edge_bonus = edge_weight * net_bps
    dd_pen = dd_penalty * max(0.0, dd - dd_cap)
    score = base + trade_bonus + edge_bonus - dd_pen
    if not np.isfinite(score):
        return -1e18
    return float(score)


def pick_best_per_symbol(
    selection_dir: str,
    mode: str,
    objective: str,
    min_test_trades: int,
    target_test_trades: int,
    trade_weight: float,
    edge_weight: float,
    dd_cap: float,
    dd_penalty: float,
    min_net_bps: float,
    min_cagr: float,
) -> Tuple[Dict, Dict]:
    scenarios = {
        "qqq_cross_off": ("QQQ", False, "SPY"),
        "qqq_cross_on": ("QQQ", True, "SPY"),
        "spy_cross_off": ("SPY", False, "QQQ"),
        "spy_cross_on": ("SPY", True, "QQQ"),
    }
    trials_key = "top_trials_no_ml" if mode == "no_ml" else "top_trials_ml"

    picks: Dict[str, Dict] = {}
    for name, (trade_sym, include_cross, cross_sym) in scenarios.items():
        p = os.path.join(selection_dir, name, "step2b_summary.json")
        if not os.path.exists(p):
            continue
        summary = load_json(p)
        trials = summary.get(trials_key, [])
        if not trials:
            continue
        for trial in trials:
            score = trial_metric(
                trial=trial,
                mode=mode,
                objective=objective,
                min_test_trades=min_test_trades,
                target_test_trades=target_test_trades,
                trade_weight=trade_weight,
                edge_weight=edge_weight,
                dd_cap=dd_cap,
                dd_penalty=dd_penalty,
                min_net_bps=min_net_bps,
                min_cagr=min_cagr,
            )
            cand = {
                "scenario_name": name,
                "include_cross": include_cross,
                "trade_symbol": trade_sym,
                "cross_symbol": cross_sym,
                "symbols": ["SPY", "QQQ"],
                "trial": trial,
                "score": score,
            }
            cur = picks.get(trade_sym)
            if cur is None or score > cur["score"]:
                picks[trade_sym] = cand

    if "QQQ" not in picks or "SPY" not in picks:
        raise RuntimeError(
            "Could not find both QQQ and SPY candidates in selection directory. "
            "Run step2_compare_and_select.py first."
        )
    if picks["QQQ"]["score"] <= -1e17 or picks["SPY"]["score"] <= -1e17:
        raise RuntimeError(
            "Found candidates, but none met --min-test-trades / objective requirements. "
            "Lower --min-test-trades or rerun Step 2b with more trials."
        )
    return picks["QQQ"], picks["SPY"]


def load_raw_data(data_dir: str, symbols: set[str], suffix: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in sorted(symbols):
        p = os.path.join(data_dir, f"{sym.lower()}{suffix}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cleaned data file: {p}")
        out[sym] = s2.read_parquet_any(p)
    return out


def daily_equity_from_events(events: pd.DataFrame, start_capital: float) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float)
    x = events.sort_values("exit_time_utc").copy()
    x["exit_time_utc"] = pd.to_datetime(x["exit_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["exit_time_utc"])
    if x.empty:
        return pd.Series(dtype=float)
    eq = start_capital * np.exp(x["net_logret"].to_numpy(dtype=float).cumsum())
    s = pd.Series(eq, index=x["exit_time_utc"]).resample("D").last()
    return s


def daily_trade_counts(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float)
    x = events.copy()
    x["exit_time_utc"] = pd.to_datetime(x["exit_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["exit_time_utc"])
    if x.empty:
        return pd.Series(dtype=float)
    return x.set_index("exit_time_utc").resample("D").size().astype(float)


def perf_from_equity_series(eq: pd.Series, start_capital: float) -> Dict:
    if eq.empty:
        return {
            "end_equity": start_capital,
            "cagr": None,
            "max_drawdown": None,
            "calmar": None,
            "n_days": 0,
        }
    e = eq.to_numpy(dtype=float)
    peak = np.maximum.accumulate(e)
    dd = 1.0 - (e / peak)
    mdd = float(dd.max())

    t0 = eq.index[0]
    t1 = eq.index[-1]
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (float(e[-1]) / float(start_capital)) ** (1.0 / years) - 1.0
    calmar = float(cagr / mdd) if mdd > 0 else None

    return {
        "start_time_utc": str(t0),
        "end_time_utc": str(t1),
        "end_equity": float(e[-1]),
        "cagr": float(cagr),
        "max_drawdown": mdd,
        "calmar": calmar,
        "n_days": int(len(eq)),
    }


def score_returns(r: pd.Series, objective: str) -> float:
    if r.empty:
        return -1e18
    x = r.to_numpy(dtype=float)
    eq = np.cumprod(1.0 + x)
    end_eq = float(eq[-1])
    if not np.isfinite(end_eq) or end_eq <= 0:
        return -1e18
    if objective == "end_equity":
        return end_eq
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    mdd = float(dd.max())
    years = max(len(eq) / 365.25, 1e-9)
    cagr = end_eq ** (1.0 / years) - 1.0
    if mdd <= 0:
        return float(cagr)
    return float(cagr / mdd)


def optimize_weight_window(
    q_hist: pd.Series,
    s_hist: pd.Series,
    weights_grid: np.ndarray,
    objective: str,
    prev_w: float,
    turnover_penalty: float,
) -> float:
    best_w = prev_w
    best_score = -1e18
    for w in weights_grid:
        r_hist = w * q_hist + (1.0 - w) * s_hist
        score = score_returns(r_hist, objective) - turnover_penalty * abs(float(w) - prev_w)
        if score > best_score:
            best_score = score
            best_w = float(w)
    return float(best_w)


def regime_tilt(
    q_hist: pd.Series,
    s_hist: pd.Series,
    momentum_days: int,
    vol_days: int,
    vol_penalty: float,
    max_tilt: float,
) -> float:
    if len(q_hist) < max(momentum_days, vol_days) or len(s_hist) < max(momentum_days, vol_days):
        return 0.0
    q_mom = float((1.0 + q_hist.iloc[-momentum_days:]).prod() - 1.0)
    s_mom = float((1.0 + s_hist.iloc[-momentum_days:]).prod() - 1.0)
    q_vol = float(q_hist.iloc[-vol_days:].std(ddof=0))
    s_vol = float(s_hist.iloc[-vol_days:].std(ddof=0))
    q_score = q_mom - vol_penalty * q_vol
    s_score = s_mom - vol_penalty * s_vol
    scale = max(abs(q_score) + abs(s_score), 1e-9)
    raw = (q_score - s_score) / scale
    return float(max_tilt * np.tanh(2.0 * raw))


def build_walk_forward_weights(
    idx: pd.DatetimeIndex,
    q_ret: pd.Series,
    s_ret: pd.Series,
    train_ratio: float,
    lookback_days: int,
    min_train_days: int,
    weight_step: float,
    objective: str,
    turnover_penalty: float,
    momentum_days: int,
    vol_days: int,
    vol_penalty: float,
    max_tilt: float,
    weight_smoothing: float,
) -> pd.Series:
    weights_grid = np.arange(0.0, 1.0 + 1e-12, weight_step)
    w_series = pd.Series(np.nan, index=idx, dtype=float)

    warmup_days = max(min_train_days, int(lookback_days * train_ratio))
    warmup_days = min(warmup_days, max(1, len(idx) - 1))

    # Seed with a static weight fit on the initial warmup block.
    q_seed = q_ret.iloc[:warmup_days]
    s_seed = s_ret.iloc[:warmup_days]
    seed_w = optimize_weight_window(
        q_hist=q_seed,
        s_hist=s_seed,
        weights_grid=weights_grid,
        objective=objective,
        prev_w=0.5,
        turnover_penalty=0.0,
    )

    month_codes = idx.tz_localize(None).to_period("M")
    unique_months = month_codes.unique()

    prev_w = float(seed_w)
    for month in unique_months:
        mask = month_codes == month
        month_idx = np.flatnonzero(mask)
        if len(month_idx) == 0:
            continue
        month_start = int(month_idx[0])

        if month_start < warmup_days:
            w_final = prev_w
        else:
            hist_end = month_start
            hist_start = max(0, hist_end - lookback_days)
            q_hist = q_ret.iloc[hist_start:hist_end]
            s_hist = s_ret.iloc[hist_start:hist_end]

            if len(q_hist) < min_train_days or len(s_hist) < min_train_days:
                w_opt = prev_w
            else:
                w_opt = optimize_weight_window(
                    q_hist=q_hist,
                    s_hist=s_hist,
                    weights_grid=weights_grid,
                    objective=objective,
                    prev_w=prev_w,
                    turnover_penalty=turnover_penalty,
                )
            tilt = regime_tilt(
                q_hist=q_hist,
                s_hist=s_hist,
                momentum_days=momentum_days,
                vol_days=vol_days,
                vol_penalty=vol_penalty,
                max_tilt=max_tilt,
            )
            w_raw = float(np.clip(w_opt + tilt, 0.0, 1.0))
            w_final = float(weight_smoothing * prev_w + (1.0 - weight_smoothing) * w_raw)

        w_series.iloc[month_idx] = float(np.clip(w_final, 0.0, 1.0))
        prev_w = float(np.clip(w_final, 0.0, 1.0))

    return w_series.fillna(0.5)


def build_static_weight_series(
    idx: pd.DatetimeIndex,
    q_ret: pd.Series,
    s_ret: pd.Series,
    train_ratio: float,
    weight_step: float,
    objective: str,
) -> Tuple[pd.Series, int]:
    split_i = int(len(idx) * train_ratio)
    split_i = min(max(split_i, 2), len(idx) - 1)
    train_slice = slice(0, split_i)
    weights_grid = np.arange(0.0, 1.0 + 1e-12, weight_step)

    best_w = 0.5
    best_score = -1e18
    for w in weights_grid:
        r_train = w * q_ret.iloc[train_slice] + (1.0 - w) * s_ret.iloc[train_slice]
        score = score_returns(r_train, objective)
        if score > best_score:
            best_score = score
            best_w = float(w)
    return pd.Series(best_w, index=idx, dtype=float), split_i


def monthly_table(eq: pd.Series, trades: pd.Series, start_capital: float) -> pd.DataFrame:
    if eq.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    month_eq = eq.resample("ME").last()
    prev = month_eq.shift(1)
    prev.iloc[0] = start_capital
    pnl = month_eq - prev
    ret = month_eq / prev - 1.0
    trades_m = trades.resample("ME").sum().reindex(month_eq.index).fillna(0.0)
    return pd.DataFrame(
        {
            "month_end": month_eq.index,
            "equity": month_eq.to_numpy(dtype=float),
            "pnl": pnl.to_numpy(dtype=float),
            "ret": ret.to_numpy(dtype=float),
            "trades": trades_m.to_numpy(dtype=float),
        }
    )


def as_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def plot_portfolio(
    out_path: str,
    eq: pd.Series,
    weights: pd.Series,
    monthly: pd.DataFrame,
    stats_text: str,
    start_capital: float,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        gridspec_kw={"height_ratios": [3.0, 1.6]},
    )

    ax1.plot(eq.index, eq.values, color="#1d3557", linewidth=2.1, label="Dual portfolio equity")
    ax1.axhline(start_capital, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_title("SPY + QQQ Dual Portfolio (Walk-Forward Regime Allocation)")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.25)

    ax1b = ax1.twinx()
    ax1b.plot(weights.index, weights.values, color="#457b9d", linewidth=1.1, alpha=0.45, label="QQQ weight")
    ax1b.set_ylim(0.0, 1.0)
    ax1b.set_ylabel("QQQ Weight")

    ax1.text(
        0.01,
        0.99,
        stats_text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.84, "edgecolor": "#999999"},
    )

    pnl = monthly["pnl"].to_numpy(dtype=float)
    colors = np.where(pnl >= 0.0, "#2a9d8f", "#d62828")
    ax2.bar(monthly["month_end"], pnl, width=20, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_title("Monthly PnL")
    ax2.set_ylabel("PnL ($)")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to cleaned bars")
    ap.add_argument("--selection-dir", required=True, help="Path to step2_out/selection")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["no_ml", "ml_sim"], default="no_ml")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet")
    ap.add_argument("--weight-step", type=float, default=0.05, help="Grid step for QQQ weight search")
    ap.add_argument("--train-ratio", type=float, default=0.60, help="Warmup fraction before walk-forward allocation")
    ap.add_argument("--objective", choices=["calmar", "end_equity"], default="calmar")
    ap.add_argument("--lookback-days", type=int, default=756, help="Trailing days per monthly optimization")
    ap.add_argument("--min-train-days", type=int, default=252, help="Minimum history required for optimization")
    ap.add_argument("--turnover-penalty", type=float, default=0.04, help="Penalty for month-to-month weight jumps")
    ap.add_argument("--weight-smoothing", type=float, default=0.35, help="Blend with previous month weight [0,1]")
    ap.add_argument("--regime-momentum-days", type=int, default=63, help="Momentum lookback for regime tilt")
    ap.add_argument("--regime-vol-days", type=int, default=21, help="Volatility lookback for regime tilt")
    ap.add_argument("--regime-vol-penalty", type=float, default=1.5, help="Vol penalty in regime score")
    ap.add_argument("--regime-max-tilt", type=float, default=0.20, help="Max additive tilt to optimized weight")
    ap.add_argument("--min-active-weight", type=float, default=0.05, help="Weight threshold for counting symbol trades")
    ap.add_argument("--min-test-trades", type=int, default=25, help="Minimum test trades for candidate eligibility")
    ap.add_argument("--target-test-trades", type=int, default=45, help="Preferred minimum trade count in test")
    ap.add_argument("--candidate-trade-weight", type=float, default=0.08, help="Candidate score weight for trade count")
    ap.add_argument("--candidate-edge-weight", type=float, default=0.02, help="Candidate score weight for net_bps_mean")
    ap.add_argument("--candidate-dd-cap", type=float, default=0.09, help="Drawdown cap before candidate penalty")
    ap.add_argument("--candidate-dd-penalty", type=float, default=3.0, help="Penalty per unit drawdown above cap")
    ap.add_argument("--candidate-min-net-bps", type=float, default=6.0, help="Minimum net_bps_mean required for candidate")
    ap.add_argument("--candidate-min-cagr", type=float, default=0.0, help="Minimum CAGR required for candidate")
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 < args.weight_step <= 1.0):
        raise ValueError("--weight-step must be in (0, 1]")
    if not (0.0 <= args.weight_smoothing <= 1.0):
        raise ValueError("--weight-smoothing must be in [0, 1]")
    if args.lookback_days < 30 or args.min_train_days < 30:
        raise ValueError("--lookback-days and --min-train-days should both be >= 30")

    ensure_dir(args.out_dir)

    qqq_cand, spy_cand = pick_best_per_symbol(
        selection_dir=args.selection_dir,
        mode=args.mode,
        objective=args.objective,
        min_test_trades=args.min_test_trades,
        target_test_trades=args.target_test_trades,
        trade_weight=args.candidate_trade_weight,
        edge_weight=args.candidate_edge_weight,
        dd_cap=args.candidate_dd_cap,
        dd_penalty=args.candidate_dd_penalty,
        min_net_bps=args.candidate_min_net_bps,
        min_cagr=args.candidate_min_cagr,
    )

    symbols = {"SPY", "QQQ"}
    raw_by_sym = load_raw_data(args.data_dir, symbols, args.bar_file_suffix)

    q_no_events, q_ml_events = s2b.build_full_events_for_result(
        result=qqq_cand["trial"],
        raw_by_sym=raw_by_sym,
        symbols=qqq_cand["symbols"],
        trade_symbol=qqq_cand["trade_symbol"],
        cross_symbol=qqq_cand["cross_symbol"],
        include_cross=bool(qqq_cand["include_cross"]),
    )
    s_no_events, s_ml_events = s2b.build_full_events_for_result(
        result=spy_cand["trial"],
        raw_by_sym=raw_by_sym,
        symbols=spy_cand["symbols"],
        trade_symbol=spy_cand["trade_symbol"],
        cross_symbol=spy_cand["cross_symbol"],
        include_cross=bool(spy_cand["include_cross"]),
    )
    q_events = q_no_events if args.mode == "no_ml" else q_ml_events
    s_events = s_no_events if args.mode == "no_ml" else s_ml_events
    if q_events.empty or s_events.empty:
        raise RuntimeError("One of the selected event sets is empty; cannot build dual-symbol portfolio.")

    q_eq = daily_equity_from_events(q_events, args.start_capital)
    s_eq = daily_equity_from_events(s_events, args.start_capital)
    q_trades = daily_trade_counts(q_events)
    s_trades = daily_trade_counts(s_events)

    idx = pd.date_range(
        min(q_eq.index.min(), s_eq.index.min()),
        max(q_eq.index.max(), s_eq.index.max()),
        freq="D",
        tz="UTC",
    )
    q_eq = q_eq.reindex(idx).ffill().fillna(args.start_capital)
    s_eq = s_eq.reindex(idx).ffill().fillna(args.start_capital)
    q_trades = q_trades.reindex(idx).fillna(0.0)
    s_trades = s_trades.reindex(idx).fillna(0.0)

    q_ret = q_eq.pct_change().fillna(0.0)
    s_ret = s_eq.pct_change().fillna(0.0)

    q_weight_static, split_i = build_static_weight_series(
        idx=idx,
        q_ret=q_ret,
        s_ret=s_ret,
        train_ratio=args.train_ratio,
        weight_step=args.weight_step,
        objective=args.objective,
    )
    q_weight_dynamic = build_walk_forward_weights(
        idx=idx,
        q_ret=q_ret,
        s_ret=s_ret,
        train_ratio=args.train_ratio,
        lookback_days=args.lookback_days,
        min_train_days=args.min_train_days,
        weight_step=args.weight_step,
        objective=args.objective,
        turnover_penalty=args.turnover_penalty,
        momentum_days=args.regime_momentum_days,
        vol_days=args.regime_vol_days,
        vol_penalty=args.regime_vol_penalty,
        max_tilt=args.regime_max_tilt,
        weight_smoothing=args.weight_smoothing,
    )
    r_static = q_weight_static * q_ret + (1.0 - q_weight_static) * s_ret
    r_dynamic = q_weight_dynamic * q_ret + (1.0 - q_weight_dynamic) * s_ret

    eval_slice = slice(split_i, len(idx))
    static_eval_score = score_returns(r_static.iloc[eval_slice], args.objective)
    dynamic_eval_score = score_returns(r_dynamic.iloc[eval_slice], args.objective)

    if dynamic_eval_score > static_eval_score:
        allocator_name = "walk_forward_regime"
        q_weight = q_weight_dynamic
        r_full = r_dynamic
    else:
        allocator_name = "static_train_optimized"
        q_weight = q_weight_static
        r_full = r_static

    spy_weight = 1.0 - q_weight
    eq_full = args.start_capital * (1.0 + r_full).cumprod()

    active_q = (q_weight >= args.min_active_weight).astype(float)
    active_s = (spy_weight >= args.min_active_weight).astype(float)
    total_trades = (q_trades * active_q + s_trades * active_s).astype(float)
    month_tab = monthly_table(eq_full, total_trades, args.start_capital)

    daily_pnl = eq_full.diff().fillna(0.0)
    best_day = daily_pnl.idxmax()
    worst_day = daily_pnl.idxmin()
    best_day_amt = float(daily_pnl.loc[best_day])
    worst_day_amt = float(daily_pnl.loc[worst_day])

    best_month_row = month_tab.loc[month_tab["pnl"].idxmax()]
    worst_month_row = month_tab.loc[month_tab["pnl"].idxmin()]
    avg_monthly_pnl = float(month_tab["pnl"].mean())
    median_monthly_pnl = float(month_tab["pnl"].median())
    avg_trades_month = float(month_tab["trades"].mean())
    max_trades_month = float(month_tab["trades"].max())

    perf = perf_from_equity_series(eq_full, args.start_capital)
    turnover = float(q_weight.diff().abs().fillna(0.0).sum())

    stats_text = "\n".join(
        [
            f"End equity: {as_money(float(perf['end_equity']))}",
            f"CAGR: {100.0 * float(perf['cagr']):.2f}%  |  Max DD: {100.0 * float(perf['max_drawdown']):.2f}%",
            f"Calmar: {float(perf['calmar']) if perf['calmar'] is not None else float('nan'):.2f}",
            f"Avg monthly PnL: {as_money(avg_monthly_pnl)}  |  Median: {as_money(median_monthly_pnl)}",
            f"Best month: {best_month_row['month_end'].strftime('%Y-%m')} {as_money(float(best_month_row['pnl']))}",
            f"Worst month: {worst_month_row['month_end'].strftime('%Y-%m')} {as_money(float(worst_month_row['pnl']))}",
            f"Best day: {best_day.strftime('%Y-%m-%d')} {as_money(best_day_amt)}",
            f"Worst day: {worst_day.strftime('%Y-%m-%d')} {as_money(worst_day_amt)}",
            f"Trades/month avg: {avg_trades_month:.1f} (max {max_trades_month:.0f})",
        ]
    )

    fig_path = os.path.join(args.out_dir, "dual_symbol_portfolio_curve.png")
    plot_portfolio(
        out_path=fig_path,
        eq=eq_full,
        weights=q_weight,
        monthly=month_tab,
        stats_text=stats_text,
        start_capital=args.start_capital,
    )

    month_weights = q_weight.groupby(q_weight.index.tz_localize(None).to_period("M")).first()
    summary = {
        "meta": {
            "script": "step2_dual_symbol_portfolio_test.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "start_capital": args.start_capital,
            "objective": args.objective,
            "train_ratio": args.train_ratio,
            "weight_step": args.weight_step,
            "lookback_days": args.lookback_days,
            "min_train_days": args.min_train_days,
            "turnover_penalty": args.turnover_penalty,
            "weight_smoothing": args.weight_smoothing,
            "regime_momentum_days": args.regime_momentum_days,
            "regime_vol_days": args.regime_vol_days,
            "regime_vol_penalty": args.regime_vol_penalty,
            "regime_max_tilt": args.regime_max_tilt,
            "min_test_trades": args.min_test_trades,
            "target_test_trades": args.target_test_trades,
            "candidate_trade_weight": args.candidate_trade_weight,
            "candidate_edge_weight": args.candidate_edge_weight,
            "candidate_dd_cap": args.candidate_dd_cap,
            "candidate_dd_penalty": args.candidate_dd_penalty,
            "candidate_min_net_bps": args.candidate_min_net_bps,
            "candidate_min_cagr": args.candidate_min_cagr,
            "selected_allocator": allocator_name,
            "note": "Selects best holdout allocator between static (train-optimized) and walk-forward regime allocation.",
        },
        "selected_candidates": {
            "qqq": {
                "scenario_name": qqq_cand["scenario_name"],
                "include_cross": qqq_cand["include_cross"],
                "filter_name": qqq_cand["trial"]["filter_name"],
                "trial_id": qqq_cand["trial"].get("trial"),
                "test": qqq_cand["trial"][args.mode]["test"],
            },
            "spy": {
                "scenario_name": spy_cand["scenario_name"],
                "include_cross": spy_cand["include_cross"],
                "filter_name": spy_cand["trial"]["filter_name"],
                "trial_id": spy_cand["trial"].get("trial"),
                "test": spy_cand["trial"][args.mode]["test"],
            },
        },
        "weights": {
            "qqq_mean": float(q_weight.mean()),
            "qqq_min": float(q_weight.min()),
            "qqq_max": float(q_weight.max()),
            "spy_mean": float(spy_weight.mean()),
            "total_turnover_abs_weight": turnover,
            "monthly_qqq_weights": [
                {"month": str(m), "qqq_weight": float(w)}
                for m, w in month_weights.items()
            ],
        },
        "allocator_comparison": {
            "holdout_split_index": int(split_i),
            "holdout_start_utc": str(idx[split_i]),
            "holdout_score_static": float(static_eval_score),
            "holdout_score_walk_forward": float(dynamic_eval_score),
            "chosen": allocator_name,
            "static_qqq_weight": float(q_weight_static.iloc[0]),
        },
        "performance": perf,
        "pnl_stats": {
            "avg_monthly_pnl": avg_monthly_pnl,
            "median_monthly_pnl": median_monthly_pnl,
            "best_month": {
                "month_end": str(best_month_row["month_end"]),
                "pnl": float(best_month_row["pnl"]),
                "ret": float(best_month_row["ret"]),
                "trades": float(best_month_row["trades"]),
            },
            "worst_month": {
                "month_end": str(worst_month_row["month_end"]),
                "pnl": float(worst_month_row["pnl"]),
                "ret": float(worst_month_row["ret"]),
                "trades": float(worst_month_row["trades"]),
            },
            "best_day": {
                "date_utc": str(best_day),
                "pnl": best_day_amt,
            },
            "worst_day": {
                "date_utc": str(worst_day),
                "pnl": worst_day_amt,
            },
        },
        "trades": {
            "avg_monthly_trades": avg_trades_month,
            "max_monthly_trades": max_trades_month,
            "qqq_total_trades": int(len(q_events)),
            "spy_total_trades": int(len(s_events)),
            "dual_counting_note": "Daily trades count a symbol only when its portfolio weight >= min_active_weight.",
        },
        "monthly_table": month_tab.to_dict(orient="records"),
        "outputs": {
            "portfolio_curve_plot": fig_path,
        },
    }
    summary = round_obj(summary, ndigits=6)
    out_json = os.path.join(args.out_dir, "dual_symbol_portfolio_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[PORTFOLIO] Wrote: {fig_path}")
    print(f"[PORTFOLIO] Wrote: {out_json}")


if __name__ == "__main__":
    main()
