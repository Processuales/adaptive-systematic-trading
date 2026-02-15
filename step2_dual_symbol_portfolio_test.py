#!/usr/bin/env python3
"""
Dual-symbol (SPY+QQQ) portfolio test using best Step 2 candidates.

Creates a single optimized portfolio equity line from historical data,
starting at a specified capital.

Outputs:
  - <out-dir>/dual_symbol_portfolio_curve.png
  - <out-dir>/dual_symbol_portfolio_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_build_events_dataset as s2
import step2b_knob_sweep_backtest as s2b

SCRIPT_VERSION = "1.0.0"


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
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return round(obj, ndigits)
    return obj


def pick_best_per_symbol(selection_dir: str, mode: str) -> Tuple[Dict, Dict]:
    """
    Returns (qqq_candidate, spy_candidate) where each candidate is a dict:
      {scenario_name, include_cross, trade_symbol, cross_symbol, symbols, trial}
    """
    scenarios = {
        "qqq_cross_off": ("QQQ", False, "SPY"),
        "qqq_cross_on": ("QQQ", True, "SPY"),
        "spy_cross_off": ("SPY", False, "QQQ"),
        "spy_cross_on": ("SPY", True, "QQQ"),
    }

    picks: Dict[str, Dict] = {}
    for name, (trade_sym, include_cross, cross_sym) in scenarios.items():
        p = os.path.join(selection_dir, name, "step2b_summary.json")
        if not os.path.exists(p):
            continue
        s = load_json(p)
        if mode == "no_ml":
            trials = s.get("top_trials_no_ml", [])
            if not trials:
                continue
            trial = trials[0]
            score = float(trial["no_ml"]["test"]["end_equity"])
        else:
            trials = s.get("top_trials_ml", [])
            if not trials:
                continue
            trial = trials[0]
            score = float(trial["ml_sim"]["test"]["end_equity"])

        cur = picks.get(trade_sym)
        cand = {
            "scenario_name": name,
            "include_cross": include_cross,
            "trade_symbol": trade_sym,
            "cross_symbol": cross_sym,
            "symbols": ["SPY", "QQQ"],
            "trial": trial,
            "score": score,
        }
        if cur is None or score > cur["score"]:
            picks[trade_sym] = cand

    if "QQQ" not in picks or "SPY" not in picks:
        raise RuntimeError(
            "Could not find both QQQ and SPY candidates in selection directory. "
            "Run step2_compare_and_select.py first."
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


def daily_equity(curve: pd.DataFrame, start_capital: float) -> pd.Series:
    if curve.empty:
        return pd.Series(dtype=float)
    x = curve.sort_values("time").copy()
    x["time"] = pd.to_datetime(x["time"], utc=True)
    # Rebase to start_capital
    first = float(x["equity"].iloc[0])
    if first > 0:
        x["equity"] = start_capital * (x["equity"] / first)
    s = x.set_index("time")["equity"].resample("D").last()
    return s


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


def objective_score(eq: pd.Series, start_capital: float, objective: str) -> float:
    p = perf_from_equity_series(eq, start_capital)
    if objective == "end_equity":
        return float(p["end_equity"])
    return float(p["calmar"] if p["calmar"] is not None else -1e18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to cleaned bars")
    ap.add_argument("--selection-dir", required=True, help="Path to step2_out/selection")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["no_ml", "ml_sim"], default="no_ml")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet")
    ap.add_argument("--weight-step", type=float, default=0.05, help="Grid step for QQQ weight search")
    ap.add_argument("--train-ratio", type=float, default=0.60, help="Fraction of history for weight optimization")
    ap.add_argument("--objective", choices=["calmar", "end_equity"], default="calmar")
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 < args.weight_step <= 1.0):
        raise ValueError("--weight-step must be in (0, 1]")

    ensure_dir(args.out_dir)

    qqq_cand, spy_cand = pick_best_per_symbol(args.selection_dir, args.mode)

    symbols = {"SPY", "QQQ"}
    raw_by_sym = load_raw_data(args.data_dir, symbols, args.bar_file_suffix)

    q_no, q_ml = s2b.build_full_equity_for_result(
        result=qqq_cand["trial"],
        raw_by_sym=raw_by_sym,
        symbols=qqq_cand["symbols"],
        trade_symbol=qqq_cand["trade_symbol"],
        cross_symbol=qqq_cand["cross_symbol"],
        include_cross=bool(qqq_cand["include_cross"]),
    )
    s_no, s_ml = s2b.build_full_equity_for_result(
        result=spy_cand["trial"],
        raw_by_sym=raw_by_sym,
        symbols=spy_cand["symbols"],
        trade_symbol=spy_cand["trade_symbol"],
        cross_symbol=spy_cand["cross_symbol"],
        include_cross=bool(spy_cand["include_cross"]),
    )
    q_curve = q_no if args.mode == "no_ml" else q_ml
    s_curve = s_no if args.mode == "no_ml" else s_ml
    if q_curve.empty or s_curve.empty:
        raise RuntimeError("One of the selected curves is empty; cannot build dual-symbol portfolio.")

    q_eq = daily_equity(q_curve, args.start_capital)
    s_eq = daily_equity(s_curve, args.start_capital)

    idx = pd.date_range(
        min(q_eq.index.min(), s_eq.index.min()),
        max(q_eq.index.max(), s_eq.index.max()),
        freq="D",
        tz="UTC",
    )
    q_eq = q_eq.reindex(idx).ffill().fillna(args.start_capital)
    s_eq = s_eq.reindex(idx).ffill().fillna(args.start_capital)

    q_ret = q_eq.pct_change().fillna(0.0)
    s_ret = s_eq.pct_change().fillna(0.0)

    split_i = int(len(idx) * args.train_ratio)
    split_i = min(max(split_i, 2), len(idx) - 1)
    train_slice = slice(0, split_i)

    weights = np.arange(0.0, 1.0 + 1e-12, args.weight_step)
    best_w = 0.5
    best_score = -1e18
    for w in weights:
        r_train = w * q_ret.iloc[train_slice] + (1.0 - w) * s_ret.iloc[train_slice]
        eq_train = args.start_capital * (1.0 + r_train).cumprod()
        score = objective_score(eq_train, args.start_capital, args.objective)
        if score > best_score:
            best_score = score
            best_w = float(w)

    r_full = best_w * q_ret + (1.0 - best_w) * s_ret
    eq_full = args.start_capital * (1.0 + r_full).cumprod()

    fig_path = os.path.join(args.out_dir, "dual_symbol_portfolio_curve.png")
    plt.figure(figsize=(12, 6))
    plt.plot(eq_full.index, eq_full.values, color="tab:blue", linewidth=2.0)
    plt.axhline(args.start_capital, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Optimized SPY+QQQ Portfolio Curve")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

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
            "note": "Uses Step 2 net returns (fees/slippage modeled).",
        },
        "selected_candidates": {
            "qqq": {
                "scenario_name": qqq_cand["scenario_name"],
                "include_cross": qqq_cand["include_cross"],
                "filter_name": qqq_cand["trial"]["filter_name"],
            },
            "spy": {
                "scenario_name": spy_cand["scenario_name"],
                "include_cross": spy_cand["include_cross"],
                "filter_name": spy_cand["trial"]["filter_name"],
            },
        },
        "optimized_weights": {
            "qqq_weight": best_w,
            "spy_weight": 1.0 - best_w,
        },
        "performance": perf_from_equity_series(eq_full, args.start_capital),
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
