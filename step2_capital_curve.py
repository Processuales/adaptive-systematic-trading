#!/usr/bin/env python3
"""
Build a "fun" capital curve chart from saved best candidates.

This uses net_logret from Step 2 events, which already includes modeled
fees/slippage/costs from Step 2 labeling.

Outputs:
  - <out-dir>/capital_curve.png
  - <out-dir>/capital_curve_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_build_events_dataset as s2
import step2b_knob_sweep_backtest as s2b

SCRIPT_VERSION = "1.0.0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_candidate(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_data(data_dir: str, symbols: Set[str], suffix: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in sorted(symbols):
        p = os.path.join(data_dir, f"{sym.lower()}{suffix}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cleaned data file: {p}")
        out[sym] = s2.read_parquet_any(p)
    return out


def perf_from_curve(curve: pd.DataFrame, start_capital: float) -> Dict:
    if curve.empty:
        return {
            "n_points": 0,
            "end_equity": start_capital,
            "cagr": None,
            "max_drawdown": None,
            "calmar": None,
        }
    x = curve.sort_values("time").copy()
    eq = x["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    mdd = float(dd.max())

    t0 = pd.to_datetime(x["time"].iloc[0], utc=True)
    t1 = pd.to_datetime(x["time"].iloc[-1], utc=True)
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (float(eq[-1]) / float(start_capital)) ** (1.0 / years) - 1.0
    calmar = float(cagr / mdd) if mdd > 0 else None

    return {
        "n_points": int(len(x)),
        "start_time_utc": str(t0),
        "end_time_utc": str(t1),
        "end_equity": float(eq[-1]),
        "cagr": float(cagr),
        "max_drawdown": mdd,
        "calmar": calmar,
    }


def rebase_curve(curve: pd.DataFrame, start_capital: float) -> pd.DataFrame:
    if curve.empty:
        return curve.copy()
    x = curve.sort_values("time").copy()
    first = float(x["equity"].iloc[0])
    if first <= 0:
        return x
    x["equity"] = start_capital * (x["equity"] / first)
    return x


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with cleaned data files")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--candidate-non-ml",
        required=True,
        help="Path to best_candidate_non_ml.json",
    )
    ap.add_argument(
        "--candidate-ml",
        default=None,
        help="Path to best_candidate_ml.json",
    )
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    cand_no = load_candidate(args.candidate_non_ml)
    cand_ml = load_candidate(args.candidate_ml) if args.candidate_ml else None

    symbols: Set[str] = set(cand_no["scenario"]["symbols"])
    if cand_ml is not None:
        symbols |= set(cand_ml["scenario"]["symbols"])
    raw_by_sym = load_raw_data(args.data_dir, symbols, args.bar_file_suffix)

    no_ml_curve_raw, _ignore_ml_from_no = s2b.build_full_equity_for_result(
        result=cand_no["trial"],
        raw_by_sym=raw_by_sym,
        symbols=cand_no["scenario"]["symbols"],
        trade_symbol=cand_no["scenario"]["trade_symbol"],
        cross_symbol=cand_no["scenario"]["cross_symbol"],
        include_cross=bool(cand_no["scenario"]["include_cross"]),
    )
    no_ml_curve = rebase_curve(no_ml_curve_raw, args.start_capital)

    ml_curve = pd.DataFrame(columns=["time", "equity"])
    if cand_ml is not None:
        _ignore_no, ml_curve_raw = s2b.build_full_equity_for_result(
            result=cand_ml["trial"],
            raw_by_sym=raw_by_sym,
            symbols=cand_ml["scenario"]["symbols"],
            trade_symbol=cand_ml["scenario"]["trade_symbol"],
            cross_symbol=cand_ml["scenario"]["cross_symbol"],
            include_cross=bool(cand_ml["scenario"]["include_cross"]),
        )
        ml_curve = rebase_curve(ml_curve_raw, args.start_capital)

    fig_path = os.path.join(args.out_dir, "capital_curve.png")
    plt.figure(figsize=(12, 6))
    if not no_ml_curve.empty:
        plt.plot(
            pd.to_datetime(no_ml_curve["time"], utc=True),
            no_ml_curve["equity"],
            color="red",
            linewidth=1.9,
            label="Best Candidate (No ML)",
        )
    if not ml_curve.empty:
        plt.plot(
            pd.to_datetime(ml_curve["time"], utc=True),
            ml_curve["equity"],
            color="green",
            linewidth=1.9,
            label="Best Candidate (ML Sim)",
        )
    plt.axhline(args.start_capital, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Bot Capital Curve (Costs Included, Historical Backtest)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    summary = {
        "meta": {
            "script": "step2_capital_curve.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "start_capital": args.start_capital,
            "note": "Uses Step 2 net_logret, which already includes modeled costs/slippage.",
        },
        "non_ml_candidate": {
            "scenario": cand_no["scenario"],
            "test_snapshot": cand_no.get("test_snapshot"),
            "full_period_perf": perf_from_curve(no_ml_curve, args.start_capital),
        },
        "ml_candidate": (
            {
                "scenario": cand_ml["scenario"],
                "test_snapshot": cand_ml.get("test_snapshot"),
                "full_period_perf": perf_from_curve(ml_curve, args.start_capital),
            }
            if cand_ml is not None
            else None
        ),
        "outputs": {
            "capital_curve_plot": fig_path,
        },
    }
    summary = round_obj(summary, ndigits=6)
    out_json = os.path.join(args.out_dir, "capital_curve_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[CAPITAL] Wrote: {fig_path}")
    print(f"[CAPITAL] Wrote: {out_json}")


if __name__ == "__main__":
    main()
