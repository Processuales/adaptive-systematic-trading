#!/usr/bin/env python3
"""
Step 3 dataset builder.

Builds independent Step 3 training datasets for QQQ and SPY from cleaned bars.
Outputs are written under step3_out only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

import step2_build_events_dataset as s2

SCRIPT_VERSION = "1.0.0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    return obj


def enforce_non_overlap(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    df = events.sort_values(["t_idx", "label_end_idx"]).reset_index(drop=True)
    t_arr = df["t_idx"].to_numpy(dtype=np.int64, copy=False)
    end_arr = df["label_end_idx"].to_numpy(dtype=np.int64, copy=False)
    keep: List[int] = []
    last_end = -10**9
    for i, (t, end) in enumerate(zip(t_arr, end_arr)):
        if t > last_end:
            keep.append(i)
            last_end = int(end)
    return df.loc[keep].reset_index(drop=True)


def build_symbol_events(
    bars_by_sym: Dict[str, pd.DataFrame],
    trade_symbol: str,
    cross_symbol: str,
    knobs: s2.Knobs,
) -> pd.DataFrame:
    tol = pd.Timedelta(knobs.cross_merge_tolerance)
    trade_bars = bars_by_sym[trade_symbol].copy()

    if knobs.include_cross_asset:
        trade_bars = s2.build_cross_features(
            trade_bars=trade_bars,
            cross_bars=bars_by_sym[cross_symbol],
            cross_symbol=cross_symbol,
            tolerance=tol,
        )

    events = s2.build_event_dataset(trade_bars, trade_symbol, knobs)
    if events.empty:
        return events

    if knobs.include_cross_asset:
        cross_sym_l = cross_symbol.lower()
        desired = [
            f"{cross_sym_l}_sigma",
            f"{cross_sym_l}_u_atr",
            f"{cross_sym_l}_trend_score",
            f"{cross_sym_l}_pullback_z",
            f"{cross_sym_l}_vol_z",
            f"{cross_sym_l}_range_ratio",
            f"{cross_sym_l}_ema_fast_slope",
            f"{cross_sym_l}_age_min",
            "rs_log",
            "ret_spread",
            "beta_proxy",
            "regime_agree",
        ]
        tb = trade_bars.reset_index(drop=True)
        events = s2.map_features_from_t_idx(events, tb, desired)

    core = ["trend_score", "pullback_z", "sigma", "u_atr"]
    events = events.dropna(subset=core).copy()
    for c in ["decision_time_utc", "entry_time_utc", "exit_time_utc"]:
        events[c] = pd.to_datetime(events[c], utc=True, errors="coerce")
    events = events.dropna(subset=["decision_time_utc", "entry_time_utc", "exit_time_utc"]).copy()
    events = events.sort_values(["decision_time_utc", "t_idx"]).reset_index(drop=True)
    events = enforce_non_overlap(events)

    events["trade_symbol"] = trade_symbol
    events["cross_symbol"] = cross_symbol
    events["y_loss"] = 1 - events["y"].astype(int)
    events["hold_bars"] = (events["label_end_idx"] - (events["t_idx"] + 1)).clip(lower=0)
    events["decision_month"] = events["decision_time_utc"].dt.strftime("%Y-%m")
    return events.reset_index(drop=True)


def infer_feature_columns(events_all: pd.DataFrame) -> List[str]:
    numeric_cols = events_all.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "y",
        "y_loss",
        "gross_logret",
        "net_logret",
        "entry_open",
        "exit_open",
        "t_idx",
        "label_end_idx",
    }
    feats = [c for c in numeric_cols if c not in exclude]
    return sorted(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with cleaned 1h parquet bars")
    ap.add_argument("--out-dir", required=True, help="Root output dir (e.g. step3_out)")
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--cross-tolerance", default="30min")
    ap.add_argument("--same-bar-policy", choices=["worst", "best", "close_direction"], default="worst")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_dir)
    dataset_dir = os.path.join(out_root, "dataset")
    ensure_dir(dataset_dir)

    symbols = [s.upper() for s in args.symbols]
    if set(symbols) != {"SPY", "QQQ"}:
        raise ValueError("--symbols currently supports exactly SPY and QQQ for Step 3.")

    knobs = s2.default_knobs()
    knobs.include_cross_asset = True
    knobs.cross_merge_tolerance = args.cross_tolerance
    knobs.same_bar_policy = args.same_bar_policy

    raw_by_sym: Dict[str, pd.DataFrame] = {}
    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p = os.path.join(args.data_dir, f"{sym.lower()}{args.bar_file_suffix}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cleaned bars: {p}")
        raw = s2.read_parquet_any(p)
        raw_by_sym[sym] = raw
        bars_by_sym[sym] = s2.compute_bar_features(raw, sym, knobs)

    q_events = build_symbol_events(
        bars_by_sym=bars_by_sym,
        trade_symbol="QQQ",
        cross_symbol="SPY",
        knobs=knobs,
    )
    s_events = build_symbol_events(
        bars_by_sym=bars_by_sym,
        trade_symbol="SPY",
        cross_symbol="QQQ",
        knobs=knobs,
    )
    if q_events.empty or s_events.empty:
        raise RuntimeError("Step 3 dataset build produced empty events for at least one symbol.")

    q_path = os.path.join(dataset_dir, "qqq_events_step3.parquet")
    s_path = os.path.join(dataset_dir, "spy_events_step3.parquet")
    q_events.to_parquet(q_path, index=False)
    s_events.to_parquet(s_path, index=False)

    all_events = pd.concat([q_events, s_events], ignore_index=True)
    all_events = all_events.sort_values(["decision_time_utc", "trade_symbol"]).reset_index(drop=True)
    all_path = os.path.join(dataset_dir, "training_events_all.parquet")
    all_events.to_parquet(all_path, index=False)

    feature_cols = infer_feature_columns(all_events)
    meta = {
        "meta": {
            "script": "step3_build_training_dataset.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "bar_file_suffix": args.bar_file_suffix,
            "cross_tolerance": args.cross_tolerance,
            "same_bar_policy": args.same_bar_policy,
            "note": "Step 3 dataset is independent from step2_out and writes only under step3_out.",
        },
        "counts": {
            "qqq_events": int(len(q_events)),
            "spy_events": int(len(s_events)),
            "all_events": int(len(all_events)),
        },
        "time_ranges": {
            "qqq_start_utc": str(q_events["decision_time_utc"].min()),
            "qqq_end_utc": str(q_events["decision_time_utc"].max()),
            "spy_start_utc": str(s_events["decision_time_utc"].min()),
            "spy_end_utc": str(s_events["decision_time_utc"].max()),
        },
        "feature_columns": feature_cols,
        "outputs": {
            "qqq_events_path": q_path,
            "spy_events_path": s_path,
            "all_events_path": all_path,
        },
    }
    meta = round_obj(meta, 6)
    meta_path = os.path.join(dataset_dir, "step3_dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[STEP3-DATA] Wrote: {q_path}")
    print(f"[STEP3-DATA] Wrote: {s_path}")
    print(f"[STEP3-DATA] Wrote: {all_path}")
    print(f"[STEP3-DATA] Wrote: {meta_path}")


if __name__ == "__main__":
    main()
