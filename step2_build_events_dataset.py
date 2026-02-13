#!/usr/bin/env python3
"""
Step 2: Build an event-level ML dataset from cleaned IBKR RTH bars.

Inputs:
  - data_clean/{symbol}_1h_rth_clean.parquet (or whatever you produced)

Outputs:
  - out_dir/bar_features/{symbol}_bar_features.parquet
  - out_dir/events/{trade_symbol}_events.parquet
  - out_dir/meta/step2_config.json

Key fixes vs earlier draft:
  - Spread model is in bps (realistic for SPY/QQQ), not ATR% * huge constant
  - Triple barrier uses high/low touch detection (more realistic) but exits at next open
  - Optional cross-asset context (SPY features merged into QQQ events)
  - "Knobs" are centralized and explicit for later Step 2b tuning/backtests
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"


# -----------------------------
# Knobs (small set, explicit)
# -----------------------------
@dataclass
class Knobs:
    # Candidate generation (structure, frequency)
    trend_fast_span: int = 12
    trend_slow_span: int = 48
    trend_in: float = 0.60                 # threshold on trend score T_t
    pullback_z: float = 1.00               # require Z_pb < -pullback_z
    trend_regime_min: float = 0.25         # T_t must exceed this to allow pullback family
    min_event_spacing_bars: int = 1        # 1 = no spacing restriction

    # Volatility + ATR
    ewma_var_alpha: float = 0.06           # EWMA alpha on r^2 (hourly-ish)
    atr_span: int = 14

    # Triple barrier (labeling)
    horizon_intra: int = 12                # bars ahead (1h bars -> 12 trading hours)
    horizon_overn: int = 10                # slightly shorter for overnight entries
    tp_mult_intra: float = 2.0             # a_t = tp_mult * u_atr
    sl_mult_intra: float = 1.5             # b_t = sl_mult * u_atr
    tp_mult_overn: float = 2.2             # tighter or wider, your call
    sl_mult_overn: float = 1.6

    # Same-bar ambiguity policy: {"close_direction", "worst", "best"}
    same_bar_policy: str = "close_direction"

    # Friction model (realistic units)
    spread_half_bps: Dict[str, float] = None     # per symbol
    slip_base_bps: float = 0.8                  # baseline slippage half-side, in bps
    slip_vol_mult: float = 0.05                 # slippage += slip_vol_mult * sigma (sigma in log-return units)
    slip_spike_add_bps: float = 2.0             # add on vol spike bars
    vol_spike_q: float = 0.90                   # quantile threshold for spike
    overnight_slip_add_bps: float = 1.5         # extra slippage half-side for overnight entries

    # Commission modeled in bps (round trip) to avoid notional dependence in labels
    commission_round_trip_bps: float = 1.0

    # Gap feature window
    gap_lookback: int = 120

    # Cross-asset context
    include_cross_asset: bool = True
    cross_symbol: str = "SPY"

    # Output control
    drop_rows_with_any_nan_features: bool = True


def default_knobs() -> Knobs:
    k = Knobs()
    # Realistic half-spreads (very conservative for liquid ETFs)
    k.spread_half_bps = {
        "SPY": 0.8,
        "QQQ": 1.2,
        "SMH": 1.8,
    }
    return k


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_parquet_any(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Standardize column names
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {path}. Found: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in {path}. Found: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def compute_tr(df: pd.DataFrame) -> pd.Series:
    c_prev = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - c_prev).abs(), (df["low"] - c_prev).abs()))
    return tr


def session_date_ny(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(NY_TZ).dt.date


def compute_bar_features(df: pd.DataFrame, sym: str, knobs: Knobs) -> pd.DataFrame:
    out = df.copy()

    # Basic returns
    out["log_close"] = np.log(out["close"])
    out["r_cc"] = out["log_close"].diff()

    # EWMA volatility on r^2
    r2 = out["r_cc"].fillna(0.0) ** 2
    out["ewm_var"] = r2.ewm(alpha=knobs.ewma_var_alpha, adjust=False).mean()
    out["sigma"] = np.sqrt(out["ewm_var"])

    # ATR and normalized ATR
    out["tr"] = compute_tr(out)
    out["atr"] = out["tr"].ewm(span=knobs.atr_span, adjust=False).mean()
    out["u_atr"] = out["atr"] / out["close"]

    # Trend EMAs and trend score
    out["ema_fast"] = out["close"].ewm(span=knobs.trend_fast_span, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=knobs.trend_slow_span, adjust=False).mean()
    out["trend_raw"] = (out["ema_fast"] - out["ema_slow"]) / out["close"]
    out["trend_score"] = out["trend_raw"] / out["sigma"].replace(0.0, np.nan)

    # Pullback z-score around fast EMA (in log space)
    pb_raw = out["log_close"] - np.log(out["ema_fast"].replace(0.0, np.nan))
    pb_std = pb_raw.ewm(span=knobs.trend_slow_span, adjust=False).std(bias=False)
    out["pullback_z"] = pb_raw / pb_std.replace(0.0, np.nan)

    # Volume z (log volume)
    lv = np.log(out["volume"].replace(0.0, np.nan))
    lv_mu = lv.rolling(200, min_periods=50).mean()
    lv_sd = lv.rolling(200, min_periods=50).std()
    out["vol_z"] = (lv - lv_mu) / lv_sd.replace(0.0, np.nan)

    # Distance to rolling high (risk of chasing)
    roll_high = out["close"].rolling(100, min_periods=20).max().shift(1)
    out["dist_to_hi"] = (out["close"] - roll_high) / out["close"]

    # Overnight entry indicator for decision at t (entry happens at t+1)
    sdate = session_date_ny(out["date"])
    sdate_next = sdate.shift(-1)
    out["entry_overnight"] = (sdate_next.notna()) & (sdate_next != sdate)

    # Gap return for overnight transitions (decision at t, entry at t+1 open)
    out["gap_ret"] = np.where(
        out["entry_overnight"],
        np.log(out["open"].shift(-1) / out["close"]),
        np.nan,
    )

    # Rolling gap stats (computed over actual gap events only, then reindexed)
    # This ensures the window covers the last M *overnight* events, not M bars.
    gap_only = out.loc[out["gap_ret"].notna(), "gap_ret"]
    gap_mu_sparse = gap_only.rolling(knobs.gap_lookback, min_periods=20).mean()
    gap_sd_sparse = gap_only.rolling(knobs.gap_lookback, min_periods=20).std()
    out["gap_mu"] = gap_mu_sparse.reindex(out.index).ffill()
    out["gap_sd"] = gap_sd_sparse.reindex(out.index).ffill()
    out["gap_z"] = (out["gap_ret"] - out["gap_mu"]) / out["gap_sd"].replace(0.0, np.nan)

    # Vol spike flag for slippage model
    sig = out["sigma"].copy()
    spike_thr = sig.rolling(500, min_periods=200).quantile(knobs.vol_spike_q)
    out["vol_spike"] = (sig > spike_thr).fillna(False)

    # Symbol tag
    out["symbol"] = sym

    return out


def build_cross_features(
    trade_bars: pd.DataFrame,
    cross_bars: pd.DataFrame,
    trade_symbol: str,
    cross_symbol: str
) -> pd.DataFrame:
    """
    Merge cross_symbol bar features into trade_symbol bar features by UTC timestamp.
    Also add a few explicit cross relationships that often help.
    """
    # Columns to take from cross symbol (keep it small)
    take = [
        "date", "close", "r_cc", "sigma", "u_atr", "trend_score", "pullback_z", "vol_z", "dist_to_hi"
    ]
    cb = cross_bars[take].copy()
    cb = cb.rename(columns={c: f"{cross_symbol.lower()}_{c}" for c in cb.columns if c != "date"})

    merged = trade_bars.merge(cb, on="date", how="left")

    # Relative strength and spread return
    merged["rs_log"] = np.log(merged["close"] / merged[f"{cross_symbol.lower()}_close"])
    merged["ret_spread"] = merged["r_cc"] - merged[f"{cross_symbol.lower()}_r_cc"]

    # Beta proxy (rolling)
    x = merged[f"{cross_symbol.lower()}_r_cc"]
    y = merged["r_cc"]
    cov = (x * y).rolling(200, min_periods=80).mean() - x.rolling(200, min_periods=80).mean() * y.rolling(200, min_periods=80).mean()
    varx = x.rolling(200, min_periods=80).var()
    merged["beta_proxy"] = cov / varx.replace(0.0, np.nan)

    # Regime agreement (sign)
    merged["regime_agree"] = np.sign(merged["trend_score"]) == np.sign(merged[f"{cross_symbol.lower()}_trend_score"])

    return merged


def candidates_from_features(bars: pd.DataFrame, knobs: Knobs) -> pd.DataFrame:
    """
    Produce candidate event rows at decision time t.
    Events are defined at index t where entry is at t+1 open.
    """
    df = bars.copy()

    # Candidate families
    df["cand_trend_long"] = df["trend_score"] > knobs.trend_in

    df["trend_regime"] = df["trend_score"] > knobs.trend_regime_min
    df["cand_pullback_long"] = df["trend_regime"] & (df["pullback_z"] < -knobs.pullback_z)

    # Combine into one candidate mask
    df["is_candidate"] = df["cand_trend_long"] | df["cand_pullback_long"]

    # Label the family for later analysis
    df["family"] = np.where(df["cand_trend_long"], "trend_long",
                    np.where(df["cand_pullback_long"], "pullback_long", ""))

    # Only keep indices where t+1 exists (needs entry open)
    df["has_next"] = df["open"].shift(-1).notna()
    df = df[df["is_candidate"] & df["has_next"]].copy()

    # Optional spacing to reduce redundant overlapping events
    if knobs.min_event_spacing_bars > 1 and not df.empty:
        keep_idx = []
        last_kept_i = None
        for i in df.index.to_list():
            if last_kept_i is None or (i - last_kept_i) >= knobs.min_event_spacing_bars:
                keep_idx.append(i)
                last_kept_i = i
        df = df.loc[keep_idx].copy()

    return df


def friction_cost_return_units(row: pd.Series, sym: str, knobs: Knobs) -> Tuple[float, float, float]:
    """
    Return:
      spread_half (return units),
      slip_half (return units),
      total_round_trip_cost (return units)
    """
    spread_half = (knobs.spread_half_bps.get(sym, 1.5)) / 10000.0

    # slippage half-side
    slip_half = (knobs.slip_base_bps / 10000.0) + knobs.slip_vol_mult * float(row["sigma"] if pd.notna(row["sigma"]) else 0.0)
    if bool(row.get("vol_spike", False)):
        slip_half += knobs.slip_spike_add_bps / 10000.0
    if bool(row.get("entry_overnight", False)):
        slip_half += knobs.overnight_slip_add_bps / 10000.0

    comm_rt = knobs.commission_round_trip_bps / 10000.0

    # round trip: buy side + sell side
    total = 2.0 * (spread_half + slip_half) + comm_rt
    return spread_half, slip_half, total


def decide_same_bar(
    policy: str,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    tp_px: float,
    sl_px: float
) -> str:
    """
    Returns "tp" or "sl" when both are touched in the same bar.
    """
    if policy == "worst":
        return "sl"
    if policy == "best":
        return "tp"
    # close_direction (balanced)
    if close_px >= open_px:
        return "tp"
    return "sl"


def label_event_long(
    bars: pd.DataFrame,
    t_idx: int,
    sym: str,
    knobs: Knobs
) -> Optional[Dict]:
    """
    Label a long event decided at t_idx with entry at t_idx+1 open.
    Barrier touch uses High/Low; exit uses next open after touch or horizon.
    """
    n = len(bars)
    if t_idx + 2 >= n:
        return None

    row_t = bars.iloc[t_idx]
    entry_i = t_idx + 1
    entry_open = float(bars.iloc[entry_i]["open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None

    # Choose intra vs overnight params based on entry_overnight at decision time
    overnight = bool(row_t.get("entry_overnight", False))
    H = knobs.horizon_overn if overnight else knobs.horizon_intra

    u = float(row_t["u_atr"]) if pd.notna(row_t["u_atr"]) else np.nan
    if not np.isfinite(u) or u <= 0:
        return None

    tp_mult = knobs.tp_mult_overn if overnight else knobs.tp_mult_intra
    sl_mult = knobs.sl_mult_overn if overnight else knobs.sl_mult_intra

    a = tp_mult * u
    b = sl_mult * u

    # Convert barriers to price levels (multiplicative)
    tp_px = entry_open * np.exp(a)
    sl_px = entry_open * np.exp(-b)

    # Friction model at entry decision time (conservative and calibratable)
    spread_half, slip_half, cost_rt = friction_cost_return_units(row_t, sym, knobs)

    # Find first barrier touch between bars [entry_i .. entry_i+H-1]
    touch_i = None
    touch_side = None

    end_i = min(entry_i + H, n - 2)  # need i+1 open for exit, so cap at n-2
    for i in range(entry_i, end_i + 1):
        r = bars.iloc[i]
        o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        hit_tp = (h >= tp_px)
        hit_sl = (l <= sl_px)

        if hit_tp and hit_sl:
            touch_side = decide_same_bar(knobs.same_bar_policy, o, h, l, c, tp_px, sl_px)
            touch_i = i
            break
        if hit_tp:
            touch_side = "tp"
            touch_i = i
            break
        if hit_sl:
            touch_side = "sl"
            touch_i = i
            break

    if touch_i is None:
        # horizon exit at open after the last monitored bar
        exit_i = min(entry_i + H, n - 2) + 1
        exit_reason = "horizon"
    else:
        exit_i = min(touch_i + 1, n - 1)
        exit_reason = "tp" if touch_side == "tp" else "sl"

    exit_open = float(bars.iloc[exit_i]["open"])
    if not np.isfinite(exit_open) or exit_open <= 0:
        return None

    gross = np.log(exit_open / entry_open)
    net = gross - cost_rt

    y = 1 if net > 0 else 0

    return {
        "symbol": sym,
        "t_idx": int(t_idx),
        "decision_time_utc": str(row_t["date"]),
        "entry_time_utc": str(bars.iloc[entry_i]["date"]),
        "exit_time_utc": str(bars.iloc[exit_i]["date"]),
        "entry_open": entry_open,
        "exit_open": exit_open,
        "overnight": int(overnight),
        "a_tp": float(a),
        "b_sl": float(b),
        "H": int(H),
        "exit_reason": exit_reason,
        "gross_logret": float(gross),
        "cost_rt": float(cost_rt),
        "net_logret": float(net),
        "y": int(y),
        # for purge/embargo later
        "label_end_idx": int(exit_i),
        # friction components (for analysis)
        "spread_half": float(spread_half),
        "slip_half": float(slip_half),
    }


def build_event_dataset(bars: pd.DataFrame, sym: str, knobs: Knobs) -> pd.DataFrame:
    cands = candidates_from_features(bars, knobs)
    if cands.empty:
        return pd.DataFrame()

    labels = []
    for t_idx in cands.index.to_list():
        lab = label_event_long(bars, t_idx, sym, knobs)
        if lab is None:
            continue

        # attach candidate family and a small subset of decision-time features
        row_t = bars.iloc[t_idx]
        lab["family"] = str(cands.loc[t_idx, "family"])

        # Decision-time features to feed ML
        feat_cols = [
            "trend_score", "pullback_z", "sigma", "u_atr", "vol_z", "dist_to_hi",
            "gap_mu", "gap_sd"
        ]
        for c in feat_cols:
            lab[c] = float(row_t[c]) if pd.notna(row_t[c]) else np.nan
        # entry_overnight is boolean — convert explicitly to int
        lab["entry_overnight"] = int(bool(row_t["entry_overnight"]))

        labels.append(lab)

    ev = pd.DataFrame(labels)
    return ev


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing cleaned parquet files")
    ap.add_argument("--out-dir", required=True, help="Where to write datasets")
    ap.add_argument("--symbols", nargs="+", required=True, help="Symbols to load (must include trade symbol and cross if used)")
    ap.add_argument("--trade-symbol", default="QQQ", help="Symbol to build events for (default QQQ)")
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet", help="Input file suffix pattern")
    ap.add_argument("--no-cross", action="store_true", help="Disable cross-asset context merge")
    args = ap.parse_args()

    knobs = default_knobs()
    if args.no_cross:
        knobs.include_cross_asset = False

    out_root = args.out_dir
    out_bar = os.path.join(out_root, "bar_features")
    out_evt = os.path.join(out_root, "events")
    out_meta = os.path.join(out_root, "meta")
    ensure_dir(out_bar)
    ensure_dir(out_evt)
    ensure_dir(out_meta)

    # Load and compute bar features for all symbols
    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        in_path = os.path.join(args.data_dir, f"{sym.lower()}{args.bar_file_suffix}")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing input parquet: {in_path}")

        print(f"[{sym}] loading {in_path}")
        raw = read_parquet_any(in_path)
        feat = compute_bar_features(raw, sym.upper(), knobs)
        bars_by_sym[sym.upper()] = feat

        out_path = os.path.join(out_bar, f"{sym.lower()}_bar_features.parquet")
        feat.to_parquet(out_path, index=False)
        print(f"[{sym}] wrote bar features: {out_path}")

    trade_sym = args.trade_symbol.upper()
    if trade_sym not in bars_by_sym:
        raise ValueError(f"trade_symbol {trade_sym} not in --symbols")

    trade_bars = bars_by_sym[trade_sym].copy()

    # Optional cross-asset merge at bar level, then events will inherit cross features later if desired
    if knobs.include_cross_asset:
        cross_sym = knobs.cross_symbol.upper()
        if cross_sym not in bars_by_sym:
            raise ValueError(f"include_cross_asset=True but cross_symbol={cross_sym} not in --symbols")
        print(f"[CROSS] merging {cross_sym} into {trade_sym} bar features")
        trade_bars = build_cross_features(trade_bars, bars_by_sym[cross_sym], trade_sym, cross_sym)

    # Build events dataset
    print(f"[EVENTS] building events for {trade_sym}")
    events = build_event_dataset(trade_bars, trade_sym, knobs)

    if events.empty:
        print("[EVENTS] no events produced. Consider loosening thresholds or checking data.")
        return

    # If cross features exist at bar level, attach a controlled subset at decision-time
    if knobs.include_cross_asset:
        cross = knobs.cross_symbol.lower()
        cross_cols = [
            f"{cross}_sigma",
            f"{cross}_u_atr",
            f"{cross}_trend_score",
            f"{cross}_pullback_z",
            f"{cross}_vol_z",
            "rs_log",
            "ret_spread",
            "beta_proxy",
            "regime_agree",
        ]
        # Reconstruct decision-time lookup by t_idx
        tb = trade_bars.reset_index(drop=True)
        for c in cross_cols:
            if c in tb.columns:
                events[c] = events["t_idx"].map(lambda i, c=c: tb.loc[int(i), c] if int(i) < len(tb) else np.nan)

    # Feature cleaning
    if knobs.drop_rows_with_any_nan_features:
        # Keep core columns safe
        non_feature_cols = {
            "symbol", "t_idx", "decision_time_utc", "entry_time_utc", "exit_time_utc",
            "entry_open", "exit_open", "overnight", "a_tp", "b_sl", "H",
            "exit_reason", "gross_logret", "cost_rt", "net_logret", "y", "label_end_idx",
            "family", "spread_half", "slip_half"
        }
        feat_cols = [c for c in events.columns if c not in non_feature_cols]
        before = len(events)
        events = events.dropna(subset=feat_cols).reset_index(drop=True)
        after = len(events)
        print(f"[CLEAN] dropped rows with NaN features: {before - after} (kept {after})")

    out_events_path = os.path.join(out_evt, f"{trade_sym.lower()}_events.parquet")
    events.to_parquet(out_events_path, index=False)
    print(f"[EVENTS] wrote: {out_events_path}")

    # Save config used
    meta = {
        "trade_symbol": trade_sym,
        "symbols_loaded": [s.upper() for s in args.symbols],
        "knobs": asdict(knobs),
        "notes": {
            "barrier_touch": "High/Low touch detection; exit at next open; same-bar policy configurable",
            "spread_model": "half-spread in bps, per symbol (configurable)",
            "cross_asset": "Optional SPY context merged into QQQ bar features and attached to events",
        },
    }
    out_meta_path = os.path.join(out_meta, "step2_config.json")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] wrote: {out_meta_path}")

    print("\nDone.")
    print(f"Events produced: {len(events)}")
    print("Next: Step 2.5 will analyze event stats (family breakdown, net returns, cost sensitivity, regime slices).")


if __name__ == "__main__":
    main()
