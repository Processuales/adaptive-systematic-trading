#!/usr/bin/env python3
"""
step2_build_events_dataset.py

Step 2: Build an event-level ML dataset from cleaned IBKR RTH bars.

Inputs (example):
  data_clean/spy_1h_rth_clean.parquet
  data_clean/qqq_1h_rth_clean.parquet

Outputs:
  out_dir/bar_features/{symbol}_bar_features.parquet
  out_dir/events/{trade_symbol}_events.parquet
  out_dir/meta/step2_config.json

Design:
  - Candidate generation at bar-close (decision index t)
  - Entry at next bar open (t+1), consistent with realistic execution latency
  - Triple-barrier labeling uses High/Low to detect barrier touches, but exits at next open
    (keeps gap risk realistic; avoids optimistic intrabar fills)
  - Overnight-aware policy: separate horizon and barrier widths when entry crosses session boundary
  - Friction model in return units (bps proxies + sigma), intended to be calibrated later with IBKR paper fills
  - Optional cross-asset context: merge SPY bar features into QQQ with merge_asof tolerance
  - Production hygiene:
      * Fix NaN-diluted gap rolling stats by computing on gap events only
      * Prevent gap leakage by shifting rolling gap stats by 1 gap event
      * Avoid cross-feature lookahead via merge_asof(direction="backward")
      * Explicit boolean handling for entry_overnight
      * Fix lambda closure bug when mapping cross columns onto events
      * Avoid overly aggressive dropna: only enforce a small core feature set

Recommended defaults:
  - same_bar_policy="worst" for conservative labeling baseline
  - merge_asof tolerance 30 minutes to avoid timestamp micro-mismatches
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
SCRIPT_VERSION = "2.1.0"


# -----------------------------
# Knobs (small set, explicit)
# -----------------------------
@dataclass
class Knobs:
    # Candidate generation (structure, frequency)
    trend_fast_span: int = 12
    trend_slow_span: int = 48
    trend_in: float = 0.60                 # threshold on trend_score
    pullback_z: float = 1.00               # require pullback_z < -pullback_z
    trend_regime_min: float = 0.25         # trend_score must exceed this for pullback family
    min_event_spacing_bars: int = 1        # 1 = no spacing restriction

    # Volatility + ATR
    ewma_var_alpha: float = 0.06           # EWMA alpha on r^2
    atr_span: int = 14

    # Triple barrier (labeling)
    horizon_intra: int = 12                # bars ahead
    horizon_overn: int = 10                # slightly shorter for overnight entries
    tp_mult_intra: float = 2.0             # a_t = tp_mult * u_atr
    sl_mult_intra: float = 1.5             # b_t = sl_mult * u_atr
    tp_mult_overn: float = 2.2
    sl_mult_overn: float = 1.6

    # Same-bar ambiguity policy: {"worst", "best", "close_direction"}
    same_bar_policy: str = "worst"

    # Friction model (realistic units)
    spread_half_bps: Dict[str, float] = None     # per symbol
    slip_base_bps: float = 0.8                  # baseline slippage half-side in bps
    slip_vol_mult: float = 0.05                 # slippage += slip_vol_mult * sigma (sigma in log-return units)
    slip_spike_add_bps: float = 2.0             # add on vol spike bars (bps)
    vol_spike_q: float = 0.90                   # quantile threshold for spike
    overnight_slip_add_bps: float = 1.5         # extra slippage half-side for overnight entries (bps)

    # Commission modeled as round-trip bps
    commission_round_trip_bps: float = 1.0

    # Gap features: window defined in number of overnight events (not bars)
    gap_lookback_events: int = 120
    gap_tail_k: float = 2.0                     # tail threshold multiple: |gap| > k * gap_sd

    # Rolling percentile / tail context features
    sigma_rank_window: int = 500
    atr_rank_window: int = 500
    rank_min_periods: int = 200
    intraday_tail_window: int = 200
    intraday_tail_min_periods: int = 50
    intraday_tail_z: float = 2.0

    # Cross-asset context
    include_cross_asset: bool = True
    cross_symbol: str = "SPY"
    cross_merge_tolerance: str = "30min"        # pd.Timedelta string

    # Output control
    dropna_core_only: bool = True               # recommended (do not drop on all optional features)


def default_knobs() -> Knobs:
    k = Knobs()
    # Conservative half-spread assumptions (half-spread per side)
    k.spread_half_bps = {
        "SPY": 0.8,
        "QQQ": 1.2,
        "SMH": 1.8,
    }
    return k


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_parquet_any(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Found: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    # basic sanity
    if (df["close"] <= 0).any() or (df["open"] <= 0).any():
        raise ValueError(f"{path}: non-positive prices detected after cleaning.")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{path}: high < low detected after cleaning.")

    return df


# -----------------------------
# Feature engineering
# -----------------------------
def compute_tr(df: pd.DataFrame) -> pd.Series:
    c_prev = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - c_prev).abs(), (df["low"] - c_prev).abs()),
    )
    return tr


def session_date_ny(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(NY_TZ).dt.date


def rolling_percentile_last(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Rolling percentile rank of the current value within its trailing window.
    Returns values in [0, 1].
    """
    # pandas rolling.rank is vectorized in C and much faster than Python apply.
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def map_features_from_t_idx(events: pd.DataFrame, source: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if events.empty or not cols:
        return events
    if "t_idx" not in events.columns:
        return events

    out = events.copy()
    t_idx = out["t_idx"].to_numpy(dtype=np.int64, copy=False)
    valid = (t_idx >= 0) & (t_idx < len(source))

    for c in cols:
        if c not in source.columns:
            continue
        src = source[c].to_numpy()
        if np.issubdtype(src.dtype, np.number) or np.issubdtype(src.dtype, np.bool_):
            vals = np.full(len(out), np.nan, dtype=float)
            vals[valid] = src[t_idx[valid]].astype(float, copy=False)
            out[c] = vals
        else:
            vals = np.empty(len(out), dtype=object)
            vals[:] = np.nan
            vals[valid] = src[t_idx[valid]]
            out[c] = vals
    return out


def compute_gap_stats_event_window(
    out: pd.DataFrame,
    knobs: Knobs,
) -> pd.DataFrame:
    """
    Compute gap_mu, gap_sd, gap_tail using only historical gap events.
    Critical: prevent leakage by shifting rolling results by 1 gap event.
    """
    # Identify bars where the next bar is next NY session (known by timestamp/calendar, not future prices)
    sdate = session_date_ny(out["date"])
    sdate_next = sdate.shift(-1)
    out["entry_overnight"] = (sdate_next.notna()) & (sdate_next != sdate)

    # Realized gap return uses open(t+1) and close(t); this is future for bar t.
    # We compute it to build HISTORICAL gap event series, but we must not leak gap_t into features at t.
    out["gap_ret_realized"] = np.where(
        out["entry_overnight"],
        np.log(out["open"].shift(-1) / out["close"]),
        np.nan,
    )

    gap_only = out.loc[out["gap_ret_realized"].notna(), "gap_ret_realized"]

    # Rolling stats over last M gap events, then shift by 1 event to exclude current event
    mu_sparse = gap_only.rolling(knobs.gap_lookback_events, min_periods=20).mean().shift(1)
    sd_sparse = gap_only.rolling(knobs.gap_lookback_events, min_periods=20).std().shift(1)

    # Tail count: |gap| > k * sd (use sd aligned to the same sparse index)
    # Use the shifted sd for threshold so the event at index i uses historical sd, not including itself.
    thr_sparse = knobs.gap_tail_k * sd_sparse
    tail_sparse = (gap_only.abs() > thr_sparse).astype(float).rolling(
        knobs.gap_lookback_events, min_periods=20
    ).sum().shift(1)

    # Reindex to all bars and forward fill so intraday bars inherit the latest known gap regime stats
    out["gap_mu"] = mu_sparse.reindex(out.index).ffill()
    out["gap_sd"] = sd_sparse.reindex(out.index).ffill()
    out["gap_tail"] = tail_sparse.reindex(out.index).ffill()

    return out


def compute_bar_features(df: pd.DataFrame, sym: str, knobs: Knobs) -> pd.DataFrame:
    out = df.copy()

    # Returns
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

    # Range ratio (liquidity/activity proxy)
    out["range_ratio"] = (out["high"] - out["low"]) / out["close"]

    # Trend EMAs and trend score
    out["ema_fast"] = out["close"].ewm(span=knobs.trend_fast_span, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=knobs.trend_slow_span, adjust=False).mean()
    out["trend_raw"] = (out["ema_fast"] - out["ema_slow"]) / out["close"]
    out["trend_score"] = out["trend_raw"] / out["sigma"].replace(0.0, np.nan)

    # EMA slope feature (requested in analysis plan)
    out["ema_fast_slope"] = (out["ema_fast"] - out["ema_fast"].shift(1)) / out["close"]

    # Pullback z-score around fast EMA (log space)
    pb_raw = out["log_close"] - np.log(out["ema_fast"].replace(0.0, np.nan))
    pb_std = pb_raw.ewm(span=knobs.trend_slow_span, adjust=False).std(bias=False)
    out["pullback_z"] = pb_raw / pb_std.replace(0.0, np.nan)

    # Volume z (log volume)
    lv = np.log(out["volume"].replace(0.0, np.nan))
    lv_mu = lv.rolling(200, min_periods=50).mean()
    lv_sd = lv.rolling(200, min_periods=50).std()
    out["vol_z"] = (lv - lv_mu) / lv_sd.replace(0.0, np.nan)

    # Percentile regime context (requested in plan/recommendations)
    out["sigma_prank"] = rolling_percentile_last(
        out["sigma"],
        window=knobs.sigma_rank_window,
        min_periods=knobs.rank_min_periods,
    )
    out["u_atr_prank"] = rolling_percentile_last(
        out["u_atr"],
        window=knobs.atr_rank_window,
        min_periods=knobs.rank_min_periods,
    )

    # Distance to rolling high (risk of chasing)
    roll_high = out["close"].rolling(100, min_periods=20).max().shift(1)
    out["dist_to_hi"] = (out["close"] - roll_high) / out["close"]

    # Intraday tail pressure: recent fraction of |r_cc| > z * sigma_prev
    sigma_prev = out["sigma"].shift(1).replace(0.0, np.nan)
    ret_z_abs = out["r_cc"].abs() / sigma_prev
    tail_flag = pd.Series(np.nan, index=out.index, dtype=float)
    valid_tail = ret_z_abs.notna()
    tail_flag.loc[valid_tail] = (ret_z_abs.loc[valid_tail] > knobs.intraday_tail_z).astype(float)
    out["intraday_tail_frac"] = tail_flag.rolling(
        knobs.intraday_tail_window,
        min_periods=knobs.intraday_tail_min_periods,
    ).mean()

    # Gap regime stats (event-window rolling, leakage-safe)
    out = compute_gap_stats_event_window(out, knobs)

    # Vol spike flag for slippage model
    sig = out["sigma"].copy()
    spike_thr = sig.rolling(500, min_periods=200).quantile(knobs.vol_spike_q)
    out["vol_spike"] = (sig > spike_thr).fillna(False)

    out["symbol"] = sym
    return out


# -----------------------------
# Cross-asset context
# -----------------------------
def build_cross_features(
    trade_bars: pd.DataFrame,
    cross_bars: pd.DataFrame,
    cross_symbol: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """
    Merge cross_symbol bar features into trade_symbol by timestamp using merge_asof.
    This is robust to minor timestamp mismatches.
    """
    take = [
        "date",
        "close",
        "r_cc",
        "sigma",
        "u_atr",
        "trend_score",
        "pullback_z",
        "vol_z",
        "dist_to_hi",
        "range_ratio",
        "ema_fast_slope",
    ]
    cb = cross_bars[take].copy()
    cross_ts_col = f"{cross_symbol.lower()}_date"
    cb = cb.rename(columns={"date": cross_ts_col})
    cb = cb.rename(columns={c: f"{cross_symbol.lower()}_{c}" for c in cb.columns if c != cross_ts_col})

    tb = trade_bars.sort_values("date").reset_index(drop=True)
    cb = cb.sort_values(cross_ts_col).reset_index(drop=True)

    merged = pd.merge_asof(
        tb,
        cb,
        left_on="date",
        right_on=cross_ts_col,
        direction="backward",
        tolerance=tolerance,
    )

    # Positive age in minutes since the matched cross bar (leakage-safe with backward join).
    merged[f"{cross_symbol.lower()}_age_min"] = (
        merged["date"] - merged[cross_ts_col]
    ).dt.total_seconds().div(60.0)

    # Derived cross relationships (keep small)
    cross_close = f"{cross_symbol.lower()}_close"
    cross_ret = f"{cross_symbol.lower()}_r_cc"

    merged["rs_log"] = np.log(merged["close"] / merged[cross_close])
    merged["ret_spread"] = merged["r_cc"] - merged[cross_ret]

    # Beta proxy (rolling)
    x = merged[cross_ret]
    y = merged["r_cc"]
    # covariance via E[xy] - E[x]E[y]
    exy = (x * y).rolling(200, min_periods=80).mean()
    ex = x.rolling(200, min_periods=80).mean()
    ey = y.rolling(200, min_periods=80).mean()
    cov = exy - ex * ey
    varx = x.rolling(200, min_periods=80).var()
    merged["beta_proxy"] = cov / varx.replace(0.0, np.nan)

    merged["regime_agree"] = (np.sign(merged["trend_score"]) == np.sign(merged[f"{cross_symbol.lower()}_trend_score"]))

    return merged


# -----------------------------
# Candidate generation
# -----------------------------
def candidates_from_features(bars: pd.DataFrame, knobs: Knobs) -> pd.DataFrame:
    df = bars.copy()

    # Candidate families
    df["cand_trend_long"] = df["trend_score"] > knobs.trend_in
    df["trend_regime"] = df["trend_score"] > knobs.trend_regime_min
    df["cand_pullback_long"] = df["trend_regime"] & (df["pullback_z"] < -knobs.pullback_z)

    df["is_candidate"] = df["cand_trend_long"] | df["cand_pullback_long"]
    df["family"] = np.where(
        df["cand_trend_long"],
        "trend_long",
        np.where(df["cand_pullback_long"], "pullback_long", ""),
    )

    # Need t+1 open for entry
    df["has_next"] = df["open"].shift(-1).notna()
    df = df[df["is_candidate"] & df["has_next"]].copy()

    # Spacing filter
    if knobs.min_event_spacing_bars > 1 and not df.empty:
        keep_idx: List[int] = []
        last_kept_i: Optional[int] = None
        for i in df.index.to_list():
            if last_kept_i is None or (i - last_kept_i) >= knobs.min_event_spacing_bars:
                keep_idx.append(i)
                last_kept_i = i
        df = df.loc[keep_idx].copy()

    return df


# -----------------------------
# Friction model
# -----------------------------
def friction_cost_return_units(row: pd.Series, sym: str, knobs: Knobs) -> Tuple[float, float, float]:
    """
    Return:
      spread_half (return units),
      slip_half   (return units),
      total_round_trip_cost (return units)
    """
    spread_half = (knobs.spread_half_bps.get(sym, 1.5)) / 10000.0

    sigma = float(row["sigma"]) if pd.notna(row.get("sigma", np.nan)) else 0.0
    slip_half = (knobs.slip_base_bps / 10000.0) + knobs.slip_vol_mult * sigma

    if bool(row.get("vol_spike", False)):
        slip_half += knobs.slip_spike_add_bps / 10000.0
    if bool(row.get("entry_overnight", False)):
        slip_half += knobs.overnight_slip_add_bps / 10000.0

    comm_rt = knobs.commission_round_trip_bps / 10000.0
    total = 2.0 * (spread_half + slip_half) + comm_rt
    return spread_half, slip_half, total


def friction_cost_from_state(
    sigma: float,
    vol_spike: bool,
    entry_overnight: bool,
    sym: str,
    knobs: Knobs,
) -> Tuple[float, float, float]:
    spread_half = (knobs.spread_half_bps.get(sym, 1.5)) / 10000.0
    sigma_v = float(sigma) if np.isfinite(sigma) else 0.0
    slip_half = (knobs.slip_base_bps / 10000.0) + knobs.slip_vol_mult * sigma_v
    if vol_spike:
        slip_half += knobs.slip_spike_add_bps / 10000.0
    if entry_overnight:
        slip_half += knobs.overnight_slip_add_bps / 10000.0
    comm_rt = knobs.commission_round_trip_bps / 10000.0
    total = 2.0 * (spread_half + slip_half) + comm_rt
    return spread_half, slip_half, total


# -----------------------------
# Labeling (triple barrier)
# -----------------------------
def decide_same_bar(
    policy: str,
    open_px: float,
    close_px: float,
) -> str:
    """
    When both TP and SL are touched within the same OHLC bar, order is unknowable.
    """
    if policy == "worst":
        return "sl"
    if policy == "best":
        return "tp"
    # close_direction
    return "tp" if close_px >= open_px else "sl"


def label_event_long(
    bars: pd.DataFrame,
    t_idx: int,
    sym: str,
    knobs: Knobs,
) -> Optional[Dict]:
    """
    Label a long event:
      - Decision at t_idx close
      - Entry at t_idx+1 open
      - Barrier touch detected using High/Low for each bar after entry
      - Exit at next bar open after first touch; if no touch, exit at horizon open

    Returns None if the event cannot be labeled due to bounds or invalid prices.
    """
    n = len(bars)
    if t_idx + 2 >= n:
        return None

    row_t = bars.iloc[t_idx]
    entry_i = t_idx + 1

    entry_open = float(bars.iloc[entry_i]["open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None

    overnight = bool(row_t.get("entry_overnight", False))
    H = knobs.horizon_overn if overnight else knobs.horizon_intra

    u = float(row_t["u_atr"]) if pd.notna(row_t.get("u_atr", np.nan)) else np.nan
    if not np.isfinite(u) or u <= 0:
        return None

    tp_mult = knobs.tp_mult_overn if overnight else knobs.tp_mult_intra
    sl_mult = knobs.sl_mult_overn if overnight else knobs.sl_mult_intra

    a = tp_mult * u
    b = sl_mult * u

    tp_px = entry_open * np.exp(a)
    sl_px = entry_open * np.exp(-b)

    spread_half, slip_half, cost_rt = friction_cost_return_units(row_t, sym, knobs)

    # Need an open for the exit bar, so cap scanning to n-2
    end_i = min(entry_i + H, n - 2)

    touch_i: Optional[int] = None
    touch_side: Optional[str] = None
    same_bar_ambiguous = 0

    for i in range(entry_i, end_i + 1):
        r = bars.iloc[i]
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])

        hit_tp = h >= tp_px
        hit_sl = l <= sl_px

        if hit_tp and hit_sl:
            same_bar_ambiguous = 1
            touch_side = decide_same_bar(knobs.same_bar_policy, o, c)
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

    horizon_capped = int((entry_i + H) > (n - 2))

    if touch_i is None:
        exit_i = min(entry_i + H, n - 2) + 1
        exit_reason = "horizon"
        touch_delay_bars = int(min(H, max(1, exit_i - entry_i)))
    else:
        exit_i = min(touch_i + 1, n - 1)
        exit_reason = "tp" if touch_side == "tp" else "sl"
        touch_delay_bars = int(max(1, touch_i - entry_i + 1))

    exit_open = float(bars.iloc[exit_i]["open"])
    if not np.isfinite(exit_open) or exit_open <= 0:
        return None

    gross = float(np.log(exit_open / entry_open))
    net = float(gross - cost_rt)
    y = int(net > 0)
    tp_to_cost = float(a / cost_rt) if cost_rt > 0 else np.nan

    return {
        "symbol": sym,
        "t_idx": int(t_idx),
        "decision_time_utc": str(row_t["date"]),
        "entry_time_utc": str(bars.iloc[entry_i]["date"]),
        "exit_time_utc": str(bars.iloc[exit_i]["date"]),
        "entry_open": float(entry_open),
        "exit_open": float(exit_open),
        "entry_overnight": int(bool(overnight)),
        "a_tp": float(a),
        "b_sl": float(b),
        "H": int(H),
        "exit_reason": str(exit_reason),
        "gross_logret": float(gross),
        "cost_rt": float(cost_rt),
        "net_logret": float(net),
        "y": int(y),
        "label_end_idx": int(exit_i),
        "truncated_horizon": int(horizon_capped),
        "touch_delay_bars": int(touch_delay_bars),
        "same_bar_ambiguous": int(same_bar_ambiguous),
        "spread_half": float(spread_half),
        "slip_half": float(slip_half),
        "tp_to_cost": tp_to_cost,
    }


def build_event_dataset(bars: pd.DataFrame, sym: str, knobs: Knobs) -> pd.DataFrame:
    cands = candidates_from_features(bars, knobs)
    if cands.empty:
        return pd.DataFrame()

    n = len(bars)
    open_px = bars["open"].to_numpy(dtype=float)
    high_px = bars["high"].to_numpy(dtype=float)
    low_px = bars["low"].to_numpy(dtype=float)
    close_px = bars["close"].to_numpy(dtype=float)
    sigma_arr = bars["sigma"].to_numpy(dtype=float)
    u_atr_arr = bars["u_atr"].to_numpy(dtype=float)
    vol_spike_arr = bars["vol_spike"].fillna(False).to_numpy(dtype=bool)
    overnight_arr = bars["entry_overnight"].fillna(False).to_numpy(dtype=bool)
    date_arr = bars["date"].to_numpy()

    feature_cols = [
        "trend_score",
        "pullback_z",
        "sigma",
        "u_atr",
        "vol_z",
        "dist_to_hi",
        "gap_mu",
        "gap_sd",
        "gap_tail",
        "range_ratio",
        "ema_fast_slope",
        "sigma_prank",
        "u_atr_prank",
        "intraday_tail_frac",
    ]
    feature_arrays: Dict[str, np.ndarray] = {
        c: bars[c].to_numpy(dtype=float) if c in bars.columns else np.full(n, np.nan, dtype=float)
        for c in feature_cols
    }
    family_by_t = cands["family"].astype(str).to_dict()

    labels: List[Dict] = []
    for t_idx in cands.index.to_numpy(dtype=np.int64, copy=False):
        if t_idx + 2 >= n:
            continue

        entry_i = int(t_idx + 1)
        entry_open = float(open_px[entry_i])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue

        overnight = bool(overnight_arr[t_idx])
        H = int(knobs.horizon_overn if overnight else knobs.horizon_intra)
        u = float(u_atr_arr[t_idx])
        if not np.isfinite(u) or u <= 0:
            continue

        tp_mult = knobs.tp_mult_overn if overnight else knobs.tp_mult_intra
        sl_mult = knobs.sl_mult_overn if overnight else knobs.sl_mult_intra
        a = float(tp_mult * u)
        b = float(sl_mult * u)

        tp_px = entry_open * np.exp(a)
        sl_px = entry_open * np.exp(-b)
        spread_half, slip_half, cost_rt = friction_cost_from_state(
            sigma=float(sigma_arr[t_idx]),
            vol_spike=bool(vol_spike_arr[t_idx]),
            entry_overnight=overnight,
            sym=sym,
            knobs=knobs,
        )

        end_i = min(entry_i + H, n - 2)
        touch_i: Optional[int] = None
        touch_side: Optional[str] = None
        same_bar_ambiguous = 0

        for i in range(entry_i, end_i + 1):
            hit_tp = high_px[i] >= tp_px
            hit_sl = low_px[i] <= sl_px
            if hit_tp and hit_sl:
                same_bar_ambiguous = 1
                touch_side = decide_same_bar(
                    knobs.same_bar_policy,
                    float(open_px[i]),
                    float(close_px[i]),
                )
                touch_i = int(i)
                break
            if hit_tp:
                touch_side = "tp"
                touch_i = int(i)
                break
            if hit_sl:
                touch_side = "sl"
                touch_i = int(i)
                break

        horizon_capped = int((entry_i + H) > (n - 2))
        if touch_i is None:
            exit_i = int(end_i + 1)
            exit_reason = "horizon"
            touch_delay_bars = int(min(H, max(1, exit_i - entry_i)))
        else:
            exit_i = int(min(touch_i + 1, n - 1))
            exit_reason = "tp" if touch_side == "tp" else "sl"
            touch_delay_bars = int(max(1, touch_i - entry_i + 1))

        exit_open = float(open_px[exit_i])
        if not np.isfinite(exit_open) or exit_open <= 0:
            continue

        gross = float(np.log(exit_open / entry_open))
        net = float(gross - cost_rt)
        lab = {
            "symbol": sym,
            "t_idx": int(t_idx),
            "decision_time_utc": str(date_arr[t_idx]),
            "entry_time_utc": str(date_arr[entry_i]),
            "exit_time_utc": str(date_arr[exit_i]),
            "entry_open": entry_open,
            "exit_open": exit_open,
            "entry_overnight": int(overnight),
            "a_tp": a,
            "b_sl": b,
            "H": H,
            "exit_reason": str(exit_reason),
            "gross_logret": gross,
            "cost_rt": float(cost_rt),
            "net_logret": net,
            "y": int(net > 0.0),
            "label_end_idx": exit_i,
            "truncated_horizon": int(horizon_capped),
            "touch_delay_bars": touch_delay_bars,
            "same_bar_ambiguous": int(same_bar_ambiguous),
            "spread_half": float(spread_half),
            "slip_half": float(slip_half),
            "tp_to_cost": float(a / cost_rt) if cost_rt > 0 else np.nan,
            "family": family_by_t.get(int(t_idx), ""),
        }
        for c, arr in feature_arrays.items():
            v = arr[t_idx]
            lab[c] = float(v) if np.isfinite(v) else np.nan
        labels.append(lab)

    return pd.DataFrame(labels)


def validate_events(events: pd.DataFrame) -> None:
    if events.empty:
        return
    d = pd.to_datetime(events["decision_time_utc"], utc=True, errors="coerce")
    e = pd.to_datetime(events["entry_time_utc"], utc=True, errors="coerce")
    x = pd.to_datetime(events["exit_time_utc"], utc=True, errors="coerce")

    if d.isna().any() or e.isna().any() or x.isna().any():
        raise ValueError("Invalid event timestamps detected.")
    if ((e <= d) | (x <= e)).any():
        raise ValueError("Invalid event time ordering detected.")
    if (events["cost_rt"] <= 0).any():
        raise ValueError("Non-positive round-trip costs detected.")
    if (events["a_tp"] <= 0).any() or (events["b_sl"] <= 0).any():
        raise ValueError("Non-positive barrier widths detected.")


def summarize_events_for_meta(events: pd.DataFrame) -> Dict[str, object]:
    if events.empty:
        return {"n_events": 0}

    out: Dict[str, object] = {
        "n_events": int(len(events)),
        "families": events["family"].value_counts(dropna=False).to_dict() if "family" in events.columns else {},
        "exit_reasons": events["exit_reason"].value_counts(dropna=False).to_dict() if "exit_reason" in events.columns else {},
        "y_rate": float(events["y"].mean()) if "y" in events.columns else None,
        "net_mean_bps": float(events["net_logret"].mean() * 10000.0),
        "gross_mean_bps": float(events["gross_logret"].mean() * 10000.0),
        "cost_mean_bps": float(events["cost_rt"].mean() * 10000.0),
        "overnight_rate": float(events["entry_overnight"].mean()) if "entry_overnight" in events.columns else None,
        "same_bar_ambiguous_rate": float(events["same_bar_ambiguous"].mean()) if "same_bar_ambiguous" in events.columns else None,
        "tp_to_cost_median": float(events["tp_to_cost"].median()) if "tp_to_cost" in events.columns else None,
    }

    d = pd.to_datetime(events["decision_time_utc"], utc=True, errors="coerce")
    if d.notna().any():
        weekly = events.assign(_d=d).set_index("_d").resample("W").size()
        if len(weekly) > 0:
            out["weekly_events_mean"] = float(weekly.mean())
            out["weekly_events_p90"] = float(weekly.quantile(0.9))
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing cleaned parquet files")
    ap.add_argument("--out-dir", required=True, help="Where to write datasets")
    ap.add_argument("--symbols", nargs="+", required=True, help="Symbols to load (must include trade symbol and cross if used)")
    ap.add_argument("--trade-symbol", default="QQQ", help="Symbol to build events for (default QQQ)")
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet", help="Input file suffix pattern")
    ap.add_argument("--no-cross", action="store_true", help="Disable cross-asset context merge")
    ap.add_argument("--cross-symbol", default="SPY", help="Cross symbol for context (default SPY)")
    ap.add_argument("--cross-tolerance", default="30min", help="merge_asof tolerance (default 30min)")
    ap.add_argument("--same-bar-policy", choices=["worst", "best", "close_direction"], default=None, help="Override same-bar policy")
    args = ap.parse_args()

    knobs = default_knobs()
    if args.no_cross:
        knobs.include_cross_asset = False
    knobs.cross_symbol = args.cross_symbol.upper()
    knobs.cross_merge_tolerance = args.cross_tolerance
    if args.same_bar_policy is not None:
        knobs.same_bar_policy = args.same_bar_policy

    tol = pd.Timedelta(knobs.cross_merge_tolerance)

    out_root = args.out_dir
    out_bar = os.path.join(out_root, "bar_features")
    out_evt = os.path.join(out_root, "events")
    out_meta = os.path.join(out_root, "meta")
    ensure_dir(out_bar)
    ensure_dir(out_evt)
    ensure_dir(out_meta)

    symbols = [s.upper() for s in args.symbols]
    trade_sym = args.trade_symbol.upper()
    if trade_sym not in symbols:
        raise ValueError(f"--trade-symbol {trade_sym} must be included in --symbols")

    # Load and compute bar features for all symbols
    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        in_path = os.path.join(args.data_dir, f"{sym.lower()}{args.bar_file_suffix}")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing input parquet: {in_path}")

        print(f"[{sym}] loading {in_path}")
        raw = read_parquet_any(in_path)
        feat = compute_bar_features(raw, sym, knobs)
        bars_by_sym[sym] = feat

        out_path = os.path.join(out_bar, f"{sym.lower()}_bar_features.parquet")
        feat.to_parquet(out_path, index=False)
        print(f"[{sym}] wrote bar features: {out_path}")

    trade_bars = bars_by_sym[trade_sym].copy()

    # Optional cross-asset merge
    cross_cols_added: List[str] = []
    if knobs.include_cross_asset:
        cross_sym = knobs.cross_symbol.upper()
        if cross_sym not in bars_by_sym:
            raise ValueError(f"Cross symbol {cross_sym} not present in --symbols")

        print(f"[CROSS] merge_asof {cross_sym} into {trade_sym} (tolerance={tol})")
        trade_bars = build_cross_features(trade_bars, bars_by_sym[cross_sym], cross_sym, tol)

        # record which cross columns exist (for later mapping into events)
        cross_cols_added = [c for c in trade_bars.columns if c.startswith(f"{cross_sym.lower()}_")] + [
            "rs_log", "ret_spread", "beta_proxy", "regime_agree"
        ]

        # quick sanity: percent non-null in one key cross column
        key_col = f"{cross_sym.lower()}_trend_score"
        if key_col in trade_bars.columns:
            pct = 100.0 * trade_bars[key_col].notna().mean()
            print(f"[CROSS] {key_col} non-null: {pct:.2f}%")
            if pct < 95.0:
                print("[CROSS] WARN: cross alignment appears imperfect. Consider increasing --cross-tolerance.")
        age_col = f"{cross_sym.lower()}_age_min"
        if age_col in trade_bars.columns:
            age = trade_bars[age_col].dropna()
            if not age.empty:
                print(f"[CROSS] {age_col} p99: {age.quantile(0.99):.2f} minutes")

    # Build events dataset
    print(f"[EVENTS] building events for {trade_sym}")
    events = build_event_dataset(trade_bars, trade_sym, knobs)

    if events.empty:
        print("[EVENTS] no events produced. Consider loosening thresholds or checking data.")
        return

    # Attach cross features at decision time (t_idx) if present
    if knobs.include_cross_asset and cross_cols_added:
        # A controlled subset to avoid feature explosion
        cross_sym_l = knobs.cross_symbol.lower()
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
        events = map_features_from_t_idx(events, tb, desired)

    # NaN handling: do NOT drop on every optional feature (too aggressive)
    if knobs.dropna_core_only:
        core = ["trend_score", "pullback_z", "sigma", "u_atr"]
        before = len(events)
        events = events.dropna(subset=core).reset_index(drop=True)
        after = len(events)
        dropped = before - after
        if dropped:
            print(f"[CLEAN] dropped rows missing CORE features: {dropped} (kept {after})")

    validate_events(events)

    out_events_path = os.path.join(out_evt, f"{trade_sym.lower()}_events.parquet")
    events.to_parquet(out_events_path, index=False)
    print(f"[EVENTS] wrote: {out_events_path}")

    tp_cost_med = float(events["tp_to_cost"].median()) if "tp_to_cost" in events.columns and not events.empty else np.nan
    if np.isfinite(tp_cost_med) and tp_cost_med < 2.0:
        print(
            f"[WARN] median tp_to_cost={tp_cost_med:.2f} is low; costs may dominate raw edge. "
            "Step 2b should tighten frequency/geometry."
        )

    meta = {
        "script": "step2_build_events_dataset.py",
        "version": SCRIPT_VERSION,
        "trade_symbol": trade_sym,
        "symbols_loaded": symbols,
        "knobs": asdict(knobs),
        "notes": {
            "barrier_touch": "High/Low touch detection; exit at next open; same-bar policy configurable",
            "gap_features": "Rolling stats computed over gap events only, shifted by 1 event to prevent leakage",
            "cross_asset": "Optional SPY context merged via merge_asof with tolerance; event-level mapping via t_idx",
            "nan_policy": "Drops rows only when core features are missing (recommended)",
        },
        "outputs": {
            "bar_features_dir": out_bar,
            "events_path": out_events_path,
        },
        "events_summary": summarize_events_for_meta(events),
    }
    out_meta_path = os.path.join(out_meta, "step2_config.json")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] wrote: {out_meta_path}")

    print("\nDone.")
    print(f"Events produced: {len(events)}")
    print("Next: Step 2.5 should analyze events (return distribution, cost sensitivity, family breakdown,")
    print("      regime slices, and cross-feature NaN rates if cross is enabled).")


if __name__ == "__main__":
    main()
