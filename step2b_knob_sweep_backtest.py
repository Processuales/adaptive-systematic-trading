#!/usr/bin/env python3
"""
Step 2b: knob sweep + holdout backtest (80/20 chronological split) with and without
a lightweight simulated ML risk gate.

Outputs:
  - <out-dir>/step2b_summary.json (compact, AI-friendly)
  - <out-dir>/best_equity_compare.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_build_events_dataset as s2

SCRIPT_VERSION = "1.1.0"


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


def add_trend_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trend_decile"] = np.nan
    valid = out["trend_score"].notna()
    if int(valid.sum()) < 100:
        return out
    try:
        out.loc[valid, "trend_decile"] = pd.qcut(
            out.loc[valid, "trend_score"], 10, labels=False, duplicates="drop"
        )
    except Exception:
        return out
    return out


def apply_policy_filter(events: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    df = add_trend_deciles(events)

    if policy_name == "baseline":
        mask = pd.Series(True, index=df.index)
    elif policy_name == "no_overnight":
        mask = df["entry_overnight"] == 0
    elif policy_name == "no_overnight_sigma_50_85":
        mask = (df["entry_overnight"] == 0) & df["sigma_prank"].between(0.50, 0.85)
    elif policy_name == "no_overnight_sigma_55_90":
        mask = (df["entry_overnight"] == 0) & df["sigma_prank"].between(0.55, 0.90)
    elif policy_name == "no_overnight_sigma_50_85_trend_decile_4_6":
        mask = (
            (df["entry_overnight"] == 0)
            & df["sigma_prank"].between(0.50, 0.85)
            & df["trend_decile"].between(4, 6)
        )
    elif policy_name == "no_overnight_sigma_50_85_trend_0p3_1p0":
        mask = (
            (df["entry_overnight"] == 0)
            & df["sigma_prank"].between(0.50, 0.85)
            & df["trend_score"].between(0.3, 1.0)
        )
    elif policy_name == "no_overnight_sigma_50_85_trend_decile_4_6_tp8":
        mask = (
            (df["entry_overnight"] == 0)
            & df["sigma_prank"].between(0.50, 0.85)
            & df["trend_decile"].between(4, 6)
            & (df["tp_to_cost"] >= 8.0)
        )
    elif policy_name == "no_overnight_sigma_50_85_trend_decile_4_6_tp10":
        mask = (
            (df["entry_overnight"] == 0)
            & df["sigma_prank"].between(0.50, 0.85)
            & df["trend_decile"].between(4, 6)
            & (df["tp_to_cost"] >= 10.0)
        )
    else:
        raise ValueError(f"Unknown policy filter: {policy_name}")

    return df[mask.fillna(False)].copy()


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


def split_train_test(events: pd.DataFrame, split_bar_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = events[events["label_end_idx"] < split_bar_idx].copy()
    test = events[events["t_idx"] >= split_bar_idx].copy()
    return train, test


def equity_curve(events: pd.DataFrame, start_equity: float = 10_000.0) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["time", "equity"])
    x = events.sort_values("exit_time_utc").copy()
    x["equity"] = start_equity * np.exp(x["net_logret"].cumsum())
    return x[["exit_time_utc", "equity"]].rename(columns={"exit_time_utc": "time"})


def summarize_performance(events: pd.DataFrame, start_equity: float = 10_000.0) -> Dict[str, Optional[float]]:
    if events.empty:
        return {
            "n": 0,
            "y_rate": None,
            "net_bps_mean": None,
            "net_bps_med": None,
            "cum_logret": None,
            "end_equity": start_equity,
            "max_drawdown": None,
            "cagr": None,
        }

    x = events.sort_values("exit_time_utc").copy()
    eq = start_equity * np.exp(x["net_logret"].cumsum())
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)

    t0 = x["exit_time_utc"].iloc[0]
    t1 = x["exit_time_utc"].iloc[-1]
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (float(eq.iloc[-1]) / start_equity) ** (1.0 / years) - 1.0

    return {
        "n": int(len(x)),
        "y_rate": float(x["y"].mean()),
        "net_bps_mean": float(x["net_logret"].mean() * 10000.0),
        "net_bps_med": float(x["net_logret"].median() * 10000.0),
        "cum_logret": float(x["net_logret"].sum()),
        "end_equity": float(eq.iloc[-1]),
        "max_drawdown": float(dd.max()),
        "cagr": float(cagr),
    }


def risk_adjusted_score(
    train_perf: Dict,
    test_perf: Dict,
    min_trades_test: int,
    target_test_trades: int = 0,
    trade_shortfall_penalty: float = 0.0,
) -> float:
    if int(test_perf.get("n") or 0) < min_trades_test:
        return -1e18
    end_eq = float(test_perf.get("end_equity") or 0.0)
    max_dd = float(test_perf.get("max_drawdown")) if test_perf.get("max_drawdown") is not None else 1.0
    train_edge = float(train_perf.get("net_bps_mean")) if train_perf.get("net_bps_mean") is not None else 0.0
    test_edge = float(test_perf.get("net_bps_mean")) if test_perf.get("net_bps_mean") is not None else 0.0
    gap_pen = abs(train_edge - test_edge)
    n_test = int(test_perf.get("n") or 0)
    trade_shortfall = max(0, target_test_trades - n_test) if target_test_trades > 0 else 0
    return end_eq * (1.0 - max_dd) - 40.0 * gap_pen - trade_shortfall_penalty * trade_shortfall


def fit_ridge_linear_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ridge_alpha: float,
) -> Optional[Dict]:
    if len(feature_cols) < 3 or len(train_df) < 80:
        return None

    y = train_df[target_col].astype(float).to_numpy()
    med = train_df[feature_cols].median()
    x = train_df[feature_cols].fillna(med).to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    xs = (x - mu) / sd
    xd = np.column_stack([np.ones(len(xs)), xs])

    eye = np.eye(xd.shape[1], dtype=float)
    eye[0, 0] = 0.0
    xtx = xd.T @ xd
    xty = xd.T @ y
    try:
        beta = np.linalg.solve(xtx + ridge_alpha * eye, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + ridge_alpha * eye) @ xty

    return {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "medians": med.to_dict(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "beta": beta.tolist(),
        "ridge_alpha": float(ridge_alpha),
    }


def fit_mock_ml_model(train_df: pd.DataFrame, feature_cols: List[str], ridge_alpha: float = 6.0) -> Optional[Dict]:
    return fit_ridge_linear_model(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col="y",
        ridge_alpha=ridge_alpha,
    )


def fit_mock_ml_return_model(train_df: pd.DataFrame, feature_cols: List[str], ridge_alpha: float = 8.0) -> Optional[Dict]:
    return fit_ridge_linear_model(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col="net_logret",
        ridge_alpha=ridge_alpha,
    )


def predict_ridge_linear(df: pd.DataFrame, model: Dict) -> np.ndarray:
    cols = model["feature_cols"]
    med = pd.Series(model["medians"])
    x = df[cols].fillna(med).to_numpy(dtype=float)
    mu = np.array(model["mu"], dtype=float)
    sd = np.array(model["sd"], dtype=float)
    beta = np.array(model["beta"], dtype=float)
    xs = (x - mu) / sd
    xd = np.column_stack([np.ones(len(xs)), xs])
    return xd @ beta


def predict_mock_ml_prob(df: pd.DataFrame, model: Dict) -> np.ndarray:
    raw = predict_ridge_linear(df, model)
    p = 1.0 / (1.0 + np.exp(-raw))
    # Shrink toward 0.5 so the simulated ML gate is deliberately conservative.
    p = 0.5 + 0.50 * (p - 0.5)
    return np.clip(p, 1e-4, 1.0 - 1e-4)


def score_mock_ml(
    df: pd.DataFrame,
    prob_model: Optional[Dict],
    ret_model: Optional[Dict],
    ret_mix_weight: float = 0.35,
) -> pd.DataFrame:
    if prob_model is None or df.empty:
        return df.iloc[0:0].copy()
    out = df.copy()
    p = predict_mock_ml_prob(out, prob_model)
    struct_edge = p * out["a_tp"].to_numpy() - (1.0 - p) * out["b_sl"].to_numpy() - out["cost_rt"].to_numpy()
    ret_pred = np.zeros(len(out), dtype=float)
    if ret_model is not None:
        ret_pred = predict_ridge_linear(out, ret_model)
    ret_pred = np.clip(ret_pred, -0.02, 0.02)
    edge_proxy = (1.0 - ret_mix_weight) * struct_edge + ret_mix_weight * ret_pred
    out["ml_prob"] = p
    out["ml_struct_edge"] = struct_edge
    out["ml_ret_pred"] = ret_pred
    out["ml_edge_proxy"] = edge_proxy
    return out


def apply_ml_thresholds(df_scored: pd.DataFrame, p_cut: float, edge_cut: float) -> pd.DataFrame:
    if df_scored.empty:
        return df_scored.copy()
    mask = (df_scored["ml_prob"] >= p_cut) & (df_scored["ml_edge_proxy"] >= edge_cut)
    return df_scored[mask].copy()


def tune_ml_thresholds(
    train_scored: pd.DataFrame,
    min_keep_train: int,
    target_keep_train: int,
) -> Tuple[Optional[Dict], pd.DataFrame]:
    if train_scored.empty:
        return None, pd.DataFrame()

    p_q_grid = [0.35, 0.45, 0.50, 0.55, 0.60, 0.65]
    e_q_grid = [0.20, 0.30, 0.40, 0.50, 0.60]

    candidates: List[Dict] = []
    p_base = train_scored["ml_prob"]
    e_base = train_scored["ml_edge_proxy"]
    for pq in p_q_grid:
        p_cut = float(max(0.49, p_base.quantile(pq)))
        for eq in e_q_grid:
            edge_cut = float(e_base.quantile(eq))
            gated = enforce_non_overlap(apply_ml_thresholds(train_scored, p_cut=p_cut, edge_cut=edge_cut))
            n = int(len(gated))
            if n < min_keep_train:
                continue
            perf = summarize_performance(gated)
            edge = float(perf["net_bps_mean"]) if perf["net_bps_mean"] is not None else -1e9
            dd = float(perf["max_drawdown"]) if perf["max_drawdown"] is not None else 1.0
            score = edge * np.sqrt(max(n, 1)) - 100.0 * dd - 0.8 * abs(n - target_keep_train)
            candidates.append(
                {
                    "p_cut": p_cut,
                    "edge_cut": edge_cut,
                    "train_perf": perf,
                    "train_n": n,
                    "tune_score": float(score),
                    "p_quantile": float(pq),
                    "edge_quantile": float(eq),
                }
            )

    if not candidates:
        return None, pd.DataFrame()
    best = max(candidates, key=lambda x: float(x["tune_score"]))
    best_df = enforce_non_overlap(
        apply_ml_thresholds(
            train_scored,
            p_cut=float(best["p_cut"]),
            edge_cut=float(best["edge_cut"]),
        )
    )
    return best, best_df


def sample_knobs(rng: np.random.Generator, friction_profile: str) -> s2.Knobs:
    k = s2.get_knobs_for_profile(friction_profile)

    k.trend_fast_span = int(rng.choice([8, 10, 12, 14, 16]))
    k.trend_slow_span = int(rng.choice([32, 40, 48, 56, 64]))
    if k.trend_slow_span <= k.trend_fast_span:
        k.trend_slow_span = k.trend_fast_span + 16

    k.trend_in = float(rng.uniform(0.45, 1.20))
    k.pullback_z = float(rng.uniform(0.70, 1.60))
    k.trend_regime_min = float(rng.uniform(0.15, 0.60))
    k.min_event_spacing_bars = int(rng.choice([1, 1, 1, 2, 3]))

    k.ewma_var_alpha = float(rng.uniform(0.04, 0.10))
    k.atr_span = int(rng.choice([10, 14, 18, 24]))

    k.horizon_intra = int(rng.choice([8, 10, 12, 14, 16]))
    k.horizon_overn = int(rng.choice([8, 10, 12, 14]))
    k.tp_mult_intra = float(rng.uniform(1.6, 3.0))
    k.sl_mult_intra = float(rng.uniform(1.0, 2.0))
    k.tp_mult_overn = float(k.tp_mult_intra + rng.uniform(0.0, 0.5))
    k.sl_mult_overn = float(k.sl_mult_intra + rng.uniform(0.0, 0.4))
    k.same_bar_policy = "worst"

    # Keep friction fixed for robustness, and stress-test in scoring phase.
    k.cross_merge_tolerance = "30min"
    k.dropna_core_only = True
    return k


def apply_market_hours_overrides(
    bars_by_sym: Dict[str, pd.DataFrame],
    market_hours: str,
) -> Dict[str, pd.DataFrame]:
    if market_hours != "24_7":
        return bars_by_sym
    out: Dict[str, pd.DataFrame] = {}
    for sym, bars in bars_by_sym.items():
        x = bars.copy()
        x["entry_overnight"] = False
        x["is_weekend"] = x["date"].dt.dayofweek.isin([5, 6]).astype(int)
        out[sym] = x
    return out


def build_events_from_knobs(
    raw_by_sym: Dict[str, pd.DataFrame],
    symbols: List[str],
    trade_symbol: str,
    cross_symbol: str,
    include_cross: bool,
    knobs: s2.Knobs,
    market_hours: str,
) -> Tuple[pd.DataFrame, int]:
    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        bars_by_sym[sym] = s2.compute_bar_features(raw_by_sym[sym], sym, knobs)
    bars_by_sym = apply_market_hours_overrides(bars_by_sym, market_hours=market_hours)

    trade_bars = bars_by_sym[trade_symbol].copy()
    tol = pd.Timedelta(knobs.cross_merge_tolerance)

    if include_cross:
        trade_bars = s2.build_cross_features(trade_bars, bars_by_sym[cross_symbol], cross_symbol, tol)

    events = s2.build_event_dataset(trade_bars, trade_symbol, knobs)
    if events.empty:
        return events, len(trade_bars)

    if include_cross:
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
    events = events.dropna(subset=core).reset_index(drop=True)

    for c in ["decision_time_utc", "entry_time_utc", "exit_time_utc"]:
        events[c] = pd.to_datetime(events[c], utc=True, errors="coerce")
    events = events.dropna(subset=["decision_time_utc", "entry_time_utc", "exit_time_utc"]).copy()
    events = events.sort_values(["decision_time_utc", "t_idx"]).reset_index(drop=True)
    return events, len(trade_bars)


def evaluate_trial(
    raw_by_sym: Dict[str, pd.DataFrame],
    symbols: List[str],
    trade_symbol: str,
    cross_symbol: str,
    include_cross: bool,
    knobs: s2.Knobs,
    friction_profile: str,
    market_hours: str,
    filter_name: str,
    min_trades_train: int,
    min_trades_test: int,
) -> Optional[Dict]:
    events, n_bars = build_events_from_knobs(
        raw_by_sym=raw_by_sym,
        symbols=symbols,
        trade_symbol=trade_symbol,
        cross_symbol=cross_symbol,
        include_cross=include_cross,
        knobs=knobs,
        market_hours=market_hours,
    )
    if events.empty:
        return None

    split_bar_idx = int(0.8 * n_bars)
    split_time = raw_by_sym[trade_symbol]["date"].iloc[min(max(split_bar_idx, 0), n_bars - 1)]

    base = apply_policy_filter(events, filter_name)
    if base.empty:
        return None

    train_base, test_base = split_train_test(base, split_bar_idx)
    if len(train_base) < min_trades_train or len(test_base) < min_trades_test:
        return None

    no_ml_train = enforce_non_overlap(train_base)
    no_ml_test = enforce_non_overlap(test_base)
    if len(no_ml_train) < min_trades_train or len(no_ml_test) < min_trades_test:
        return None

    ml_candidates = [
        "trend_score",
        "pullback_z",
        "sigma",
        "u_atr",
        "sigma_prank",
        "u_atr_prank",
        "gap_mu",
        "gap_sd",
        "gap_tail",
        "vol_z",
        "dist_to_hi",
        "ema_fast_slope",
        "intraday_tail_frac",
        "range_ratio",
        "tp_to_cost",
        "ret_spread",
        "beta_proxy",
        "entry_overnight",
    ]
    ml_features = [c for c in ml_candidates if c in train_base.columns]
    prob_model = fit_mock_ml_model(train_base, ml_features, ridge_alpha=6.0)
    ret_model = fit_mock_ml_return_model(train_base, ml_features, ridge_alpha=8.0)
    no_ml_train_perf = summarize_performance(no_ml_train)
    no_ml_test_perf = summarize_performance(no_ml_test)

    ml_valid = False
    p_cut = None
    edge_cut = None
    p_quantile = None
    edge_quantile = None
    tune_score = None
    target_keep_train = max(min_trades_train, 70)
    ml_train_perf = summarize_performance(pd.DataFrame())
    ml_test_perf = summarize_performance(pd.DataFrame())
    ml_train = pd.DataFrame()
    ml_test = pd.DataFrame()

    train_scored = score_mock_ml(train_base, prob_model=prob_model, ret_model=ret_model, ret_mix_weight=0.35)
    test_scored = score_mock_ml(test_base, prob_model=prob_model, ret_model=ret_model, ret_mix_weight=0.35)
    if not train_scored.empty and not test_scored.empty:
        min_keep_train = max(35, min_trades_train // 3)
        best_thr, tuned_train = tune_ml_thresholds(
            train_scored=train_scored,
            min_keep_train=min_keep_train,
            target_keep_train=target_keep_train,
        )
        if best_thr is not None:
            p_cut = float(best_thr["p_cut"])
            edge_cut = float(best_thr["edge_cut"])
            p_quantile = float(best_thr["p_quantile"])
            edge_quantile = float(best_thr["edge_quantile"])
            tune_score = float(best_thr["tune_score"])
            ml_train = tuned_train
            ml_test = enforce_non_overlap(apply_ml_thresholds(test_scored, p_cut=p_cut, edge_cut=edge_cut))

        if len(ml_train) >= min_keep_train and len(ml_test) >= min_trades_test:
            ml_valid = True
            ml_train_perf = summarize_performance(ml_train)
            ml_test_perf = summarize_performance(ml_test)

    return {
        "filter_name": filter_name,
        "friction_profile": str(friction_profile),
        "market_hours": str(market_hours),
        "knobs": asdict(knobs),
        "n_bars": int(n_bars),
        "split_bar_idx": int(split_bar_idx),
        "split_time_utc": str(split_time),
        "base_counts": {
            "all_events": int(len(events)),
            "filtered_events": int(len(base)),
            "train_filtered": int(len(train_base)),
            "test_filtered": int(len(test_base)),
        },
        "no_ml": {
            "train": no_ml_train_perf,
            "test": no_ml_test_perf,
            "score": risk_adjusted_score(
                train_perf=no_ml_train_perf,
                test_perf=no_ml_test_perf,
                min_trades_test=min_trades_test,
            ),
        },
        "ml_sim": {
            "model_features": ml_features,
            "p_cut": p_cut,
            "edge_cut": edge_cut,
            "p_quantile": p_quantile,
            "edge_quantile": edge_quantile,
            "threshold_tune_score": tune_score,
            "target_keep_train": int(target_keep_train),
            "ret_mix_weight": 0.35,
            "valid": ml_valid,
            "train": ml_train_perf,
            "test": ml_test_perf,
            "score": (
                risk_adjusted_score(
                    train_perf=ml_train_perf,
                    test_perf=ml_test_perf,
                    min_trades_test=min_trades_test,
                    target_test_trades=max(min_trades_test + 10, 35),
                    trade_shortfall_penalty=80.0,
                )
                if ml_valid
                else -1e18
            ),
        },
    }


def build_full_events_for_result(
    result: Dict,
    raw_by_sym: Dict[str, pd.DataFrame],
    symbols: List[str],
    trade_symbol: str,
    cross_symbol: str,
    include_cross: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    knobs = s2.Knobs(**result["knobs"])
    market_hours = str(result.get("market_hours") or "rth")
    events, _n_bars = build_events_from_knobs(
        raw_by_sym=raw_by_sym,
        symbols=symbols,
        trade_symbol=trade_symbol,
        cross_symbol=cross_symbol,
        include_cross=include_cross,
        knobs=knobs,
        market_hours=market_hours,
    )
    base = apply_policy_filter(events, result["filter_name"])
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    split_bar_idx = int(result["split_bar_idx"])
    train_base, _test_base = split_train_test(base, split_bar_idx)

    no_ml_full = enforce_non_overlap(base)

    ml_features = result["ml_sim"]["model_features"]
    prob_model = fit_mock_ml_model(train_base, ml_features, ridge_alpha=6.0)
    ret_model = fit_mock_ml_return_model(train_base, ml_features, ridge_alpha=8.0)
    ml_full = pd.DataFrame()
    if (
        result["ml_sim"].get("valid", False)
        and result["ml_sim"].get("p_cut") is not None
        and result["ml_sim"].get("edge_cut") is not None
    ):
        p_cut = float(result["ml_sim"]["p_cut"])
        edge_cut = float(result["ml_sim"]["edge_cut"])
        ret_mix_weight = float(result["ml_sim"].get("ret_mix_weight", 0.35))
        full_scored = score_mock_ml(
            base,
            prob_model=prob_model,
            ret_model=ret_model,
            ret_mix_weight=ret_mix_weight,
        )
        ml_full = enforce_non_overlap(apply_ml_thresholds(full_scored, p_cut=p_cut, edge_cut=edge_cut))

    return no_ml_full.reset_index(drop=True), ml_full.reset_index(drop=True)


def build_full_equity_for_result(
    result: Dict,
    raw_by_sym: Dict[str, pd.DataFrame],
    symbols: List[str],
    trade_symbol: str,
    cross_symbol: str,
    include_cross: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    no_ml_full, ml_full = build_full_events_for_result(
        result=result,
        raw_by_sym=raw_by_sym,
        symbols=symbols,
        trade_symbol=trade_symbol,
        cross_symbol=cross_symbol,
        include_cross=include_cross,
    )
    return equity_curve(no_ml_full), equity_curve(ml_full)


def plot_best_equity(no_ml_curve: pd.DataFrame, ml_curve: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(11, 6))
    if not no_ml_curve.empty:
        plt.plot(no_ml_curve["time"], no_ml_curve["equity"], color="red", linewidth=1.8, label="No ML simulation")
    if not ml_curve.empty:
        plt.plot(ml_curve["time"], ml_curve["equity"], color="green", linewidth=1.8, label="With ML simulation")
    plt.axhline(10_000.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Best Step 2b Simulation Equity (Start = $10,000)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with cleaned parquet files")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Loaded symbols")
    ap.add_argument("--trade-symbol", default="QQQ", help="Trade symbol")
    ap.add_argument("--cross-symbol", default="SPY", help="Cross context symbol")
    ap.add_argument("--no-cross", action="store_true", help="Disable cross-asset features")
    ap.add_argument("--bar-file-suffix", default="_1h_rth_clean.parquet")
    ap.add_argument("--out-dir", required=True, help="Output directory for step2b artifacts")
    ap.add_argument("--n-trials", type=int, default=120, help="Random knob trials")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min-trades-train", type=int, default=80, help="Minimum train trades after gating")
    ap.add_argument("--min-trades-test", type=int, default=25, help="Minimum test trades after gating")
    ap.add_argument("--top-n-json", type=int, default=30, help="How many top trials to keep in output JSON")
    ap.add_argument("--filters", type=str, default="", help="Comma-separated filter names to sweep (default built-in set)")
    ap.add_argument(
        "--friction-profile",
        choices=["equity", "crypto"],
        default="equity",
        help="Friction model profile passed into Step 2 event generation.",
    )
    ap.add_argument(
        "--market-hours",
        choices=["rth", "24_7"],
        default="rth",
        help="Market-hours mode: use '24_7' for crypto-style data.",
    )
    ap.add_argument("--pretty-json", action="store_true", help="Write indented JSON")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    symbols = [s.upper() for s in args.symbols]
    trade_symbol = args.trade_symbol.upper()
    cross_symbol = args.cross_symbol.upper()
    include_cross = not args.no_cross

    if trade_symbol not in symbols:
        raise ValueError(f"--trade-symbol {trade_symbol} must be included in --symbols")
    if include_cross and cross_symbol not in symbols:
        raise ValueError(f"--cross-symbol {cross_symbol} must be included in --symbols when cross is enabled")

    raw_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p = os.path.join(args.data_dir, f"{sym.lower()}{args.bar_file_suffix}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")
        raw_by_sym[sym] = s2.read_parquet_any(p)

    filter_names_default = [
        "baseline",
        "no_overnight",
        "no_overnight_sigma_50_85",
        "no_overnight_sigma_55_90",
        "no_overnight_sigma_50_85_trend_decile_4_6",
        "no_overnight_sigma_50_85_trend_0p3_1p0",
        "no_overnight_sigma_50_85_trend_decile_4_6_tp8",
        "no_overnight_sigma_50_85_trend_decile_4_6_tp10",
    ]
    if args.filters.strip():
        filter_names = [x.strip() for x in args.filters.split(",") if x.strip()]
        if not filter_names:
            raise ValueError("--filters was provided but no valid names were parsed.")
    else:
        filter_names = filter_names_default

    rng = np.random.default_rng(args.seed)
    results: List[Dict] = []

    for i in range(args.n_trials):
        knobs = sample_knobs(rng, friction_profile=args.friction_profile)
        knobs.include_cross_asset = include_cross
        knobs.cross_symbol = cross_symbol
        filter_name = str(rng.choice(filter_names))

        res = evaluate_trial(
            raw_by_sym=raw_by_sym,
            symbols=symbols,
            trade_symbol=trade_symbol,
            cross_symbol=cross_symbol,
            include_cross=include_cross,
            knobs=knobs,
            friction_profile=args.friction_profile,
            market_hours=args.market_hours,
            filter_name=filter_name,
            min_trades_train=args.min_trades_train,
            min_trades_test=args.min_trades_test,
        )

        if res is not None:
            res["trial"] = i + 1
            results.append(res)

        if (i + 1) % 10 == 0:
            print(f"[STEP2B] trial {i + 1}/{args.n_trials} | valid={len(results)}")

    if not results:
        raise RuntimeError("No valid trials produced. Try increasing n-trials or lowering min trade constraints.")

    results_ml_valid = [r for r in results if r["ml_sim"].get("valid", False)]
    results_ml_sorted = sorted(results_ml_valid, key=lambda r: float(r["ml_sim"]["score"]), reverse=True)
    results_no_sorted = sorted(results, key=lambda r: float(r["no_ml"]["score"]), reverse=True)

    if results_ml_sorted:
        best = results_ml_sorted[0]
    else:
        best = results_no_sorted[0]

    no_ml_curve, ml_curve = build_full_equity_for_result(
        result=best,
        raw_by_sym=raw_by_sym,
        symbols=symbols,
        trade_symbol=trade_symbol,
        cross_symbol=cross_symbol,
        include_cross=include_cross,
    )

    fig_path = os.path.join(args.out_dir, "best_equity_compare.png")
    plot_best_equity(no_ml_curve, ml_curve, fig_path)

    summary = {
        "meta": {
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_trials_requested": int(args.n_trials),
            "n_trials_valid": int(len(results)),
            "n_trials_ml_valid": int(len(results_ml_sorted)),
            "trade_symbol": trade_symbol,
            "symbols": symbols,
            "cross_symbol": cross_symbol,
            "include_cross": include_cross,
            "friction_profile": str(args.friction_profile),
            "market_hours": str(args.market_hours),
            "split_policy": "80/20 chronological by bar index; train uses label_end_idx < split, test uses t_idx >= split",
        },
        "constraints": {
            "min_trades_train": int(args.min_trades_train),
            "min_trades_test": int(args.min_trades_test),
        },
        "best_trial_ml": best,
        "top_trials_ml": results_ml_sorted[: max(1, args.top_n_json)],
        "top_trials_no_ml": results_no_sorted[: max(1, args.top_n_json)],
        "artifacts": {
            "best_equity_plot": fig_path,
        },
        "notes": {
            "ml_simulation": "Ridge probability + ridge return head on train set; threshold grid tuned on train only.",
            "overfit_controls": "Chronological split, minimum trade counts, risk-adjusted score, no test-time tuning.",
        },
    }

    summary = round_obj(summary, ndigits=6)
    out_json = os.path.join(args.out_dir, "step2b_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        if args.pretty_json:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        else:
            json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[STEP2B] Wrote: {out_json}")
    print(f"[STEP2B] Wrote: {fig_path}")


if __name__ == "__main__":
    main()
