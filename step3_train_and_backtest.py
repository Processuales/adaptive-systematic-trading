#!/usr/bin/env python3
"""
Step 3 walk-forward training + backtest.

Trains per-symbol ML models (QQQ/SPY), applies safe/aggressive policy decisions,
and builds a dual-symbol portfolio report under step3_out.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

SCRIPT_VERSION = "2.6.0"

PATTERN_CONTEXT_COLS = [
    "trend_score",
    "pullback_z",
    "sigma",
    "u_atr",
    "sigma_prank",
    "u_atr_prank",
    "gap_mu",
    "gap_sd",
    "gap_tail",
    "intraday_tail_frac",
    "range_ratio",
    "ema_fast_slope",
    "spy_sigma",
    "spy_u_atr",
    "spy_sigma_prank",
    "spy_u_atr_prank",
    "spy_trend_score",
    "spy_pullback_z",
    "spy_gap_sd",
    "spy_gap_tail",
    "spy_range_ratio",
    "spy_ema_fast_slope",
    "rs_log",
    "ret_spread",
    "beta_proxy",
    "regime_agree",
    "entry_overnight",
]

# Features below are only known after barrier resolution and leak future info.
LEAKY_FEATURE_COLS = {
    "hold_bars",
    "touch_delay_bars",
    "same_bar_ambiguous",
    "truncated_horizon",
    "y_loss",
}

REGIME_CONTEXT_COLS = [
    "trend_score",
    "pullback_z",
    "sigma",
    "u_atr",
    "vol_z",
    "dist_to_hi",
    "range_ratio",
    "ema_fast_slope",
    "sigma_prank",
    "u_atr_prank",
    "gap_mu",
    "gap_sd",
    "gap_tail",
    "intraday_tail_frac",
    "is_weekend",
    "entry_overnight",
    "tp_to_cost",
    "ret_spread",
    "beta_proxy",
    "regime_agree",
    "spy_sigma",
    "spy_u_atr",
    "spy_sigma_prank",
    "spy_u_atr_prank",
    "spy_trend_score",
    "spy_pullback_z",
    "spy_gap_sd",
    "spy_gap_tail",
    "spy_range_ratio",
    "spy_ema_fast_slope",
    "qqq_sigma",
    "qqq_u_atr",
    "qqq_sigma_prank",
    "qqq_u_atr_prank",
    "qqq_trend_score",
    "qqq_pullback_z",
    "qqq_gap_sd",
    "qqq_gap_tail",
    "qqq_range_ratio",
    "qqq_ema_fast_slope",
]


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
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    if not text:
        return out
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def load_meta_features(meta_path: str) -> List[str]:
    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    feats = m.get("feature_columns", [])
    if not feats:
        raise ValueError(f"No feature_columns in {meta_path}")
    return list(feats)


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def model_config_grid(policy_profile: str) -> List[Dict]:
    if policy_profile == "growth":
        return [
            {
                "name": "tree_heavy",
                "ridge_alpha_prob": 6.0,
                "ridge_alpha_ret": 8.0,
                "tree_weight_prob": 0.75,
                "tree_weight_ret": 0.80,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "min_child_samples": 28,
                "n_estimators": 240,
            },
            {
                "name": "tree_stable",
                "ridge_alpha_prob": 8.0,
                "ridge_alpha_ret": 10.0,
                "tree_weight_prob": 0.68,
                "tree_weight_ret": 0.72,
                "num_leaves": 23,
                "learning_rate": 0.04,
                "min_child_samples": 36,
                "n_estimators": 260,
            },
            {
                "name": "blend_conservative",
                "ridge_alpha_prob": 10.0,
                "ridge_alpha_ret": 12.0,
                "tree_weight_prob": 0.58,
                "tree_weight_ret": 0.62,
                "num_leaves": 15,
                "learning_rate": 0.035,
                "min_child_samples": 44,
                "n_estimators": 280,
            },
        ]
    return [
        {
            "name": "balanced_tree",
            "ridge_alpha_prob": 10.0,
            "ridge_alpha_ret": 12.0,
            "tree_weight_prob": 0.62,
            "tree_weight_ret": 0.66,
            "num_leaves": 23,
            "learning_rate": 0.04,
            "min_child_samples": 36,
            "n_estimators": 240,
        },
        {
            "name": "balanced_blend",
            "ridge_alpha_prob": 12.0,
            "ridge_alpha_ret": 14.0,
            "tree_weight_prob": 0.52,
            "tree_weight_ret": 0.56,
            "num_leaves": 15,
            "learning_rate": 0.03,
            "min_child_samples": 42,
            "n_estimators": 280,
        },
    ]


def estimate_embargo_bars(df: pd.DataFrame) -> int:
    if "H" not in df.columns or df.empty:
        return 1
    h = pd.to_numeric(df["H"], errors="coerce").dropna()
    if h.empty:
        return 1
    return max(1, int(np.ceil(float(h.quantile(0.90)))))


def purge_by_label_end(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    embargo_bars: int = 0,
) -> pd.DataFrame:
    if train_df.empty:
        return train_df.copy()
    required = {"t_idx", "label_end_idx"}
    if not required.issubset(train_df.columns) or not required.issubset(eval_df.columns):
        return train_df.copy()

    eval_t = pd.to_numeric(eval_df["t_idx"], errors="coerce").dropna()
    if eval_t.empty:
        return train_df.copy()
    first_eval_t = int(eval_t.min())
    cut = first_eval_t - max(0, int(embargo_bars))

    train_end = pd.to_numeric(train_df["label_end_idx"], errors="coerce")
    keep = train_end < cut
    return train_df[keep.fillna(False)].copy()


def make_inner_splits(
    fit_df: pd.DataFrame,
    min_fit_events: int,
    min_val_events: int,
    embargo_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    x = fit_df.sort_values("decision_time_utc").reset_index(drop=True)
    n = len(x)
    if n < (min_fit_events + min_val_events):
        return []
    ratios = [0.58, 0.72]
    val_frac = 0.18
    out: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for r in ratios:
        val_start_i = int(n * r)
        val_len = max(min_val_events, int(n * val_frac))
        val_end_i = min(n, val_start_i + val_len)
        if val_end_i - val_start_i < min_val_events:
            continue
        val_df = x.iloc[val_start_i:val_end_i].copy()
        if val_df.empty:
            continue
        val_start_t = pd.to_datetime(val_df["decision_time_utc"].iloc[0], utc=True)
        fit_end_t = val_start_t - pd.Timedelta(days=embargo_days)
        fit_split = x[(x["decision_time_utc"] < fit_end_t) & (x["exit_time_utc"] < val_start_t)].copy()
        fit_split = purge_by_label_end(
            fit_split,
            val_df,
            embargo_bars=estimate_embargo_bars(val_df),
        )
        if len(fit_split) < min_fit_events:
            continue
        out.append((fit_split, val_df))
    return out


def fit_lgbm_classifier(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict,
) -> Tuple[lgb.LGBMClassifier, Dict]:
    med = train_df[feature_cols].median()
    x = train_df[feature_cols].fillna(med).astype(float)
    y = train_df["y"].astype(int).to_numpy()
    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        min_child_samples=int(params["min_child_samples"]),
        n_estimators=int(params["n_estimators"]),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.6,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(x, y)
    return clf, med.to_dict()


def fit_lgbm_regressor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict,
    quantile_alpha: Optional[float] = None,
) -> Tuple[lgb.LGBMRegressor, Dict]:
    med = train_df[feature_cols].median()
    x = train_df[feature_cols].fillna(med).astype(float)
    y = train_df["net_logret"].astype(float).to_numpy()
    objective = "quantile" if quantile_alpha is not None else "regression"
    reg = lgb.LGBMRegressor(
        objective=objective,
        alpha=float(quantile_alpha) if quantile_alpha is not None else 0.5,
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        min_child_samples=int(params["min_child_samples"]),
        n_estimators=int(params["n_estimators"]),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.8,
        random_state=42,
        verbosity=-1,
    )
    reg.fit(x, y)
    return reg, med.to_dict()


def fit_model_bundle(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model_cfg: Dict,
) -> Dict:
    ridge_prob = fit_ridge_linear_model(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col="y",
        ridge_alpha=float(model_cfg["ridge_alpha_prob"]),
    )
    ridge_ret = fit_ridge_linear_model(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col="net_logret",
        ridge_alpha=float(model_cfg["ridge_alpha_ret"]),
    )
    if ridge_prob is None or ridge_ret is None:
        raise RuntimeError("Could not fit ridge models in model bundle.")

    cls, med_tree = fit_lgbm_classifier(train_df, feature_cols, model_cfg)
    reg, _ = fit_lgbm_regressor(train_df, feature_cols, model_cfg, quantile_alpha=None)
    q50, _ = fit_lgbm_regressor(train_df, feature_cols, model_cfg, quantile_alpha=0.50)
    q10, _ = fit_lgbm_regressor(train_df, feature_cols, model_cfg, quantile_alpha=0.10)
    q90, _ = fit_lgbm_regressor(train_df, feature_cols, model_cfg, quantile_alpha=0.90)

    return {
        "feature_cols": feature_cols,
        "model_cfg": model_cfg,
        "ridge_prob": ridge_prob,
        "ridge_ret": ridge_ret,
        "lgb_cls": cls,
        "lgb_ret": reg,
        "lgb_q50": q50,
        "lgb_q10": q10,
        "lgb_q90": q90,
        "tree_medians": med_tree,
    }


def predict_model_bundle(
    df: pd.DataFrame,
    bundle: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = bundle["model_cfg"]
    feature_cols = bundle["feature_cols"]
    med_tree = pd.Series(bundle["tree_medians"])
    x_tree = df[feature_cols].fillna(med_tree).astype(float)

    p_ridge = sigmoid(predict_ridge_linear(df, bundle["ridge_prob"]))
    r_ridge = predict_ridge_linear(df, bundle["ridge_ret"])

    p_tree = bundle["lgb_cls"].predict_proba(x_tree)[:, 1]
    r_tree_mean = bundle["lgb_ret"].predict(x_tree)
    r_tree_q50 = bundle["lgb_q50"].predict(x_tree)
    r_tree = 0.65 * r_tree_mean + 0.35 * r_tree_q50
    q10 = bundle["lgb_q10"].predict(x_tree)
    q90 = bundle["lgb_q90"].predict(x_tree)

    wp = float(cfg["tree_weight_prob"])
    wr = float(cfg["tree_weight_ret"])
    p_raw = np.clip(wp * p_tree + (1.0 - wp) * p_ridge, 1e-4, 1.0 - 1e-4)
    r_raw = np.clip(wr * r_tree + (1.0 - wr) * r_ridge, -0.05, 0.05)
    uncert = np.clip(q90 - q10, 1e-6, 0.08)
    return p_raw, r_raw, np.clip(q10, -0.05, 0.05), np.clip(q90, -0.05, 0.05), uncert


def fit_probability_calibrator(y_true: np.ndarray, p_raw: np.ndarray) -> Dict:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-4, 1.0 - 1e-4)
    if len(np.unique(y)) < 2 or len(np.unique(np.round(p, 4))) < 8:
        return {"method": "identity"}
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return {"method": "isotonic", "model": iso}
    except Exception:
        pass
    try:
        lr = LogisticRegression(max_iter=400, solver="lbfgs")
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr.fit(x, y)
        return {"method": "platt", "model": lr}
    except Exception:
        return {"method": "identity"}


def apply_probability_calibrator(calib: Dict, p_raw: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-4, 1.0 - 1e-4)
    method = calib.get("method", "identity")
    if method == "isotonic":
        out = np.asarray(calib["model"].predict(p), dtype=float)
        return np.clip(out, 1e-4, 1.0 - 1e-4)
    if method == "platt":
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        out = np.asarray(calib["model"].predict_proba(x)[:, 1], dtype=float)
        return np.clip(out, 1e-4, 1.0 - 1e-4)
    return p


def fit_confidence_calibrator(y_true: np.ndarray, conf_raw: np.ndarray) -> Dict:
    y = np.asarray(y_true, dtype=int)
    c = np.clip(np.asarray(conf_raw, dtype=float), 0.0, 1.0)
    if len(y) < 30 or len(np.unique(y)) < 2 or len(np.unique(np.round(c, 3))) < 8:
        return {"method": "identity"}
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(c, y)
        return {"method": "isotonic", "model": iso}
    except Exception:
        return {"method": "identity"}


def apply_confidence_calibrator(calib: Dict, conf_raw: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(conf_raw, dtype=float), 0.0, 1.0)
    if calib.get("method") == "isotonic":
        out = np.asarray(calib["model"].predict(c), dtype=float)
        return np.clip(out, 0.0, 1.0)
    return c


def fit_uncertainty_calibrator(uncertainty: np.ndarray) -> Dict:
    u = np.asarray(uncertainty, dtype=float)
    return {
        "u_q25": float(np.quantile(u, 0.25)),
        "u_q75": float(np.quantile(u, 0.75)),
    }


def confidence_from_predictions(
    p_cal: np.ndarray,
    uncertainty: np.ndarray,
    unc_calib: Dict,
) -> np.ndarray:
    p = np.asarray(p_cal, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    conf_prob = np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)
    uq25 = float(unc_calib.get("u_q25", np.quantile(u, 0.25)))
    uq75 = float(unc_calib.get("u_q75", np.quantile(u, 0.75)))
    conf_unc = 1.0 - np.clip((u - uq25) / max(uq75 - uq25, 1e-8), 0.0, 1.0)
    conf = 0.65 * conf_prob + 0.35 * conf_unc
    return np.clip(conf, 0.0, 1.0)


def select_regime_cols(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    numeric = set(df.select_dtypes(include=[np.number]).columns.tolist())
    cols: List[str] = []
    for c in candidate_cols:
        if c not in df.columns or c not in numeric:
            continue
        if c in LEAKY_FEATURE_COLS:
            continue
        if df[c].notna().mean() < 0.30:
            continue
        sd = float(pd.to_numeric(df[c], errors="coerce").std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-9:
            continue
        cols.append(c)
    return sorted(cols)


def _forward_window_mean(s: pd.Series, lookahead: int, min_periods: int) -> pd.Series:
    # Forward mean over next `lookahead` events (excluding current row).
    return s.shift(-1).rolling(lookahead, min_periods=min_periods).mean().shift(-(lookahead - 1))


def fit_regime_classifier(
    train_df: pd.DataFrame,
    candidate_cols: List[str],
    lookahead_events: int,
    label_quantile: float,
    min_train_samples: int,
) -> Dict:
    cols = select_regime_cols(train_df, candidate_cols)
    if len(cols) < 6:
        return {"enabled": False, "reason": "insufficient_regime_features", "selected_cols": cols}

    x = train_df.sort_values("decision_time_utc").reset_index(drop=True).copy()
    n = len(x)
    look = int(max(6, lookahead_events))
    if n < max(min_train_samples, look + 20):
        return {"enabled": False, "reason": "insufficient_regime_rows", "selected_cols": cols}

    net = pd.to_numeric(x["net_logret"], errors="coerce")
    win = pd.to_numeric(x["y"], errors="coerce")
    min_p = int(max(4, 0.7 * look))
    fwd_ret = _forward_window_mean(net, lookahead=look, min_periods=min_p)
    fwd_win = _forward_window_mean(win, lookahead=look, min_periods=min_p)
    regime_score = 10000.0 * fwd_ret + 20.0 * (fwd_win - 0.5)

    valid = regime_score.notna()
    if int(valid.sum()) < min_train_samples:
        return {"enabled": False, "reason": "insufficient_regime_labels", "selected_cols": cols}

    q = float(np.clip(label_quantile, 0.45, 0.70))
    threshold = float(regime_score.loc[valid].quantile(q))
    regime_y = (regime_score >= threshold).astype(int)
    train_lab = x.loc[valid, cols].copy()
    train_lab["regime_y"] = regime_y.loc[valid].astype(int).to_numpy()
    train_lab = train_lab.dropna(subset=["regime_y"])
    if len(train_lab) < min_train_samples:
        return {"enabled": False, "reason": "insufficient_regime_labeled_rows", "selected_cols": cols}

    y_train = train_lab["regime_y"].astype(int).to_numpy()
    pos_rate = float(np.mean(y_train))
    if pos_rate < 0.20 or pos_rate > 0.80:
        return {
            "enabled": False,
            "reason": "regime_label_imbalance",
            "selected_cols": cols,
            "positive_rate": pos_rate,
        }

    med = train_lab[cols].median()
    x_train = train_lab[cols].fillna(med).astype(float)
    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=15,
        learning_rate=0.04,
        min_child_samples=24,
        n_estimators=220,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.8,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(x_train, y_train)

    p_raw = clf.predict_proba(x_train)[:, 1]
    calib = fit_probability_calibrator(y_train, p_raw)
    p_cal = apply_probability_calibrator(calib, p_raw)
    brier = float(np.mean((p_cal - y_train.astype(float)) ** 2))

    return {
        "enabled": True,
        "selected_cols": cols,
        "lookahead_events": look,
        "label_quantile": q,
        "threshold": threshold,
        "medians": med.to_dict(),
        "model": clf,
        "calibrator": calib,
        "n_train": int(len(train_lab)),
        "positive_rate": pos_rate,
        "brier_in_sample": brier,
    }


def predict_regime_classifier(df: pd.DataFrame, regime_model: Dict) -> np.ndarray:
    n = len(df)
    if n <= 0:
        return np.array([], dtype=float)
    if not bool(regime_model.get("enabled")):
        return np.full(n, 0.5, dtype=float)
    cols = list(regime_model.get("selected_cols") or [])
    if not cols:
        return np.full(n, 0.5, dtype=float)
    med = pd.Series(regime_model.get("medians") or {})
    x = df[cols].fillna(med).astype(float)
    p_raw = regime_model["model"].predict_proba(x)[:, 1]
    p_cal = apply_probability_calibrator(regime_model.get("calibrator") or {"method": "identity"}, p_raw)
    return np.clip(np.asarray(p_cal, dtype=float), 1e-4, 1.0 - 1e-4)


def select_pattern_cols(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    numeric = set(df.select_dtypes(include=[np.number]).columns.tolist())
    cols: List[str] = []
    for c in candidate_cols:
        if c not in df.columns or c not in numeric:
            continue
        if df[c].notna().mean() < 0.45:
            continue
        sd = float(pd.to_numeric(df[c], errors="coerce").std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-9:
            continue
        cols.append(c)
    return cols


def fit_pattern_context_model(
    train_df: pd.DataFrame,
    pattern_cols: List[str],
    n_clusters: int,
    min_cluster_samples: int,
    prior_strength: float,
    consistency_tol: float,
) -> Dict:
    cols = select_pattern_cols(train_df, pattern_cols)
    min_needed = max(120, int(max(2, n_clusters) * max(10, min_cluster_samples)))
    if len(cols) < 6 or len(train_df) < min_needed:
        return {"enabled": False, "reason": "insufficient_data_or_features", "selected_cols": cols}

    x = train_df[cols].copy()
    med = x.median()
    xv = x.fillna(med).to_numpy(dtype=float)
    mu = xv.mean(axis=0)
    sd = xv.std(axis=0)
    sd[sd == 0.0] = 1.0
    xs = (xv - mu) / sd

    y = train_df["y"].to_numpy(dtype=int)
    r = train_df["net_logret"].to_numpy(dtype=float)
    global_win = float(np.mean(y))
    global_ev = float(np.mean(r))

    max_k = max(2, min(int(n_clusters), len(train_df) // max(12, min_cluster_samples)))
    if max_k < 2:
        return {"enabled": False, "reason": "too_few_events_for_clusters", "selected_cols": cols}

    try:
        km = KMeans(n_clusters=max_k, random_state=42, n_init=20)
        cidx = km.fit_predict(xs)
    except Exception:
        return {"enabled": False, "reason": "kmeans_fit_failed", "selected_cols": cols}

    split_i = max(1, int(0.5 * len(train_df)))
    order = np.arange(len(train_df), dtype=int)
    is_a = order < split_i
    is_b = ~is_a

    prob_delta = np.zeros(max_k, dtype=float)
    ev_delta = np.zeros(max_k, dtype=float)
    consistency_arr = np.zeros(max_k, dtype=float)
    support_arr = np.zeros(max_k, dtype=int)
    cluster_rows: List[Dict] = []

    min_half = max(6, min_cluster_samples // 3)
    prior = float(max(1.0, prior_strength))
    tol = float(max(1e-6, consistency_tol))

    for c in range(max_k):
        m = cidx == c
        n = int(np.sum(m))
        support_arr[c] = n
        if n <= 0:
            continue

        yc = y[m]
        rc = r[m]
        win = float(np.mean(yc))
        ev = float(np.mean(rc))
        w = float(n / (n + prior))
        win_shrunk = global_win + w * (win - global_win)
        ev_shrunk = global_ev + w * (ev - global_ev)

        ma = m & is_a
        mb = m & is_b
        na = int(np.sum(ma))
        nb = int(np.sum(mb))
        if na < min_half or nb < min_half:
            consistency = 0.0
            ev_a = 0.0
            ev_b = 0.0
        else:
            ev_a = float(np.mean(r[ma]))
            ev_b = float(np.mean(r[mb]))
            sign_ok = (np.sign(ev_a) == np.sign(ev_b)) or (abs(ev_a) < 1e-8) or (abs(ev_b) < 1e-8)
            stab = float(np.clip(1.0 - abs(ev_a - ev_b) / tol, 0.0, 1.0))
            consistency = (1.0 if sign_ok else 0.25) * stab

        if n < min_cluster_samples:
            consistency *= float(n / max(1, min_cluster_samples))

        dp = (win_shrunk - global_win) * consistency
        de = ev_shrunk * consistency
        prob_delta[c] = float(np.clip(dp, -0.25, 0.25))
        ev_delta[c] = float(np.clip(de, -0.01, 0.01))
        consistency_arr[c] = float(np.clip(consistency, 0.0, 1.0))

        cluster_rows.append(
            {
                "cluster": int(c),
                "n": n,
                "win_rate_raw": win,
                "ev_raw": ev,
                "win_rate_shrunk": win_shrunk,
                "ev_shrunk": ev_shrunk,
                "ev_half_a": ev_a,
                "ev_half_b": ev_b,
                "consistency": consistency_arr[c],
                "prob_delta": prob_delta[c],
                "ev_delta": ev_delta[c],
            }
        )

    return {
        "enabled": True,
        "selected_cols": cols,
        "medians": med.to_dict(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "global_win_rate": global_win,
        "global_ev": global_ev,
        "n_clusters": int(max_k),
        "min_cluster_samples": int(min_cluster_samples),
        "prior_strength": float(prior_strength),
        "consistency_tol": float(consistency_tol),
        "kmeans": km,
        "cluster_prob_delta": prob_delta.tolist(),
        "cluster_ev_delta": ev_delta.tolist(),
        "cluster_consistency": consistency_arr.tolist(),
        "cluster_support": support_arr.tolist(),
        "cluster_rows": cluster_rows,
    }


def apply_pattern_context_model(
    df: pd.DataFrame,
    pattern_model: Dict,
    prob_strength: float,
    ret_strength: float,
    prob_max_abs_delta: float,
    ret_max_abs_delta: float,
) -> Dict:
    n = int(len(df))
    if n <= 0 or (not bool(pattern_model.get("enabled"))):
        return {
            "pattern_cluster": np.full(n, -1, dtype=int),
            "pattern_prob_delta": np.zeros(n, dtype=float),
            "pattern_ret_delta": np.zeros(n, dtype=float),
            "pattern_consistency": np.zeros(n, dtype=float),
            "pattern_support": np.zeros(n, dtype=float),
        }

    cols = list(pattern_model["selected_cols"])
    med = pd.Series(pattern_model["medians"])
    mu = np.array(pattern_model["mu"], dtype=float)
    sd = np.array(pattern_model["sd"], dtype=float)
    xv = df[cols].fillna(med).to_numpy(dtype=float)
    xs = (xv - mu) / sd

    km: KMeans = pattern_model["kmeans"]
    cidx = km.predict(xs).astype(int)
    p_delta_arr = np.array(pattern_model["cluster_prob_delta"], dtype=float)
    r_delta_arr = np.array(pattern_model["cluster_ev_delta"], dtype=float)
    cons_arr = np.array(pattern_model["cluster_consistency"], dtype=float)
    supp_arr = np.array(pattern_model["cluster_support"], dtype=float)

    p_delta = np.clip(float(prob_strength) * p_delta_arr[cidx], -float(prob_max_abs_delta), float(prob_max_abs_delta))
    r_delta = np.clip(float(ret_strength) * r_delta_arr[cidx], -float(ret_max_abs_delta), float(ret_max_abs_delta))
    return {
        "pattern_cluster": cidx,
        "pattern_prob_delta": p_delta.astype(float),
        "pattern_ret_delta": r_delta.astype(float),
        "pattern_consistency": cons_arr[cidx].astype(float),
        "pattern_support": supp_arr[cidx].astype(float),
    }


def _daily_drift_rate_from_scored(scored: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    if scored.empty or "decision_time_utc" not in scored.columns or "drift_flag" not in scored.columns:
        return pd.Series(0.0, index=idx, dtype=float)
    x = scored.copy()
    x["decision_time_utc"] = pd.to_datetime(x["decision_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["decision_time_utc"])
    if x.empty:
        return pd.Series(0.0, index=idx, dtype=float)
    x["d"] = x["decision_time_utc"].dt.floor("D")
    d = x.groupby("d")["drift_flag"].mean().sort_index()
    return d.reindex(idx).ffill().fillna(0.0).astype(float)


def build_spy_guard_series(
    idx: pd.DatetimeIndex,
    q_ret: pd.Series,
    s_ret: pd.Series,
    spy_scored: pd.DataFrame,
    lookback_days: int,
    drift_lookback_days: int,
    min_mult: float,
    dd_penalty: float,
) -> pd.Series:
    q = q_ret.reindex(idx).fillna(0.0)
    s = s_ret.reindex(idx).fillna(0.0)

    rel = (s - q).shift(1)
    rel_mu = rel.rolling(lookback_days).mean()
    rel_sd = rel.rolling(lookback_days).std(ddof=0).replace(0.0, np.nan)
    rel_z = (rel_mu / rel_sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trend_score = 0.5 + 0.5 * np.tanh(1.8 * rel_z)

    drift_daily = _daily_drift_rate_from_scored(spy_scored, idx)
    drift_rate = drift_daily.shift(1).rolling(drift_lookback_days).mean().fillna(0.0)
    drift_score = np.clip(1.0 - drift_rate / 0.60, 0.0, 1.0)

    s_eq = (1.0 + s).cumprod()
    s_dd = 1.0 - s_eq / s_eq.cummax()
    dd_worst = s_dd.shift(1).rolling(lookback_days).max().fillna(0.0)
    dd_score = np.clip(1.0 - dd_penalty * dd_worst, 0.0, 1.0)

    raw = 0.55 * trend_score + 0.30 * drift_score + 0.15 * dd_score
    return pd.Series(np.clip(raw, min_mult, 1.0), index=idx, dtype=float)


def feature_drift_snapshot(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    if train_df.empty or test_df.empty:
        return {"mean_abs_z": 0.0, "max_abs_z": 0.0, "top_shift_features": []}
    mu = train_df[feature_cols].mean()
    sd = train_df[feature_cols].std(ddof=0).replace(0.0, np.nan)
    tm = test_df[feature_cols].mean()
    z = ((tm - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    absz = z.abs().sort_values(ascending=False)
    top = [{"feature": str(k), "abs_z": float(v)} for k, v in absz.head(5).items()]
    return {
        "mean_abs_z": float(absz.mean()),
        "max_abs_z": float(absz.max()),
        "top_shift_features": top,
    }


def performance_drift_flag(prev_fold_rows: List[Dict], current_feature_drift: Dict) -> Tuple[bool, Dict]:
    mean_abs_z = float(current_feature_drift.get("mean_abs_z") or 0.0)
    recent = prev_fold_rows[-6:] if len(prev_fold_rows) >= 2 else prev_fold_rows
    recent_bps = [float(r.get("test_net_bps_mean") or 0.0) for r in recent if r.get("test_net_bps_mean") is not None]
    recent_calmar = [float(r.get("test_calmar") or -1.0) for r in recent if r.get("test_calmar") is not None]
    bps_med = float(np.median(recent_bps)) if recent_bps else 0.0
    cal_med = float(np.median(recent_calmar)) if recent_calmar else -1.0
    flag = (mean_abs_z > 1.35) or (bps_med < 0.0 and cal_med < 0.0)
    return flag, {
        "mean_abs_z": mean_abs_z,
        "recent_median_net_bps": bps_med,
        "recent_median_calmar": cal_med,
    }


def perf_from_trade_logrets(
    times: pd.Series,
    logrets: np.ndarray,
    n_trades: int,
    aggressive_rate: float,
    start_equity: float = 10_000.0,
) -> Dict:
    if n_trades <= 0:
        return {
            "n": 0,
            "end_equity": start_equity,
            "cagr": None,
            "max_drawdown": None,
            "calmar": None,
            "net_bps_mean": None,
            "aggressive_rate": 0.0,
            "trades_per_month": 0.0,
        }
    eq = start_equity * np.exp(np.cumsum(logrets))
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    mdd = float(dd.max())

    t0 = pd.to_datetime(times.iloc[0], utc=True)
    t1 = pd.to_datetime(times.iloc[-1], utc=True)
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1.0 / 12.0)
    end_ratio = max(float(eq[-1]) / float(start_equity), 1e-12)
    cagr = np.exp(np.clip(np.log(end_ratio) / years, -20.0, 20.0)) - 1.0
    calmar = float(cagr / mdd) if mdd > 0 else None
    months = max((t1 - t0).total_seconds() / (30.44 * 24 * 3600), 1e-9)
    trades_per_month = float(n_trades / months)
    return {
        "n": int(n_trades),
        "end_equity": float(eq[-1]),
        "cagr": float(cagr),
        "max_drawdown": mdd,
        "calmar": calmar,
        "net_bps_mean": float(np.mean(logrets) * 10000.0),
        "aggressive_rate": float(aggressive_rate),
        "trades_per_month": trades_per_month,
    }


def policy_score(perf: Dict, min_trades: int) -> float:
    n = int(perf.get("n") or 0)
    if n < min_trades:
        return -1e18
    calmar = float(perf["calmar"]) if perf.get("calmar") is not None else -1.0
    dd = float(perf["max_drawdown"]) if perf.get("max_drawdown") is not None else 1.0
    net_bps = float(perf["net_bps_mean"]) if perf.get("net_bps_mean") is not None else -1000.0
    tpm = float(perf.get("trades_per_month") or 0.0)
    return calmar + 0.06 * min(tpm, 12.0) + 0.01 * max(net_bps, 0.0) - 2.5 * max(0.0, dd - 0.12)


def simulate_policy(
    df: pd.DataFrame,
    p_pred: np.ndarray,
    ret_pred: np.ndarray,
    params: Dict,
    q10_pred: Optional[np.ndarray] = None,
    regime_prob: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    out["p_pred"] = np.clip(p_pred, 1e-4, 1.0 - 1e-4)
    out["ret_pred"] = np.clip(ret_pred, -0.03, 0.03)
    if q10_pred is None:
        out["q10_pred"] = out["ret_pred"]
    else:
        q10 = np.asarray(q10_pred, dtype=float)
        if len(q10) != len(out):
            raise ValueError("q10_pred length must match df rows in simulate_policy.")
        out["q10_pred"] = np.clip(q10, -0.05, 0.05)
    out["ev_struct"] = (
        out["p_pred"] * out["a_tp"] - (1.0 - out["p_pred"]) * out["b_sl"] - out["cost_rt"]
    )
    mix = float(params["mix_struct_weight"])
    out["ev_final"] = mix * out["ev_struct"] + (1.0 - mix) * out["ret_pred"]

    if regime_prob is None:
        out["regime_prob"] = 0.5
    else:
        rp = np.asarray(regime_prob, dtype=float)
        if len(rp) != len(out):
            raise ValueError("regime_prob length must match df rows in simulate_policy.")
        out["regime_prob"] = np.clip(rp, 1e-4, 1.0 - 1e-4)

    regime_enabled = bool(params.get("regime_enable", False))
    regime_center = np.clip((out["regime_prob"].to_numpy(dtype=float) - 0.5) * 2.0, -1.0, 1.0)
    out["regime_scale"] = 1.0
    if regime_enabled:
        ev_scale = float(params.get("regime_ev_scale", 0.0))
        out["ev_final"] = out["ev_final"] + ev_scale * regime_center * out["cost_rt"]

    q10_cut = float(params.get("q10_cut", -1e9))
    agg_q10_cut = float(params.get("agg_q10_cut", q10_cut))
    tail_pass = out["q10_pred"] >= q10_cut
    agg_tail_pass = out["q10_pred"] >= agg_q10_cut
    trade = (out["p_pred"] >= float(params["p_cut"])) & (out["ev_final"] >= float(params["ev_cut"])) & tail_pass
    agg = trade & (out["p_pred"] >= float(params["agg_p_cut"])) & (
        out["ev_final"] >= float(params["agg_ev_cut"])
    ) & agg_tail_pass
    if regime_enabled:
        trade = trade & (out["regime_prob"] >= float(params.get("regime_p_cut", 0.5)))
        agg = agg & (out["regime_prob"] >= float(params.get("regime_agg_p_cut", 0.6)))

    out["tail_pass"] = tail_pass.astype(int)
    out["trade"] = trade.astype(int)
    out["mode"] = np.where(agg, "aggressive", np.where(trade, "safe", "flat"))
    out["size_mult"] = 0.0
    safe_base = float(params["safe_size"])
    agg_base = float(params.get("aggressive_size", 1.0))
    use_conf = bool(params.get("use_confidence_sizing", False)) and ("confidence" in out.columns)
    safe_mask = trade & (~agg)
    agg_mask = agg
    if use_conf:
        conf = np.clip(out["confidence"].to_numpy(dtype=float), 0.0, 1.0)
        safe_lo = float(params.get("safe_size_floor_mult", 0.70))
        safe_hi = float(params.get("safe_size_ceiling_mult", 1.20))
        agg_lo = float(params.get("agg_size_floor_mult", 0.80))
        agg_hi = float(params.get("agg_size_ceiling_mult", 1.25))
        safe_scale = safe_lo + (safe_hi - safe_lo) * conf
        agg_scale = agg_lo + (agg_hi - agg_lo) * conf
        safe_vals = np.clip(safe_base * safe_scale, 0.0, safe_base * safe_hi)
        agg_cap = float(params.get("max_aggressive_size", max(1.0, agg_base)))
        agg_vals = np.clip(agg_base * agg_scale, 0.0, agg_cap)
        out.loc[safe_mask, "size_mult"] = safe_vals[safe_mask.to_numpy(dtype=bool)]
        out.loc[agg_mask, "size_mult"] = agg_vals[agg_mask.to_numpy(dtype=bool)]
    else:
        out.loc[safe_mask, "size_mult"] = safe_base
        out.loc[agg_mask, "size_mult"] = agg_base
    if regime_enabled:
        reg_size_scale = float(params.get("regime_size_scale", 0.0))
        reg_min = float(params.get("regime_size_min_mult", 0.60))
        reg_max = float(params.get("regime_size_max_mult", 1.20))
        reg_mult = np.clip(1.0 + reg_size_scale * regime_center, reg_min, reg_max)
        out["regime_scale"] = reg_mult
        trade_mask_np = trade.to_numpy(dtype=bool)
        if trade_mask_np.any():
            cur = out.loc[trade, "size_mult"].to_numpy(dtype=float)
            out.loc[trade, "size_mult"] = cur * reg_mult[trade_mask_np]
    out["weighted_net_logret"] = out["size_mult"] * out["net_logret"]

    traded = out[out["trade"] == 1].copy()
    n_trades = int(len(traded))
    aggr_rate = float((traded["mode"] == "aggressive").mean()) if n_trades > 0 else 0.0
    perf = perf_from_trade_logrets(
        times=traded["exit_time_utc"] if n_trades > 0 else pd.Series(dtype="datetime64[ns, UTC]"),
        logrets=traded["weighted_net_logret"].to_numpy(dtype=float) if n_trades > 0 else np.array([]),
        n_trades=n_trades,
        aggressive_rate=aggr_rate,
    )
    perf["safe_size"] = float(params["safe_size"])
    perf["p_cut"] = float(params["p_cut"])
    perf["agg_p_cut"] = float(params["agg_p_cut"])
    perf["ev_cut"] = float(params["ev_cut"])
    perf["agg_ev_cut"] = float(params["agg_ev_cut"])
    perf["aggressive_size"] = float(params.get("aggressive_size", 1.0))
    perf["q10_cut"] = q10_cut
    perf["agg_q10_cut"] = agg_q10_cut
    perf["tail_pass_rate"] = float(np.mean(tail_pass.to_numpy(dtype=bool)))
    perf["regime_enable"] = bool(params.get("regime_enable", False))
    perf["regime_p_cut"] = float(params.get("regime_p_cut", 0.5))
    perf["regime_agg_p_cut"] = float(params.get("regime_agg_p_cut", 0.6))
    return traded, perf


def select_feature_cols(
    df: pd.DataFrame,
    candidate_cols: List[str],
    exclude_leaky_features: bool = False,
) -> List[str]:
    cols = [c for c in candidate_cols if c in df.columns]
    numeric = set(df.select_dtypes(include=[np.number]).columns.tolist())
    cols = [c for c in cols if c in numeric]
    cols = [c for c in cols if df[c].notna().mean() >= 0.35]
    if exclude_leaky_features:
        cols = [c for c in cols if c not in LEAKY_FEATURE_COLS]
    if len(cols) < 8:
        raise RuntimeError("Too few valid feature columns after NaN filtering.")
    return sorted(cols)


def tune_fold(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    mix_struct_weight: float,
    min_val_trades: int,
    policy_profile: str,
    max_aggressive_size: float,
    pattern_aid_enable: bool,
    pattern_n_clusters: int,
    pattern_min_cluster_samples: int,
    pattern_prior_strength: float,
    pattern_consistency_tol: float,
    pattern_prob_strength: float,
    pattern_ret_strength: float,
    pattern_prob_max_abs_delta: float,
    pattern_ret_max_abs_delta: float,
    regime_model_enable: bool,
    regime_lookahead_events: int,
    regime_label_quantile: float,
    regime_min_train_samples: int,
    regime_p_cut: float,
    regime_agg_p_cut: float,
    regime_ev_scale: float,
    regime_size_scale: float,
    regime_size_min_mult: float,
    regime_size_max_mult: float,
    tail_q10_cut: float,
    tail_agg_q10_cut: float,
) -> Tuple[Dict, Dict, List[Dict]]:
    cfg_grid = model_config_grid(policy_profile)
    inner_splits = make_inner_splits(
        fit_df=fit_df,
        min_fit_events=max(80, int(0.45 * len(fit_df))),
        min_val_events=max(25, int(0.12 * len(fit_df))),
        embargo_days=5,
    )
    if not inner_splits:
        raise RuntimeError("No valid inner splits for nested tuning.")

    nested_rows: List[Dict] = []
    best_cfg: Optional[Dict] = None
    best_cfg_score = float("-inf")

    for cfg in cfg_grid:
        split_scores: List[float] = []
        for i_split, (inner_fit, inner_val) in enumerate(inner_splits, start=1):
            if len(inner_fit) < 120 or len(inner_val) < 25:
                continue
            calib_i = max(20, int(0.20 * len(inner_fit)))
            core_train = inner_fit.iloc[:-calib_i].copy()
            calib_df = inner_fit.iloc[-calib_i:].copy()
            if len(core_train) < 80 or len(calib_df) < 15:
                continue
            try:
                bundle = fit_model_bundle(core_train, feature_cols, cfg)
                p_cal_raw, _r_cal_raw, _q10_cal, _q90_cal, u_cal = predict_model_bundle(calib_df, bundle)
                pattern_model = {"enabled": False}
                if pattern_aid_enable:
                    pattern_model = fit_pattern_context_model(
                        train_df=core_train,
                        pattern_cols=PATTERN_CONTEXT_COLS,
                        n_clusters=pattern_n_clusters,
                        min_cluster_samples=pattern_min_cluster_samples,
                        prior_strength=pattern_prior_strength,
                        consistency_tol=pattern_consistency_tol,
                    )
                    if bool(pattern_model.get("enabled")):
                        p_adj_cal = apply_pattern_context_model(
                            df=calib_df,
                            pattern_model=pattern_model,
                            prob_strength=pattern_prob_strength,
                            ret_strength=pattern_ret_strength,
                            prob_max_abs_delta=pattern_prob_max_abs_delta,
                            ret_max_abs_delta=pattern_ret_max_abs_delta,
                        )
                        p_cal_raw = np.clip(p_cal_raw + p_adj_cal["pattern_prob_delta"], 1e-4, 1.0 - 1e-4)

                regime_model = {"enabled": False}
                if regime_model_enable:
                    regime_model = fit_regime_classifier(
                        train_df=core_train,
                        candidate_cols=REGIME_CONTEXT_COLS,
                        lookahead_events=regime_lookahead_events,
                        label_quantile=regime_label_quantile,
                        min_train_samples=regime_min_train_samples,
                    )

                prob_cal = fit_probability_calibrator(
                    y_true=calib_df["y"].to_numpy(dtype=int),
                    p_raw=p_cal_raw,
                )
                unc_cal = fit_uncertainty_calibrator(u_cal)
                p_cal = apply_probability_calibrator(prob_cal, p_cal_raw)
                conf_cal_raw = confidence_from_predictions(p_cal, u_cal, unc_cal)
                conf_cal = fit_confidence_calibrator(
                    y_true=(calib_df["net_logret"].to_numpy(dtype=float) > 0.0).astype(int),
                    conf_raw=conf_cal_raw,
                )

                p_val_raw, r_val_raw, q10_val_raw, _q90_val_raw, u_val = predict_model_bundle(inner_val, bundle)
                if bool(pattern_model.get("enabled")):
                    p_adj_val = apply_pattern_context_model(
                        df=inner_val,
                        pattern_model=pattern_model,
                        prob_strength=pattern_prob_strength,
                        ret_strength=pattern_ret_strength,
                        prob_max_abs_delta=pattern_prob_max_abs_delta,
                        ret_max_abs_delta=pattern_ret_max_abs_delta,
                    )
                    p_val_raw = np.clip(p_val_raw + p_adj_val["pattern_prob_delta"], 1e-4, 1.0 - 1e-4)
                    r_val_raw = np.clip(r_val_raw + p_adj_val["pattern_ret_delta"], -0.03, 0.03)
                    q10_val_raw = np.clip(q10_val_raw + p_adj_val["pattern_ret_delta"], -0.05, 0.05)
                p_val = apply_probability_calibrator(prob_cal, p_val_raw)
                conf_val_raw = confidence_from_predictions(p_val, u_val, unc_cal)
                conf_val = apply_confidence_calibrator(conf_cal, conf_val_raw)
                val_eval = inner_val.copy()
                val_eval["confidence"] = conf_val
                regime_val = predict_regime_classifier(val_eval, regime_model)

                ev_struct = (
                    p_val * val_eval["a_tp"].to_numpy()
                    - (1.0 - p_val) * val_eval["b_sl"].to_numpy()
                    - val_eval["cost_rt"].to_numpy()
                )
                ev_final = mix_struct_weight * ev_struct + (1.0 - mix_struct_weight) * np.clip(r_val_raw, -0.03, 0.03)
                base_params = {
                    "mix_struct_weight": mix_struct_weight,
                    "p_cut": 0.50 if policy_profile == "growth" else 0.53,
                    "agg_p_cut": 0.62 if policy_profile == "growth" else 0.67,
                    "safe_size": 0.80 if policy_profile == "growth" else 0.60,
                    "aggressive_size": float(max(1.0, min(max_aggressive_size, 1.20))),
                    "ev_cut": float(np.quantile(ev_final, 0.48 if policy_profile == "growth" else 0.55)),
                    "agg_ev_cut": float(np.quantile(ev_final, 0.62 if policy_profile == "growth" else 0.70)),
                    "use_confidence_sizing": True,
                    "safe_size_floor_mult": 0.72,
                    "safe_size_ceiling_mult": 1.20,
                    "agg_size_floor_mult": 0.82,
                    "agg_size_ceiling_mult": 1.25,
                    "max_aggressive_size": max_aggressive_size,
                    "q10_cut": float(tail_q10_cut),
                    "agg_q10_cut": float(max(tail_q10_cut, tail_agg_q10_cut)),
                    "regime_enable": bool(regime_model_enable and regime_model.get("enabled", False)),
                    "regime_p_cut": float(regime_p_cut),
                    "regime_agg_p_cut": float(max(regime_agg_p_cut, regime_p_cut)),
                    "regime_ev_scale": float(regime_ev_scale),
                    "regime_size_scale": float(regime_size_scale),
                    "regime_size_min_mult": float(regime_size_min_mult),
                    "regime_size_max_mult": float(regime_size_max_mult),
                }
                _tr, perf = simulate_policy(
                    val_eval,
                    p_val,
                    r_val_raw,
                    base_params,
                    q10_pred=q10_val_raw,
                    regime_prob=regime_val,
                )
                perf_s = policy_score(perf, min_trades=max(4, min_val_trades // 2))
                brier = float(np.mean((p_val - val_eval["y"].to_numpy(dtype=float)) ** 2))
                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            val_eval["net_logret"].to_numpy(dtype=float),
                            np.clip(r_val_raw, -0.03, 0.03),
                        )
                    )
                )
                s = perf_s - 0.75 * brier - 2.0 * rmse
                split_scores.append(float(s))
                nested_rows.append(
                    {
                        "cfg_name": cfg["name"],
                        "inner_split": i_split,
                        "score": float(s),
                        "policy_score": float(perf_s),
                        "brier": brier,
                        "rmse": rmse,
                        "n_val": int(len(inner_val)),
                    }
                )
            except Exception:
                continue
        if not split_scores:
            continue
        mean_score = float(np.mean(split_scores))
        if mean_score > best_cfg_score:
            best_cfg_score = mean_score
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError("No valid model config from nested tuning.")

    calib_n = max(20, int(0.40 * len(val_df)))
    calib_n = min(calib_n, max(10, len(val_df) - 10))
    calib_outer = val_df.iloc[:calib_n].copy()
    tune_outer = val_df.iloc[calib_n:].copy()
    if len(tune_outer) < max(10, min_val_trades):
        tune_outer = val_df.copy()
        calib_n2 = max(20, int(0.20 * len(fit_df)))
        calib_outer = fit_df.iloc[-calib_n2:].copy()

    tuned_bundle = fit_model_bundle(fit_df, feature_cols, best_cfg)
    regime_model_outer = {"enabled": False}
    if regime_model_enable:
        regime_model_outer = fit_regime_classifier(
            train_df=fit_df,
            candidate_cols=REGIME_CONTEXT_COLS,
            lookahead_events=regime_lookahead_events,
            label_quantile=regime_label_quantile,
            min_train_samples=regime_min_train_samples,
        )
    p_cal_raw, _r_cal_raw, _q10_cal, _q90_cal, u_cal = predict_model_bundle(calib_outer, tuned_bundle)
    prob_cal = fit_probability_calibrator(calib_outer["y"].to_numpy(dtype=int), p_cal_raw)
    unc_cal = fit_uncertainty_calibrator(u_cal)
    p_cal = apply_probability_calibrator(prob_cal, p_cal_raw)
    conf_cal_raw = confidence_from_predictions(p_cal, u_cal, unc_cal)
    conf_cal = fit_confidence_calibrator(
        y_true=(calib_outer["net_logret"].to_numpy(dtype=float) > 0.0).astype(int),
        conf_raw=conf_cal_raw,
    )

    p_val_raw, r_val_raw, q10_val_raw, _q90_val_raw, u_val = predict_model_bundle(tune_outer, tuned_bundle)
    p_val = apply_probability_calibrator(prob_cal, p_val_raw)
    conf_val_raw = confidence_from_predictions(p_val, u_val, unc_cal)
    conf_val = apply_confidence_calibrator(conf_cal, conf_val_raw)
    tune_eval = tune_outer.copy()
    tune_eval["confidence"] = conf_val
    regime_val = predict_regime_classifier(tune_eval, regime_model_outer)
    ev_struct = p_val * tune_eval["a_tp"].to_numpy() - (1.0 - p_val) * tune_eval["b_sl"].to_numpy() - tune_eval["cost_rt"].to_numpy()
    ev_final = mix_struct_weight * ev_struct + (1.0 - mix_struct_weight) * np.clip(r_val_raw, -0.03, 0.03)

    best_params: Optional[Dict] = None
    best_threshold_score = -1e18
    if policy_profile == "growth":
        p_grid = [0.46, 0.50, 0.54, 0.58]
        ev_q_grid = [0.33, 0.43, 0.53]
        agg_p_grid = [0.58, 0.64, 0.70]
        safe_grid = [0.65, 0.82, 0.98]
    else:
        p_grid = [0.50, 0.53, 0.56, 0.60]
        ev_q_grid = [0.45, 0.55, 0.65]
        agg_p_grid = [0.62, 0.68, 0.74]
        safe_grid = [0.45, 0.60, 0.75]

    agg_size_grid = sorted(
        {
            1.0,
            float(max(1.0, min(max_aggressive_size, 1.12))),
            float(max(1.0, min(max_aggressive_size, 1.20))),
            float(max(1.0, min(max_aggressive_size, 1.35))),
        }
    )

    for p_cut in p_grid:
        for ev_q in ev_q_grid:
            ev_cut = float(np.quantile(ev_final, ev_q))
            for agg_p in agg_p_grid:
                for safe_size in safe_grid:
                    for aggressive_size in agg_size_grid:
                        params = {
                            "mix_struct_weight": mix_struct_weight,
                            "p_cut": float(p_cut),
                            "agg_p_cut": float(agg_p),
                            "safe_size": float(safe_size),
                            "aggressive_size": float(aggressive_size),
                            "ev_cut": ev_cut,
                            "agg_ev_cut": float(max(ev_cut, ev_cut + 0.0002)),
                            "use_confidence_sizing": True,
                            "safe_size_floor_mult": 0.70,
                            "safe_size_ceiling_mult": 1.22,
                            "agg_size_floor_mult": 0.80,
                            "agg_size_ceiling_mult": 1.28,
                            "max_aggressive_size": float(max_aggressive_size),
                            "q10_cut": float(tail_q10_cut),
                            "agg_q10_cut": float(max(tail_q10_cut, tail_agg_q10_cut)),
                            "regime_enable": bool(regime_model_enable and regime_model_outer.get("enabled", False)),
                            "regime_p_cut": float(regime_p_cut),
                            "regime_agg_p_cut": float(max(regime_agg_p_cut, regime_p_cut)),
                            "regime_ev_scale": float(regime_ev_scale),
                            "regime_size_scale": float(regime_size_scale),
                            "regime_size_min_mult": float(regime_size_min_mult),
                            "regime_size_max_mult": float(regime_size_max_mult),
                        }
                        _, perf = simulate_policy(
                            tune_eval,
                            p_val,
                            r_val_raw,
                            params,
                            q10_pred=q10_val_raw,
                            regime_prob=regime_val,
                        )
                        s = policy_score(perf, min_trades=max(1, min_val_trades // 2))
                        if s > best_threshold_score:
                            best_threshold_score = s
                            best_params = params
    if best_params is None:
        # Rare fallback: if strict gates leave too few validation trades, keep a
        # conservative deterministic policy so walk-forward can continue.
        ev_cut_fallback = float(np.quantile(ev_final, min(ev_q_grid)))
        best_params = {
            "mix_struct_weight": mix_struct_weight,
            "p_cut": float(min(p_grid)),
            "agg_p_cut": float(max(min(agg_p_grid), min(p_grid))),
            "safe_size": float(min(safe_grid)),
            "aggressive_size": float(min(agg_size_grid)),
            "ev_cut": ev_cut_fallback,
            "agg_ev_cut": float(max(ev_cut_fallback, ev_cut_fallback + 0.0002)),
            "use_confidence_sizing": True,
            "safe_size_floor_mult": 0.70,
            "safe_size_ceiling_mult": 1.22,
            "agg_size_floor_mult": 0.80,
            "agg_size_ceiling_mult": 1.28,
            "max_aggressive_size": float(max_aggressive_size),
            "q10_cut": float(tail_q10_cut),
            "agg_q10_cut": float(max(tail_q10_cut, tail_agg_q10_cut)),
            "regime_enable": bool(regime_model_enable and regime_model_outer.get("enabled", False)),
            "regime_p_cut": float(regime_p_cut),
            "regime_agg_p_cut": float(max(regime_agg_p_cut, regime_p_cut)),
            "regime_ev_scale": float(regime_ev_scale),
            "regime_size_scale": float(regime_size_scale),
            "regime_size_min_mult": float(regime_size_min_mult),
            "regime_size_max_mult": float(regime_size_max_mult),
        }

    nested_rows.append({"selected_cfg": best_cfg["name"], "nested_score": best_cfg_score})
    return best_cfg, best_params, nested_rows


def train_symbol_walkforward(
    df: pd.DataFrame,
    symbol: str,
    feature_cols: List[str],
    models_out_dir: str,
    train_lookback_days: int,
    min_train_events: int,
    min_val_events: int,
    min_test_events: int,
    embargo_days: int,
    mix_struct_weight: float,
    policy_profile: str,
    max_aggressive_size: float,
    retune_every_folds: int,
    spy_drift_kill_switch: str,
    spy_drift_feature_z_cap: float,
    drought_relief_enable: bool,
    drought_relief_symbol: str,
    drought_target_trades_per_month: float,
    drought_p_cut_relax: float,
    drought_ev_relax: float,
    drought_size_boost: float,
    pattern_aid_enable: bool,
    pattern_n_clusters: int,
    pattern_min_cluster_samples: int,
    pattern_prior_strength: float,
    pattern_consistency_tol: float,
    pattern_prob_strength: float,
    pattern_ret_strength: float,
    pattern_prob_max_abs_delta: float,
    pattern_ret_max_abs_delta: float,
    regime_model_enable: bool,
    regime_lookahead_events: int,
    regime_label_quantile: float,
    regime_min_train_samples: int,
    regime_p_cut: float,
    regime_agg_p_cut: float,
    regime_ev_scale: float,
    regime_size_scale: float,
    regime_size_min_mult: float,
    regime_size_max_mult: float,
    tail_q10_cut: float,
    tail_agg_q10_cut: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], Dict]:
    x = df.copy()
    x["decision_time_utc"] = pd.to_datetime(x["decision_time_utc"], utc=True, errors="coerce")
    x["exit_time_utc"] = pd.to_datetime(x["exit_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["decision_time_utc", "exit_time_utc"]).sort_values("decision_time_utc").reset_index(drop=True)

    symbol_dir = os.path.join(models_out_dir, symbol.lower())
    ensure_dir(symbol_dir)

    start_m = x["decision_time_utc"].min().floor("D")
    end_m = x["decision_time_utc"].max().floor("D")
    month_starts = pd.date_range(start_m, end_m, freq="MS", tz="UTC")

    all_scored: List[pd.DataFrame] = []
    all_trades: List[pd.DataFrame] = []
    fold_rows: List[Dict] = []
    active_model_cfg: Optional[Dict] = None
    active_best_params: Optional[Dict] = None

    for i in range(len(month_starts) - 1):
        test_start = month_starts[i]
        test_end = month_starts[i + 1]
        test_df = x[(x["decision_time_utc"] >= test_start) & (x["decision_time_utc"] < test_end)].copy()
        if len(test_df) < min_test_events:
            continue

        train_end = test_start - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.Timedelta(days=train_lookback_days)
        train_df = x[
            (x["decision_time_utc"] >= train_start)
            & (x["decision_time_utc"] < train_end)
            & (x["exit_time_utc"] < test_start)
        ].copy()
        train_df = purge_by_label_end(
            train_df,
            test_df,
            embargo_bars=estimate_embargo_bars(test_df),
        )
        if len(train_df) < min_train_events:
            continue
        train_df = train_df.sort_values("decision_time_utc").reset_index(drop=True)
        split_i = int(len(train_df) * 0.80)
        split_i = min(max(split_i, min_train_events - min_val_events), len(train_df) - min_val_events)
        fit_df = train_df.iloc[:split_i].copy()
        val_df = train_df.iloc[split_i:].copy()
        fit_df = purge_by_label_end(
            fit_df,
            val_df,
            embargo_bars=estimate_embargo_bars(val_df),
        )
        if len(fit_df) < (min_train_events - min_val_events) or len(val_df) < min_val_events:
            continue

        should_retune = (active_model_cfg is None) or (i % max(1, retune_every_folds) == 0)
        nested_rows_fold: List[Dict] = []
        if should_retune:
            try:
                tuned_cfg, tuned_params, nested_rows_fold = tune_fold(
                    fit_df=fit_df,
                    val_df=val_df,
                    feature_cols=feature_cols,
                    mix_struct_weight=mix_struct_weight,
                    min_val_trades=max(6, min_test_events // 2),
                    policy_profile=policy_profile,
                    max_aggressive_size=max_aggressive_size,
                    pattern_aid_enable=pattern_aid_enable,
                    pattern_n_clusters=pattern_n_clusters,
                    pattern_min_cluster_samples=pattern_min_cluster_samples,
                    pattern_prior_strength=pattern_prior_strength,
                    pattern_consistency_tol=pattern_consistency_tol,
                    pattern_prob_strength=pattern_prob_strength,
                    pattern_ret_strength=pattern_ret_strength,
                    pattern_prob_max_abs_delta=pattern_prob_max_abs_delta,
                    pattern_ret_max_abs_delta=pattern_ret_max_abs_delta,
                    regime_model_enable=regime_model_enable,
                    regime_lookahead_events=regime_lookahead_events,
                    regime_label_quantile=regime_label_quantile,
                    regime_min_train_samples=regime_min_train_samples,
                    regime_p_cut=regime_p_cut,
                    regime_agg_p_cut=regime_agg_p_cut,
                    regime_ev_scale=regime_ev_scale,
                    regime_size_scale=regime_size_scale,
                    regime_size_min_mult=regime_size_min_mult,
                    regime_size_max_mult=regime_size_max_mult,
                    tail_q10_cut=tail_q10_cut,
                    tail_agg_q10_cut=tail_agg_q10_cut,
                )
                active_model_cfg = tuned_cfg
                active_best_params = tuned_params
            except Exception:
                continue

        if active_model_cfg is None or active_best_params is None:
            continue

        calib_n = max(25, int(0.15 * len(train_df)))
        calib_n = min(calib_n, max(20, len(train_df) - 80))
        model_train = train_df.iloc[:-calib_n].copy()
        calib_df = train_df.iloc[-calib_n:].copy()
        if len(model_train) < 80 or len(calib_df) < 20:
            continue

        try:
            bundle = fit_model_bundle(model_train, feature_cols, active_model_cfg)
        except Exception:
            continue

        pattern_model = {"enabled": False}
        if pattern_aid_enable:
            pattern_model = fit_pattern_context_model(
                train_df=model_train,
                pattern_cols=PATTERN_CONTEXT_COLS,
                n_clusters=pattern_n_clusters,
                min_cluster_samples=pattern_min_cluster_samples,
                prior_strength=pattern_prior_strength,
                consistency_tol=pattern_consistency_tol,
            )
        regime_model = {"enabled": False}
        if regime_model_enable:
            regime_model = fit_regime_classifier(
                train_df=model_train,
                candidate_cols=REGIME_CONTEXT_COLS,
                lookahead_events=regime_lookahead_events,
                label_quantile=regime_label_quantile,
                min_train_samples=regime_min_train_samples,
            )

        p_cal_raw, _r_cal_raw, _q10_cal, _q90_cal, u_cal = predict_model_bundle(calib_df, bundle)
        p_adj_cal = None
        if bool(pattern_model.get("enabled")):
            p_adj_cal = apply_pattern_context_model(
                df=calib_df,
                pattern_model=pattern_model,
                prob_strength=pattern_prob_strength,
                ret_strength=pattern_ret_strength,
                prob_max_abs_delta=pattern_prob_max_abs_delta,
                ret_max_abs_delta=pattern_ret_max_abs_delta,
            )
            p_cal_raw = np.clip(p_cal_raw + p_adj_cal["pattern_prob_delta"], 1e-4, 1.0 - 1e-4)
        prob_cal = fit_probability_calibrator(
            y_true=calib_df["y"].to_numpy(dtype=int),
            p_raw=p_cal_raw,
        )
        unc_cal = fit_uncertainty_calibrator(u_cal)
        p_cal = apply_probability_calibrator(prob_cal, p_cal_raw)
        conf_cal_raw = confidence_from_predictions(p_cal, u_cal, unc_cal)
        conf_cal = fit_confidence_calibrator(
            y_true=(calib_df["net_logret"].to_numpy(dtype=float) > 0.0).astype(int),
            conf_raw=conf_cal_raw,
        )

        p_test_raw, r_test_raw, q10_test_raw, q90_test_raw, u_test = predict_model_bundle(test_df, bundle)
        p_adj_test = None
        if bool(pattern_model.get("enabled")):
            p_adj_test = apply_pattern_context_model(
                df=test_df,
                pattern_model=pattern_model,
                prob_strength=pattern_prob_strength,
                ret_strength=pattern_ret_strength,
                prob_max_abs_delta=pattern_prob_max_abs_delta,
                ret_max_abs_delta=pattern_ret_max_abs_delta,
            )
            p_test_raw = np.clip(p_test_raw + p_adj_test["pattern_prob_delta"], 1e-4, 1.0 - 1e-4)
            r_test_raw = np.clip(r_test_raw + p_adj_test["pattern_ret_delta"], -0.03, 0.03)
            q10_test_raw = np.clip(q10_test_raw + p_adj_test["pattern_ret_delta"], -0.05, 0.05)
            q90_test_raw = np.clip(q90_test_raw + p_adj_test["pattern_ret_delta"], -0.05, 0.05)
        p_test = apply_probability_calibrator(prob_cal, p_test_raw)
        confidence_test_raw = confidence_from_predictions(p_test, u_test, unc_cal)
        confidence_test = apply_confidence_calibrator(conf_cal, confidence_test_raw)
        regime_test = predict_regime_classifier(test_df, regime_model)

        drift_features = feature_drift_snapshot(train_df, test_df, feature_cols)
        drift_flag, drift_state = performance_drift_flag(fold_rows, drift_features)

        params_live = dict(active_best_params)
        params_live["use_confidence_sizing"] = True
        params_live["max_aggressive_size"] = float(max_aggressive_size)
        params_live["q10_cut"] = float(tail_q10_cut)
        params_live["agg_q10_cut"] = float(max(tail_q10_cut, tail_agg_q10_cut))
        params_live["regime_enable"] = bool(regime_model_enable and regime_model.get("enabled", False))
        params_live["regime_p_cut"] = float(regime_p_cut)
        params_live["regime_agg_p_cut"] = float(max(regime_agg_p_cut, regime_p_cut))
        params_live["regime_ev_scale"] = float(regime_ev_scale)
        params_live["regime_size_scale"] = float(regime_size_scale)
        params_live["regime_size_min_mult"] = float(regime_size_min_mult)
        params_live["regime_size_max_mult"] = float(regime_size_max_mult)
        if drift_flag:
            # Auto safety mode under drift: tighten entries and reduce leverage.
            params_live["p_cut"] = float(min(0.92, params_live["p_cut"] + 0.02))
            params_live["agg_p_cut"] = float(min(0.96, params_live["agg_p_cut"] + 0.03))
            params_live["safe_size"] = float(max(0.30, params_live["safe_size"] * 0.90))
            params_live["aggressive_size"] = float(max(0.75, params_live["aggressive_size"] * 0.88))
            params_live["agg_ev_cut"] = float(max(params_live["agg_ev_cut"], params_live["ev_cut"] + 0.0003))

        is_spy = symbol.upper() == "SPY"
        spy_kill_active = is_spy and (
            bool(drift_flag) or float(drift_features.get("mean_abs_z") or 0.0) >= float(spy_drift_feature_z_cap)
        )
        if spy_kill_active and spy_drift_kill_switch != "none":
            if spy_drift_kill_switch == "hard":
                params_live["p_cut"] = float(max(params_live["p_cut"], 0.69))
                params_live["agg_p_cut"] = 0.999
                params_live["safe_size"] = float(max(0.25, params_live["safe_size"] * 0.72))
                params_live["aggressive_size"] = float(max(0.50, params_live["safe_size"] * 0.90))
                params_live["ev_cut"] = float(max(params_live["ev_cut"], params_live["agg_ev_cut"]))
                params_live["agg_ev_cut"] = float(max(params_live["agg_ev_cut"], params_live["ev_cut"] + 0.0006))
            else:
                params_live["p_cut"] = float(min(0.95, params_live["p_cut"] + 0.04))
                params_live["agg_p_cut"] = float(min(0.99, params_live["agg_p_cut"] + 0.08))
                params_live["safe_size"] = float(max(0.28, params_live["safe_size"] * 0.82))
                params_live["aggressive_size"] = float(max(0.65, params_live["aggressive_size"] * 0.72))
                params_live["agg_ev_cut"] = float(max(params_live["agg_ev_cut"], params_live["ev_cut"] + 0.0005))

        drought_severity = 0.0
        relief_symbol_match = (
            drought_relief_symbol == "BOTH" or symbol.upper() == drought_relief_symbol
        )
        if (
            drought_relief_enable
            and relief_symbol_match
            and (not drift_flag)
            and (not spy_kill_active)
            and drought_target_trades_per_month > 0.0
        ):
            recent_rows = fold_rows[-5:] if len(fold_rows) >= 1 else []
            recent_tpm = [
                float(r.get("test_trades_per_month") or 0.0)
                for r in recent_rows
                if r.get("test_trades_per_month") is not None
            ]
            recent_med = float(np.median(recent_tpm)) if recent_tpm else 0.0
            drought_severity = float(
                np.clip(
                    (drought_target_trades_per_month - recent_med) / max(drought_target_trades_per_month, 1e-6),
                    0.0,
                    1.0,
                )
            )
            if drought_severity > 0.0:
                params_live["p_cut"] = float(max(0.42, params_live["p_cut"] - drought_p_cut_relax * drought_severity))
                params_live["agg_p_cut"] = float(
                    max(0.50, params_live["agg_p_cut"] - 0.5 * drought_p_cut_relax * drought_severity)
                )
                params_live["ev_cut"] = float(params_live["ev_cut"] - drought_ev_relax * drought_severity)
                params_live["safe_size"] = float(
                    min(1.25, params_live["safe_size"] * (1.0 + drought_size_boost * drought_severity))
                )

        test_eval = test_df.copy()
        test_eval["confidence"] = confidence_test
        test_eval["uncertainty"] = u_test
        if p_adj_test is not None:
            test_eval["pattern_cluster"] = p_adj_test["pattern_cluster"].astype(int)
            test_eval["pattern_prob_delta"] = p_adj_test["pattern_prob_delta"].astype(float)
            test_eval["pattern_ret_delta"] = p_adj_test["pattern_ret_delta"].astype(float)
            test_eval["pattern_consistency"] = p_adj_test["pattern_consistency"].astype(float)
            test_eval["pattern_support"] = p_adj_test["pattern_support"].astype(float)
        else:
            test_eval["pattern_cluster"] = -1
            test_eval["pattern_prob_delta"] = 0.0
            test_eval["pattern_ret_delta"] = 0.0
            test_eval["pattern_consistency"] = 0.0
            test_eval["pattern_support"] = 0.0
        traded, fold_perf = simulate_policy(
            test_eval,
            p_test,
            r_test_raw,
            params_live,
            q10_pred=q10_test_raw,
            regime_prob=regime_test,
        )
        fold_score = policy_score(fold_perf, min_trades=max(6, min_test_events // 2))
        pattern_active_rate = float((test_eval["pattern_cluster"].to_numpy(dtype=int) >= 0).mean())
        pattern_mean_abs_prob_delta = float(np.mean(np.abs(test_eval["pattern_prob_delta"].to_numpy(dtype=float))))
        pattern_mean_abs_ret_delta = float(np.mean(np.abs(test_eval["pattern_ret_delta"].to_numpy(dtype=float))))
        pattern_mean_consistency = float(np.mean(test_eval["pattern_consistency"].to_numpy(dtype=float)))

        pattern_meta = {
            "enabled": bool(pattern_model.get("enabled")),
            "prob_strength": float(pattern_prob_strength),
            "ret_strength": float(pattern_ret_strength),
            "prob_max_abs_delta": float(pattern_prob_max_abs_delta),
            "ret_max_abs_delta": float(pattern_ret_max_abs_delta),
        }
        if bool(pattern_model.get("enabled")):
            pattern_meta.update(
                {
                    "selected_cols": pattern_model.get("selected_cols"),
                    "n_clusters": int(pattern_model.get("n_clusters") or 0),
                    "global_win_rate": float(pattern_model.get("global_win_rate") or 0.0),
                    "global_ev": float(pattern_model.get("global_ev") or 0.0),
                    "min_cluster_samples": int(pattern_model.get("min_cluster_samples") or 0),
                    "prior_strength": float(pattern_model.get("prior_strength") or 0.0),
                    "consistency_tol": float(pattern_model.get("consistency_tol") or 0.0),
                    "cluster_rows": pattern_model.get("cluster_rows") or [],
                }
            )

        fold_id = f"{symbol.lower()}_{test_start.strftime('%Y%m')}"
        scored = test_eval.copy()
        scored["p_pred"] = p_test
        scored["ret_pred"] = np.clip(r_test_raw, -0.03, 0.03)
        scored["q10_pred"] = np.clip(q10_test_raw, -0.05, 0.05)
        scored["q90_pred"] = np.clip(q90_test_raw, -0.05, 0.05)
        scored["regime_prob"] = regime_test
        scored["regime_enabled"] = int(bool(params_live.get("regime_enable", False)))
        scored["drift_flag"] = int(drift_flag)
        scored["feature_drift_mean_abs_z"] = float(drift_features.get("mean_abs_z") or 0.0)
        scored["fold_id"] = fold_id
        scored["fold_test_start_utc"] = test_start
        scored["fold_test_end_utc"] = test_end
        all_scored.append(scored)

        if not traded.empty:
            traded["fold_id"] = fold_id
            traded["fold_test_start_utc"] = test_start
            traded["fold_test_end_utc"] = test_end
            traded["drift_flag"] = int(drift_flag)
            all_trades.append(traded)

        cls_path = os.path.join(symbol_dir, f"{fold_id}_lgb_cls.txt")
        ret_path = os.path.join(symbol_dir, f"{fold_id}_lgb_ret.txt")
        q50_path = os.path.join(symbol_dir, f"{fold_id}_lgb_q50.txt")
        q10_path = os.path.join(symbol_dir, f"{fold_id}_lgb_q10.txt")
        q90_path = os.path.join(symbol_dir, f"{fold_id}_lgb_q90.txt")
        bundle["lgb_cls"].booster_.save_model(cls_path)
        bundle["lgb_ret"].booster_.save_model(ret_path)
        bundle["lgb_q50"].booster_.save_model(q50_path)
        bundle["lgb_q10"].booster_.save_model(q10_path)
        bundle["lgb_q90"].booster_.save_model(q90_path)
        regime_path = None
        if bool(regime_model.get("enabled")):
            regime_path = os.path.join(symbol_dir, f"{fold_id}_regime_lgb_cls.txt")
            regime_model["model"].booster_.save_model(regime_path)

        if prob_cal.get("method") == "isotonic":
            cal_obj = {
                "method": "isotonic",
                "x_thresholds": [float(v) for v in prob_cal["model"].X_thresholds_],
                "y_thresholds": [float(v) for v in prob_cal["model"].y_thresholds_],
            }
        elif prob_cal.get("method") == "platt":
            cal_obj = {
                "method": "platt",
                "coef": [float(v) for v in prob_cal["model"].coef_.ravel()],
                "intercept": [float(v) for v in np.atleast_1d(prob_cal["model"].intercept_)],
            }
        else:
            cal_obj = {"method": "identity"}

        if conf_cal.get("method") == "isotonic":
            conf_cal_obj = {
                "method": "isotonic",
                "x_thresholds": [float(v) for v in conf_cal["model"].X_thresholds_],
                "y_thresholds": [float(v) for v in conf_cal["model"].y_thresholds_],
            }
        else:
            conf_cal_obj = {"method": "identity"}

        regime_cal = regime_model.get("calibrator") or {"method": "identity"}
        if regime_cal.get("method") == "isotonic":
            regime_cal_obj = {
                "method": "isotonic",
                "x_thresholds": [float(v) for v in regime_cal["model"].X_thresholds_],
                "y_thresholds": [float(v) for v in regime_cal["model"].y_thresholds_],
            }
        elif regime_cal.get("method") == "platt":
            regime_cal_obj = {
                "method": "platt",
                "coef": [float(v) for v in regime_cal["model"].coef_.ravel()],
                "intercept": [float(v) for v in np.atleast_1d(regime_cal["model"].intercept_)],
            }
        else:
            regime_cal_obj = {"method": "identity"}

        model_blob = {
            "fold_id": fold_id,
            "symbol": symbol,
            "test_start_utc": str(test_start),
            "test_end_utc": str(test_end),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_trades_test": int(fold_perf["n"]),
            "policy_score_test": float(fold_score),
            "best_params": params_live,
            "base_params_before_drift_guard": active_best_params,
            "model_config": active_model_cfg,
            "ridge_prob_model": bundle["ridge_prob"],
            "ridge_ret_model": bundle["ridge_ret"],
            "lgb_paths": {
                "classifier": cls_path,
                "regressor": ret_path,
                "quantile_q50": q50_path,
                "quantile_q10": q10_path,
                "quantile_q90": q90_path,
                "regime_classifier": regime_path,
            },
            "probability_calibration": cal_obj,
            "confidence_calibration": conf_cal_obj,
            "uncertainty_calibration": unc_cal,
            "nested_tuning_rows": nested_rows_fold,
            "drift": {
                "flag": bool(drift_flag),
                "feature_drift": drift_features,
                "performance_drift": drift_state,
            },
            "pattern_aid": pattern_meta,
            "regime_model": {
                "enabled": bool(regime_model.get("enabled")),
                "selected_cols": regime_model.get("selected_cols"),
                "lookahead_events": regime_model.get("lookahead_events"),
                "label_quantile": regime_model.get("label_quantile"),
                "threshold": regime_model.get("threshold"),
                "n_train": regime_model.get("n_train"),
                "positive_rate": regime_model.get("positive_rate"),
                "brier_in_sample": regime_model.get("brier_in_sample"),
                "calibration": regime_cal_obj,
                "policy_params": {
                    "regime_p_cut": float(params_live.get("regime_p_cut", 0.5)),
                    "regime_agg_p_cut": float(params_live.get("regime_agg_p_cut", 0.6)),
                    "regime_ev_scale": float(params_live.get("regime_ev_scale", 0.0)),
                    "regime_size_scale": float(params_live.get("regime_size_scale", 0.0)),
                    "regime_size_min_mult": float(params_live.get("regime_size_min_mult", 0.6)),
                    "regime_size_max_mult": float(params_live.get("regime_size_max_mult", 1.2)),
                },
            },
            "feature_cols": feature_cols,
        }
        model_path = os.path.join(symbol_dir, f"{fold_id}_model.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(round_obj(model_blob, 8), f, separators=(",", ":"), ensure_ascii=True)

        fold_rows.append(
            {
                "fold_id": fold_id,
                "symbol": symbol,
                "test_start_utc": str(test_start),
                "test_end_utc": str(test_end),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_trades_test": int(fold_perf["n"]),
                "test_end_equity": fold_perf["end_equity"],
                "test_calmar": fold_perf["calmar"],
                "test_max_drawdown": fold_perf["max_drawdown"],
                "test_net_bps_mean": fold_perf["net_bps_mean"],
                "test_trades_per_month": fold_perf["trades_per_month"],
                "test_aggressive_rate": fold_perf["aggressive_rate"],
                "test_tail_pass_rate": fold_perf.get("tail_pass_rate"),
                "q10_cut": fold_perf.get("q10_cut"),
                "agg_q10_cut": fold_perf.get("agg_q10_cut"),
                "policy_score_test": fold_score,
                "drift_flag": bool(drift_flag),
                "feature_drift_mean_abs_z": float(drift_features.get("mean_abs_z") or 0.0),
                "spy_kill_active": bool(spy_kill_active),
                "drought_relief_severity": float(drought_severity),
                "pattern_aid_enabled": bool(pattern_meta.get("enabled")),
                "pattern_active_rate": pattern_active_rate,
                "pattern_mean_abs_prob_delta": pattern_mean_abs_prob_delta,
                "pattern_mean_abs_ret_delta": pattern_mean_abs_ret_delta,
                "pattern_mean_consistency": pattern_mean_consistency,
                "regime_enabled": bool(params_live.get("regime_enable", False)),
                "regime_mean_prob": float(np.mean(regime_test)),
                "regime_trade_pass_rate": float(
                    np.mean(regime_test >= float(params_live.get("regime_p_cut", 0.5)))
                ),
                "nested_selected_cfg": active_model_cfg.get("name"),
                "model_path": model_path,
            }
        )

    if not fold_rows:
        raise RuntimeError(f"No valid walk-forward folds for {symbol}.")

    scored_df = pd.concat(all_scored, ignore_index=True).sort_values("decision_time_utc").reset_index(drop=True)
    trades_df = (
        pd.concat(all_trades, ignore_index=True).sort_values("exit_time_utc").reset_index(drop=True)
        if all_trades
        else pd.DataFrame()
    )
    fold_df = pd.DataFrame(fold_rows).sort_values("test_start_utc").reset_index(drop=True)

    symbol_perf = (
        perf_from_trade_logrets(
            times=trades_df["exit_time_utc"],
            logrets=trades_df["weighted_net_logret"].to_numpy(dtype=float),
            n_trades=int(len(trades_df)),
            aggressive_rate=float((trades_df["mode"] == "aggressive").mean()) if not trades_df.empty else 0.0,
        )
        if not trades_df.empty
        else perf_from_trade_logrets(
            times=pd.Series(dtype="datetime64[ns, UTC]"),
            logrets=np.array([]),
            n_trades=0,
            aggressive_rate=0.0,
        )
    )
    symbol_summary = {
        "symbol": symbol,
        "n_folds": int(len(fold_df)),
        "n_scored_events": int(len(scored_df)),
        "n_trades": int(symbol_perf["n"]),
        "aggressive_rate": symbol_perf["aggressive_rate"],
        "fold_stability": {
            "median_test_calmar": float(
                pd.to_numeric(fold_df["test_calmar"], errors="coerce").median()
            ),
            "p25_test_calmar": float(
                pd.to_numeric(fold_df["test_calmar"], errors="coerce").quantile(0.25)
            ),
            "negative_test_calmar_rate": float(
                (
                    pd.to_numeric(fold_df["test_calmar"], errors="coerce")
                    .fillna(-1.0)
                    .lt(0.0)
                ).mean()
            ),
            "median_test_trades": float(
                pd.to_numeric(fold_df["n_trades_test"], errors="coerce").median()
            ),
            "drift_trigger_rate": float(
                pd.to_numeric(fold_df.get("drift_flag"), errors="coerce").fillna(0.0).mean()
            ),
            "pattern_active_rate": float(
                pd.to_numeric(fold_df.get("pattern_active_rate"), errors="coerce").fillna(0.0).mean()
            ),
            "pattern_mean_abs_prob_delta": float(
                pd.to_numeric(fold_df.get("pattern_mean_abs_prob_delta"), errors="coerce").fillna(0.0).mean()
            ),
            "pattern_mean_abs_ret_delta": float(
                pd.to_numeric(fold_df.get("pattern_mean_abs_ret_delta"), errors="coerce").fillna(0.0).mean()
            ),
            "regime_enabled_rate": float(
                pd.to_numeric(fold_df.get("regime_enabled"), errors="coerce").fillna(0.0).mean()
            ),
            "regime_mean_prob": float(
                pd.to_numeric(fold_df.get("regime_mean_prob"), errors="coerce").fillna(0.5).mean()
            ),
            "regime_trade_pass_rate": float(
                pd.to_numeric(fold_df.get("regime_trade_pass_rate"), errors="coerce").fillna(0.5).mean()
            ),
            "tail_gate_pass_rate": float(
                pd.to_numeric(fold_df.get("test_tail_pass_rate"), errors="coerce").fillna(0.0).mean()
            ),
        },
        "perf": symbol_perf,
    }
    return scored_df, trades_df, fold_rows, symbol_summary


def daily_equity_from_trades(
    trades: pd.DataFrame,
    start_capital: float,
    logret_col: str = "weighted_net_logret",
) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    x = trades.sort_values("exit_time_utc").copy()
    x["exit_time_utc"] = pd.to_datetime(x["exit_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["exit_time_utc"])
    if logret_col not in x.columns:
        return pd.Series(dtype=float)
    if x.empty:
        return pd.Series(dtype=float)
    eq = start_capital * np.exp(x[logret_col].to_numpy(dtype=float).cumsum())
    return pd.Series(eq, index=x["exit_time_utc"]).resample("D").last()


def flat_equity_from_scored(scored: pd.DataFrame) -> pd.Series:
    if scored.empty or "decision_time_utc" not in scored.columns:
        return pd.Series(dtype=float)
    d = pd.to_datetime(scored["decision_time_utc"], utc=True, errors="coerce").dropna()
    if d.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(d.min().floor("D"), d.max().floor("D"), freq="D", tz="UTC")
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0, index=idx, dtype=float)


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
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1.0 / 12.0)
    end_ratio = max(float(e[-1]) / float(start_capital), 1e-12)
    cagr = np.exp(np.clip(np.log(end_ratio) / years, -20.0, 20.0)) - 1.0
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


def with_stressed_trade_cost(trades: pd.DataFrame, cost_multiplier: float) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    x = trades.copy()
    gross = x["gross_logret"].to_numpy(dtype=float)
    cost = x["cost_rt"].to_numpy(dtype=float)
    size = x["size_mult"].to_numpy(dtype=float)
    net_stress = gross - float(cost_multiplier) * cost
    x["net_logret_stress"] = net_stress
    x["weighted_net_logret_stress"] = size * net_stress
    return x


def portfolio_cost_stress_summary(
    idx: pd.DatetimeIndex,
    q_trades: pd.DataFrame,
    s_trades: pd.DataFrame,
    q_weight: pd.Series,
    spy_weight: pd.Series,
    start_capital: float,
    cost_multiplier: float,
) -> Dict:
    q_stress = with_stressed_trade_cost(q_trades, cost_multiplier=cost_multiplier)
    s_stress = with_stressed_trade_cost(s_trades, cost_multiplier=cost_multiplier)

    eq_q_unit = daily_equity_from_trades(
        q_stress,
        start_capital=1.0,
        logret_col="weighted_net_logret_stress",
    )
    eq_s_unit = daily_equity_from_trades(
        s_stress,
        start_capital=1.0,
        logret_col="weighted_net_logret_stress",
    )
    if eq_q_unit.empty or eq_s_unit.empty:
        return {
            "cost_multiplier": float(cost_multiplier),
            "error": "empty_stress_equity",
        }

    eq_q_unit = eq_q_unit.reindex(idx).ffill().fillna(1.0)
    eq_s_unit = eq_s_unit.reindex(idx).ffill().fillna(1.0)
    q_ret = eq_q_unit.pct_change().fillna(0.0)
    s_ret = eq_s_unit.pct_change().fillna(0.0)
    r_dual = q_weight * q_ret + spy_weight * s_ret
    eq_dual = start_capital * (1.0 + r_dual).cumprod()
    perf = perf_from_equity_series(eq_dual, start_capital=start_capital)

    all_trades = pd.concat([q_stress, s_stress], ignore_index=True).sort_values("exit_time_utc").reset_index(drop=True)
    monthly = monthly_table(eq_dual, all_trades, start_capital=start_capital)
    avg_monthly_pnl = float(monthly["pnl"].mean()) if not monthly.empty else 0.0
    med_monthly_pnl = float(monthly["pnl"].median()) if not monthly.empty else 0.0
    pos_rate = float((monthly["pnl"] > 0.0).mean()) if not monthly.empty else 0.0

    return {
        "cost_multiplier": float(cost_multiplier),
        "dual_perf": perf,
        "avg_monthly_pnl": avg_monthly_pnl,
        "median_monthly_pnl": med_monthly_pnl,
        "monthly_positive_rate": pos_rate,
    }


def monthly_block_bootstrap_summary(
    monthly: pd.DataFrame,
    start_capital: float,
    n_samples: int,
    block_months: int,
    seed: int,
) -> Dict:
    if monthly.empty:
        return {"enabled": False, "reason": "empty_monthly_table"}
    if n_samples <= 0:
        return {"enabled": False, "reason": "n_samples<=0"}

    rets = monthly["ret"].to_numpy(dtype=float)
    pnls = monthly["pnl"].to_numpy(dtype=float)
    n = len(monthly)
    block = int(max(1, block_months))
    rng = np.random.default_rng(int(seed))

    avg_pnl_arr = np.zeros(n_samples, dtype=float)
    med_pnl_arr = np.zeros(n_samples, dtype=float)
    pos_rate_arr = np.zeros(n_samples, dtype=float)
    end_eq_arr = np.zeros(n_samples, dtype=float)
    mdd_arr = np.zeros(n_samples, dtype=float)
    calmar_arr = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        picks: List[int] = []
        while len(picks) < n:
            st = int(rng.integers(0, n))
            for j in range(block):
                picks.append((st + j) % n)
                if len(picks) >= n:
                    break
        ix = np.array(picks[:n], dtype=int)
        r = rets[ix]
        p = pnls[ix]
        eq = start_capital * np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(eq)
        dd = 1.0 - (eq / np.maximum(peak, 1e-12))
        mdd = float(np.max(dd))
        years = max(n / 12.0, 1e-9)
        end_ratio = max(float(eq[-1]) / float(start_capital), 1e-12)
        cagr = np.exp(np.clip(np.log(end_ratio) / years, -20.0, 20.0)) - 1.0
        calmar = float(cagr / mdd) if mdd > 0.0 else 0.0

        avg_pnl_arr[i] = float(np.mean(p))
        med_pnl_arr[i] = float(np.median(p))
        pos_rate_arr[i] = float(np.mean(p > 0.0))
        end_eq_arr[i] = float(eq[-1])
        mdd_arr[i] = mdd
        calmar_arr[i] = calmar

    def q(arr: np.ndarray) -> Dict:
        return {
            "p05": float(np.quantile(arr, 0.05)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "mean": float(np.mean(arr)),
        }

    return {
        "enabled": True,
        "n_samples": int(n_samples),
        "block_months": int(block),
        "seed": int(seed),
        "avg_monthly_pnl": q(avg_pnl_arr),
        "median_monthly_pnl": q(med_pnl_arr),
        "monthly_positive_rate": q(pos_rate_arr),
        "end_equity": q(end_eq_arr),
        "max_drawdown": q(mdd_arr),
        "calmar": q(calmar_arr),
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
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    weights_grid = np.arange(min_weight, max_weight + 1e-12, weight_step)
    w_series = pd.Series(np.nan, index=idx, dtype=float)

    warmup_days = max(min_train_days, int(lookback_days * train_ratio))
    warmup_days = min(warmup_days, max(1, len(idx) - 1))

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
            w_raw = float(np.clip(w_opt + tilt, min_weight, max_weight))
            w_final = float(weight_smoothing * prev_w + (1.0 - weight_smoothing) * w_raw)

        w_series.iloc[month_idx] = float(np.clip(w_final, min_weight, max_weight))
        prev_w = float(np.clip(w_final, min_weight, max_weight))

    return w_series.fillna(0.5)


def build_static_weight_series(
    idx: pd.DatetimeIndex,
    q_ret: pd.Series,
    s_ret: pd.Series,
    train_ratio: float,
    weight_step: float,
    objective: str,
    min_weight: float,
    max_weight: float,
) -> Tuple[pd.Series, int]:
    split_i = int(len(idx) * train_ratio)
    split_i = min(max(split_i, 2), len(idx) - 1)
    train_slice = slice(0, split_i)
    weights_grid = np.arange(min_weight, max_weight + 1e-12, weight_step)

    best_w = 0.5
    best_score = -1e18
    for w in weights_grid:
        r_train = w * q_ret.iloc[train_slice] + (1.0 - w) * s_ret.iloc[train_slice]
        score = score_returns(r_train, objective)
        if score > best_score:
            best_score = score
            best_w = float(w)
    return pd.Series(best_w, index=idx, dtype=float), split_i


def monthly_table(eq: pd.Series, trades: pd.DataFrame, start_capital: float) -> pd.DataFrame:
    if eq.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    month_eq = eq.resample("ME").last()
    prev = month_eq.shift(1)
    prev.iloc[0] = start_capital
    pnl = month_eq - prev
    ret = month_eq / prev - 1.0
    if trades.empty:
        trades_m = pd.Series(0.0, index=month_eq.index)
    else:
        t = trades.copy()
        t["exit_time_utc"] = pd.to_datetime(t["exit_time_utc"], utc=True, errors="coerce")
        trades_m = t.set_index("exit_time_utc").resample("ME").size().reindex(month_eq.index).fillna(0.0)
    return pd.DataFrame(
        {
            "month_end": month_eq.index,
            "equity": month_eq.to_numpy(dtype=float),
            "pnl": pnl.to_numpy(dtype=float),
            "ret": ret.to_numpy(dtype=float),
            "trades": trades_m.to_numpy(dtype=float),
        }
    )


def plot_dual(
    eq_dual: pd.Series,
    eq_q: pd.Series,
    eq_s: pd.Series,
    q_weight: pd.Series,
    monthly: pd.DataFrame,
    out_path: str,
    start_capital: float,
    stats_text: str,
    symbol_1_label: str = "QQQ",
    symbol_2_label: str = "SPY",
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [3.2, 1.4, 1.1]},
    )
    ax1.plot(eq_dual.index, eq_dual.values, color="#1d3557", linewidth=2.2, label="Dual ML portfolio")
    ax1.plot(
        eq_q.index,
        eq_q.values,
        color="#457b9d",
        linewidth=1.2,
        alpha=0.7,
        label=f"{symbol_1_label} model only",
    )
    ax1.plot(
        eq_s.index,
        eq_s.values,
        color="#2a9d8f",
        linewidth=1.2,
        alpha=0.7,
        label=f"{symbol_2_label} model only",
    )
    ax1.axhline(start_capital, color="black", linewidth=0.8, alpha=0.6, linestyle="--", label="Start capital")
    ax1.set_title(f"Step 3 Real ML Backtest: {symbol_2_label} + {symbol_1_label}")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.22)
    ax1.legend()
    ax1b = ax1.twinx()
    ax1b.plot(
        q_weight.index,
        q_weight.values,
        color="#264653",
        linewidth=1.1,
        alpha=0.35,
        label=f"{symbol_1_label} weight",
    )
    ax1b.set_ylim(0.0, 1.0)
    ax1b.set_ylabel(f"{symbol_1_label} Weight Share")
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
    ax2.set_title("Monthly Profit / Loss")
    ax2.set_ylabel("Monthly PnL ($)")
    ax2.grid(alpha=0.2)

    trades = monthly["trades"].to_numpy(dtype=float)
    ax3.plot(monthly["month_end"], trades, color="#264653", linewidth=1.4, label="Trades per month")
    ax3.fill_between(monthly["month_end"], trades, 0.0, color="#264653", alpha=0.12)
    ax3.set_title("Trading Activity")
    ax3.set_ylabel("# Trades")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to step3_out/dataset")
    ap.add_argument("--out-dir", required=True, help="Path to step3_out")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument(
        "--display-symbol-1",
        default="QQQ",
        help="Display label for the primary sleeve (dataset alias: QQQ).",
    )
    ap.add_argument(
        "--display-symbol-2",
        default="SPY",
        help="Display label for the secondary sleeve (dataset alias: SPY).",
    )
    ap.add_argument("--train-lookback-days", type=int, default=1095)
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--min-train-events", type=int, default=200)
    ap.add_argument("--min-val-events", type=int, default=40)
    ap.add_argument("--min-test-events", type=int, default=6)
    ap.add_argument("--mix-struct-weight", type=float, default=0.65, help="Weight on structural EV head")
    ap.add_argument(
        "--retune-every-folds",
        type=int,
        default=3,
        help="Nested model retuning cadence in walk-forward folds (1=tune every fold).",
    )
    ap.add_argument(
        "--policy-profile",
        choices=["balanced", "growth"],
        default="growth",
        help="Growth profile accepts more risk/trade frequency; balanced is more conservative.",
    )
    ap.add_argument("--max-aggressive-size", type=float, default=1.20, help="Cap for aggressive size multiplier.")
    ap.add_argument(
        "--portfolio-allocator",
        choices=["equal_split", "dynamic_regime", "dynamic_regime_forced"],
        default="dynamic_regime",
        help="Portfolio allocation layer across QQQ/SPY strategy returns.",
    )
    ap.add_argument("--portfolio-objective", choices=["calmar", "end_equity"], default="calmar")
    ap.add_argument("--portfolio-train-ratio", type=float, default=0.60)
    ap.add_argument("--portfolio-weight-step", type=float, default=0.05)
    ap.add_argument("--portfolio-lookback-days", type=int, default=756)
    ap.add_argument("--portfolio-min-train-days", type=int, default=252)
    ap.add_argument("--portfolio-turnover-penalty", type=float, default=0.04)
    ap.add_argument("--portfolio-weight-smoothing", type=float, default=0.30)
    ap.add_argument("--portfolio-momentum-days", type=int, default=63)
    ap.add_argument("--portfolio-vol-days", type=int, default=21)
    ap.add_argument("--portfolio-vol-penalty", type=float, default=1.4)
    ap.add_argument("--portfolio-max-tilt", type=float, default=0.20)
    ap.add_argument("--portfolio-min-weight", type=float, default=0.05)
    ap.add_argument("--portfolio-max-weight", type=float, default=0.95)
    ap.add_argument(
        "--portfolio-no-spy-guard",
        action="store_true",
        help="Disable SPY regime guard (enabled by default).",
    )
    ap.add_argument("--portfolio-spy-guard-lookback-days", type=int, default=63)
    ap.add_argument("--portfolio-spy-guard-drift-lookback-days", type=int, default=42)
    ap.add_argument("--portfolio-spy-guard-min-mult", type=float, default=0.25)
    ap.add_argument("--portfolio-spy-guard-dd-penalty", type=float, default=1.2)
    ap.add_argument(
        "--spy-drift-kill-switch",
        choices=["none", "soft", "hard"],
        default="soft",
        help="SPY-only extra safety guard under feature/performance drift.",
    )
    ap.add_argument(
        "--spy-drift-feature-z-cap",
        type=float,
        default=1.25,
        help="SPY kill-switch activates when feature drift mean_abs_z >= this cap.",
    )
    ap.add_argument(
        "--drought-relief-enable",
        action="store_true",
        help="Enable controlled threshold relaxation when trailing trades are too low.",
    )
    ap.add_argument(
        "--drought-relief-symbol",
        choices=["QQQ", "SPY", "BOTH"],
        default="QQQ",
        help="Symbol(s) eligible for drought relief.",
    )
    ap.add_argument("--drought-target-trades-per-month", type=float, default=5.5)
    ap.add_argument("--drought-p-cut-relax", type=float, default=0.015)
    ap.add_argument("--drought-ev-relax", type=float, default=0.00018)
    ap.add_argument("--drought-size-boost", type=float, default=0.08)
    ap.add_argument(
        "--pattern-aid-enable",
        action="store_true",
        help="Enable cross-asset pattern context model as a secondary ML aid.",
    )
    ap.add_argument("--pattern-n-clusters", type=int, default=6)
    ap.add_argument("--pattern-min-cluster-samples", type=int, default=40)
    ap.add_argument("--pattern-prior-strength", type=float, default=120.0)
    ap.add_argument("--pattern-consistency-tol", type=float, default=0.004)
    ap.add_argument("--pattern-prob-strength", type=float, default=0.55)
    ap.add_argument("--pattern-ret-strength", type=float, default=0.65)
    ap.add_argument("--pattern-prob-max-abs-delta", type=float, default=0.04)
    ap.add_argument("--pattern-ret-max-abs-delta", type=float, default=0.0035)
    ap.add_argument(
        "--exclude-leaky-features",
        action="store_true",
        help="Exclude post-outcome features that are not available at decision time.",
    )
    ap.add_argument(
        "--regime-model-enable",
        action="store_true",
        help="Enable third-model regime classifier to gate/scale trades.",
    )
    ap.add_argument("--regime-lookahead-events", type=int, default=18)
    ap.add_argument("--regime-label-quantile", type=float, default=0.55)
    ap.add_argument("--regime-min-train-samples", type=int, default=160)
    ap.add_argument("--regime-p-cut", type=float, default=0.52)
    ap.add_argument("--regime-agg-p-cut", type=float, default=0.62)
    ap.add_argument("--regime-ev-scale", type=float, default=0.80)
    ap.add_argument("--regime-size-scale", type=float, default=0.30)
    ap.add_argument("--regime-size-min-mult", type=float, default=0.60)
    ap.add_argument("--regime-size-max-mult", type=float, default=1.20)
    ap.add_argument(
        "--tail-q10-cut",
        type=float,
        default=-0.004,
        help="Tail-risk gate: require predicted q10(net_logret) >= this value to trade.",
    )
    ap.add_argument(
        "--tail-agg-q10-cut",
        type=float,
        default=0.0,
        help="Optional stricter q10 cut for aggressive trades (must be >= --tail-q10-cut).",
    )
    ap.add_argument(
        "--cost-stress-multipliers",
        default="1.25,1.50",
        help="Comma-separated cost multipliers for robustness checks (e.g. 1.25,1.50).",
    )
    ap.add_argument("--bootstrap-samples", type=int, default=800, help="Monthly block-bootstrap sample count.")
    ap.add_argument("--bootstrap-block-months", type=int, default=6, help="Monthly block length for bootstrap.")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 < args.portfolio_train_ratio < 1.0):
        raise ValueError("--portfolio-train-ratio must be in (0, 1)")
    if not (0.0 < args.portfolio_weight_step <= 1.0):
        raise ValueError("--portfolio-weight-step must be in (0, 1]")
    if not (0.0 <= args.portfolio_weight_smoothing <= 1.0):
        raise ValueError("--portfolio-weight-smoothing must be in [0, 1]")
    if not (0.0 <= args.portfolio_min_weight < args.portfolio_max_weight <= 1.0):
        raise ValueError("--portfolio-min-weight / --portfolio-max-weight must satisfy 0 <= min < max <= 1")
    if not (0.0 <= args.portfolio_spy_guard_min_mult <= 1.0):
        raise ValueError("--portfolio-spy-guard-min-mult must be in [0, 1]")
    if args.max_aggressive_size < 1.0 or args.max_aggressive_size > 1.8:
        raise ValueError("--max-aggressive-size should be in [1.0, 1.8]")
    if args.retune_every_folds < 1:
        raise ValueError("--retune-every-folds must be >= 1")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be >= 0")
    if args.bootstrap_block_months < 1:
        raise ValueError("--bootstrap-block-months must be >= 1")
    if args.pattern_n_clusters < 2 or args.pattern_n_clusters > 12:
        raise ValueError("--pattern-n-clusters must be in [2, 12]")
    if args.pattern_min_cluster_samples < 12:
        raise ValueError("--pattern-min-cluster-samples must be >= 12")
    if args.pattern_prior_strength <= 0.0:
        raise ValueError("--pattern-prior-strength must be > 0")
    if args.pattern_consistency_tol <= 0.0:
        raise ValueError("--pattern-consistency-tol must be > 0")
    if args.pattern_prob_strength < 0.0 or args.pattern_ret_strength < 0.0:
        raise ValueError("--pattern-prob-strength/--pattern-ret-strength must be >= 0")
    if args.pattern_prob_max_abs_delta <= 0.0 or args.pattern_ret_max_abs_delta <= 0.0:
        raise ValueError("--pattern-prob-max-abs-delta/--pattern-ret-max-abs-delta must be > 0")
    if args.regime_lookahead_events < 6 or args.regime_lookahead_events > 96:
        raise ValueError("--regime-lookahead-events must be in [6, 96]")
    if args.regime_label_quantile <= 0.0 or args.regime_label_quantile >= 1.0:
        raise ValueError("--regime-label-quantile must be in (0, 1)")
    if args.regime_min_train_samples < 80:
        raise ValueError("--regime-min-train-samples must be >= 80")
    if args.regime_p_cut <= 0.0 or args.regime_p_cut >= 1.0:
        raise ValueError("--regime-p-cut must be in (0, 1)")
    if args.regime_agg_p_cut <= 0.0 or args.regime_agg_p_cut >= 1.0:
        raise ValueError("--regime-agg-p-cut must be in (0, 1)")
    if args.regime_agg_p_cut < args.regime_p_cut:
        raise ValueError("--regime-agg-p-cut must be >= --regime-p-cut")
    if args.regime_ev_scale < 0.0 or args.regime_ev_scale > 3.0:
        raise ValueError("--regime-ev-scale must be in [0, 3]")
    if args.regime_size_scale < 0.0 or args.regime_size_scale > 1.5:
        raise ValueError("--regime-size-scale must be in [0, 1.5]")
    if args.regime_size_min_mult <= 0.0 or args.regime_size_max_mult <= 0.0:
        raise ValueError("--regime-size min/max multipliers must be > 0")
    if args.regime_size_min_mult > args.regime_size_max_mult:
        raise ValueError("--regime-size-min-mult must be <= --regime-size-max-mult")
    if args.tail_q10_cut < -0.02 or args.tail_q10_cut > 0.02:
        raise ValueError("--tail-q10-cut should be in [-0.02, 0.02]")
    if args.tail_agg_q10_cut < args.tail_q10_cut:
        raise ValueError("--tail-agg-q10-cut must be >= --tail-q10-cut")
    if args.tail_agg_q10_cut > 0.03:
        raise ValueError("--tail-agg-q10-cut should be <= 0.03")

    symbol_1_label = str(args.display_symbol_1 or "").strip().upper()
    symbol_2_label = str(args.display_symbol_2 or "").strip().upper()
    if not symbol_1_label or not symbol_2_label:
        raise ValueError("--display-symbol-1 and --display-symbol-2 must be non-empty")
    if symbol_1_label == symbol_2_label:
        raise ValueError("--display-symbol-1 and --display-symbol-2 must be different")

    cost_stress_multipliers = sorted({v for v in parse_float_list(args.cost_stress_multipliers) if v > 1.0})

    out_root = os.path.abspath(args.out_dir)
    models_dir = os.path.join(out_root, "models")
    backtest_dir = os.path.join(out_root, "backtest")
    ensure_dir(models_dir)
    ensure_dir(backtest_dir)

    meta_path = os.path.join(args.dataset_dir, "step3_dataset_meta.json")
    q_path = os.path.join(args.dataset_dir, "qqq_events_step3.parquet")
    s_path = os.path.join(args.dataset_dir, "spy_events_step3.parquet")
    if not (os.path.exists(meta_path) and os.path.exists(q_path) and os.path.exists(s_path)):
        raise FileNotFoundError("Missing Step 3 dataset artifacts. Run step3_build_training_dataset.py first.")

    feature_candidates = load_meta_features(meta_path)
    q_df = pd.read_parquet(q_path)
    s_df = pd.read_parquet(s_path)
    q_feats = select_feature_cols(
        q_df,
        feature_candidates,
        exclude_leaky_features=bool(args.exclude_leaky_features),
    )
    s_feats = select_feature_cols(
        s_df,
        feature_candidates,
        exclude_leaky_features=bool(args.exclude_leaky_features),
    )

    q_scored, q_trades, q_fold_rows, q_summary = train_symbol_walkforward(
        df=q_df,
        symbol="QQQ",
        feature_cols=q_feats,
        models_out_dir=models_dir,
        train_lookback_days=args.train_lookback_days,
        min_train_events=args.min_train_events,
        min_val_events=args.min_val_events,
        min_test_events=args.min_test_events,
        embargo_days=args.embargo_days,
        mix_struct_weight=args.mix_struct_weight,
        policy_profile=args.policy_profile,
        max_aggressive_size=args.max_aggressive_size,
        retune_every_folds=args.retune_every_folds,
        spy_drift_kill_switch=args.spy_drift_kill_switch,
        spy_drift_feature_z_cap=args.spy_drift_feature_z_cap,
        drought_relief_enable=args.drought_relief_enable,
        drought_relief_symbol=args.drought_relief_symbol,
        drought_target_trades_per_month=args.drought_target_trades_per_month,
        drought_p_cut_relax=args.drought_p_cut_relax,
        drought_ev_relax=args.drought_ev_relax,
        drought_size_boost=args.drought_size_boost,
        pattern_aid_enable=args.pattern_aid_enable,
        pattern_n_clusters=args.pattern_n_clusters,
        pattern_min_cluster_samples=args.pattern_min_cluster_samples,
        pattern_prior_strength=args.pattern_prior_strength,
        pattern_consistency_tol=args.pattern_consistency_tol,
        pattern_prob_strength=args.pattern_prob_strength,
        pattern_ret_strength=args.pattern_ret_strength,
        pattern_prob_max_abs_delta=args.pattern_prob_max_abs_delta,
        pattern_ret_max_abs_delta=args.pattern_ret_max_abs_delta,
        regime_model_enable=args.regime_model_enable,
        regime_lookahead_events=args.regime_lookahead_events,
        regime_label_quantile=args.regime_label_quantile,
        regime_min_train_samples=args.regime_min_train_samples,
        regime_p_cut=args.regime_p_cut,
        regime_agg_p_cut=args.regime_agg_p_cut,
        regime_ev_scale=args.regime_ev_scale,
        regime_size_scale=args.regime_size_scale,
        regime_size_min_mult=args.regime_size_min_mult,
        regime_size_max_mult=args.regime_size_max_mult,
        tail_q10_cut=args.tail_q10_cut,
        tail_agg_q10_cut=args.tail_agg_q10_cut,
    )
    s_scored, s_trades, s_fold_rows, s_summary = train_symbol_walkforward(
        df=s_df,
        symbol="SPY",
        feature_cols=s_feats,
        models_out_dir=models_dir,
        train_lookback_days=args.train_lookback_days,
        min_train_events=args.min_train_events,
        min_val_events=args.min_val_events,
        min_test_events=args.min_test_events,
        embargo_days=args.embargo_days,
        mix_struct_weight=args.mix_struct_weight,
        policy_profile=args.policy_profile,
        max_aggressive_size=args.max_aggressive_size,
        retune_every_folds=args.retune_every_folds,
        spy_drift_kill_switch=args.spy_drift_kill_switch,
        spy_drift_feature_z_cap=args.spy_drift_feature_z_cap,
        drought_relief_enable=args.drought_relief_enable,
        drought_relief_symbol=args.drought_relief_symbol,
        drought_target_trades_per_month=args.drought_target_trades_per_month,
        drought_p_cut_relax=args.drought_p_cut_relax,
        drought_ev_relax=args.drought_ev_relax,
        drought_size_boost=args.drought_size_boost,
        pattern_aid_enable=args.pattern_aid_enable,
        pattern_n_clusters=args.pattern_n_clusters,
        pattern_min_cluster_samples=args.pattern_min_cluster_samples,
        pattern_prior_strength=args.pattern_prior_strength,
        pattern_consistency_tol=args.pattern_consistency_tol,
        pattern_prob_strength=args.pattern_prob_strength,
        pattern_ret_strength=args.pattern_ret_strength,
        pattern_prob_max_abs_delta=args.pattern_prob_max_abs_delta,
        pattern_ret_max_abs_delta=args.pattern_ret_max_abs_delta,
        regime_model_enable=args.regime_model_enable,
        regime_lookahead_events=args.regime_lookahead_events,
        regime_label_quantile=args.regime_label_quantile,
        regime_min_train_samples=args.regime_min_train_samples,
        regime_p_cut=args.regime_p_cut,
        regime_agg_p_cut=args.regime_agg_p_cut,
        regime_ev_scale=args.regime_ev_scale,
        regime_size_scale=args.regime_size_scale,
        regime_size_min_mult=args.regime_size_min_mult,
        regime_size_max_mult=args.regime_size_max_mult,
        tail_q10_cut=args.tail_q10_cut,
        tail_agg_q10_cut=args.tail_agg_q10_cut,
    )

    q_scored_path = os.path.join(backtest_dir, "qqq_scored_events.parquet")
    s_scored_path = os.path.join(backtest_dir, "spy_scored_events.parquet")
    q_trades_path = os.path.join(backtest_dir, "qqq_trades.parquet")
    s_trades_path = os.path.join(backtest_dir, "spy_trades.parquet")
    q_scored.to_parquet(q_scored_path, index=False)
    s_scored.to_parquet(s_scored_path, index=False)
    q_trades.to_parquet(q_trades_path, index=False)
    s_trades.to_parquet(s_trades_path, index=False)

    q_folds_path = os.path.join(backtest_dir, "qqq_fold_summary.json")
    s_folds_path = os.path.join(backtest_dir, "spy_fold_summary.json")
    with open(q_folds_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(q_fold_rows, 6), f, separators=(",", ":"), ensure_ascii=True)
    with open(s_folds_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(s_fold_rows, 6), f, separators=(",", ":"), ensure_ascii=True)

    eq_q_unit = daily_equity_from_trades(q_trades, 1.0)
    eq_s_unit = daily_equity_from_trades(s_trades, 1.0)
    if eq_q_unit.empty:
        eq_q_unit = flat_equity_from_scored(q_scored)
    if eq_s_unit.empty:
        eq_s_unit = flat_equity_from_scored(s_scored)
    if eq_q_unit.empty or eq_s_unit.empty:
        raise RuntimeError("One symbol produced empty trade/scored equity; cannot build dual Step 3 portfolio.")
    idx = pd.date_range(
        min(eq_q_unit.index.min(), eq_s_unit.index.min()),
        max(eq_q_unit.index.max(), eq_s_unit.index.max()),
        freq="D",
        tz="UTC",
    )
    eq_q_unit = eq_q_unit.reindex(idx).ffill().fillna(1.0)
    eq_s_unit = eq_s_unit.reindex(idx).ffill().fillna(1.0)
    q_ret = eq_q_unit.pct_change().fillna(0.0)
    s_ret = eq_s_unit.pct_change().fillna(0.0)

    allocator_name = "equal_split"
    q_weight = pd.Series(0.5, index=idx, dtype=float)
    static_eval_score = None
    dynamic_eval_score = None
    split_i = int(len(idx) * args.portfolio_train_ratio)
    split_i = min(max(split_i, 2), len(idx) - 1)
    q_weight_static = pd.Series(0.5, index=idx, dtype=float)
    if args.portfolio_allocator in {"dynamic_regime", "dynamic_regime_forced"}:
        q_weight_static, split_i = build_static_weight_series(
            idx=idx,
            q_ret=q_ret,
            s_ret=s_ret,
            train_ratio=args.portfolio_train_ratio,
            weight_step=args.portfolio_weight_step,
            objective=args.portfolio_objective,
            min_weight=args.portfolio_min_weight,
            max_weight=args.portfolio_max_weight,
        )
        q_weight_dynamic = build_walk_forward_weights(
            idx=idx,
            q_ret=q_ret,
            s_ret=s_ret,
            train_ratio=args.portfolio_train_ratio,
            lookback_days=args.portfolio_lookback_days,
            min_train_days=args.portfolio_min_train_days,
            weight_step=args.portfolio_weight_step,
            objective=args.portfolio_objective,
            turnover_penalty=args.portfolio_turnover_penalty,
            momentum_days=args.portfolio_momentum_days,
            vol_days=args.portfolio_vol_days,
            vol_penalty=args.portfolio_vol_penalty,
            max_tilt=args.portfolio_max_tilt,
            weight_smoothing=args.portfolio_weight_smoothing,
            min_weight=args.portfolio_min_weight,
            max_weight=args.portfolio_max_weight,
        )
        r_static = q_weight_static * q_ret + (1.0 - q_weight_static) * s_ret
        r_dynamic = q_weight_dynamic * q_ret + (1.0 - q_weight_dynamic) * s_ret
        eval_slice = slice(split_i, len(idx))
        static_eval_score = score_returns(r_static.iloc[eval_slice], args.portfolio_objective)
        dynamic_eval_score = score_returns(r_dynamic.iloc[eval_slice], args.portfolio_objective)
        if args.portfolio_allocator == "dynamic_regime_forced":
            allocator_name = "dynamic_regime_forced"
            q_weight = q_weight_dynamic
        elif dynamic_eval_score > static_eval_score:
            allocator_name = "dynamic_regime"
            q_weight = q_weight_dynamic
        else:
            allocator_name = "static_train_optimized"
            q_weight = q_weight_static

    spy_weight = 1.0 - q_weight
    spy_weight_pre_guard = spy_weight.copy()
    spy_guard = pd.Series(1.0, index=idx, dtype=float)
    if not args.portfolio_no_spy_guard:
        spy_guard = build_spy_guard_series(
            idx=idx,
            q_ret=q_ret,
            s_ret=s_ret,
            spy_scored=s_scored,
            lookback_days=args.portfolio_spy_guard_lookback_days,
            drift_lookback_days=args.portfolio_spy_guard_drift_lookback_days,
            min_mult=args.portfolio_spy_guard_min_mult,
            dd_penalty=args.portfolio_spy_guard_dd_penalty,
        )
        spy_weight = (spy_weight * spy_guard).clip(lower=0.0, upper=1.0)
        q_weight = (1.0 - spy_weight).clip(lower=0.0, upper=1.0)

    r_dual = q_weight * q_ret + spy_weight * s_ret
    eq_dual = args.start_capital * (1.0 + r_dual).cumprod()

    eq_q = args.start_capital * eq_q_unit
    eq_s = args.start_capital * eq_s_unit

    trade_frames = [
        t for t in [q_trades, s_trades]
        if (not t.empty) and ("exit_time_utc" in t.columns)
    ]
    if trade_frames:
        all_trades = pd.concat(trade_frames, ignore_index=True).sort_values("exit_time_utc").reset_index(drop=True)
    else:
        all_trades = pd.DataFrame(columns=["exit_time_utc", "weighted_net_logret", "mode"])
    monthly = monthly_table(eq_dual, all_trades, args.start_capital)
    perf_dual = perf_from_equity_series(eq_dual, args.start_capital)
    perf_q = perf_from_equity_series(eq_q, args.start_capital)
    perf_s = perf_from_equity_series(eq_s, args.start_capital)

    best_month = monthly.loc[monthly["pnl"].idxmax()]
    worst_month = monthly.loc[monthly["pnl"].idxmin()]
    avg_monthly_pnl = float(monthly["pnl"].mean())
    median_monthly_pnl = float(monthly["pnl"].median())
    avg_monthly_trades = float(monthly["trades"].mean())
    monthly_positive_rate = float((monthly["pnl"] > 0.0).mean())
    monthly_negative_rate = float((monthly["pnl"] < 0.0).mean())
    positive_month_pnl = monthly.loc[monthly["pnl"] > 0.0, "pnl"].sort_values(ascending=False)
    positive_pnl_total = float(positive_month_pnl.sum()) if not positive_month_pnl.empty else 0.0
    top5_positive_pnl_share = (
        float(positive_month_pnl.head(5).sum() / positive_pnl_total) if positive_pnl_total > 1e-9 else 0.0
    )
    zero_trade_month_rate = float((monthly["trades"] <= 0.0).mean())
    aggr_rate_all = float((all_trades["mode"] == "aggressive").mean()) if not all_trades.empty else 0.0
    daily_pnl = eq_dual.diff().fillna(0.0)
    best_day = daily_pnl.idxmax()
    worst_day = daily_pnl.idxmin()
    best_day_amt = float(daily_pnl.loc[best_day])
    worst_day_amt = float(daily_pnl.loc[worst_day])
    q_weight_turnover = float(q_weight.diff().abs().fillna(0.0).sum())
    spy_guard_mean = float(spy_guard.mean())
    spy_guard_min = float(spy_guard.min())
    spy_guard_max = float(spy_guard.max())
    spy_weight_pre_guard_mean = float(spy_weight_pre_guard.mean())
    q_weight_mean = float(q_weight.mean())
    q_weight_min = float(q_weight.min())
    q_weight_max = float(q_weight.max())
    q_pat_rate = float((q_summary.get("fold_stability") or {}).get("pattern_active_rate") or 0.0)
    s_pat_rate = float((s_summary.get("fold_stability") or {}).get("pattern_active_rate") or 0.0)
    q_pat_prob = float((q_summary.get("fold_stability") or {}).get("pattern_mean_abs_prob_delta") or 0.0)
    s_pat_prob = float((s_summary.get("fold_stability") or {}).get("pattern_mean_abs_prob_delta") or 0.0)
    q_regime_rate = float((q_summary.get("fold_stability") or {}).get("regime_trade_pass_rate") or 0.0)
    s_regime_rate = float((s_summary.get("fold_stability") or {}).get("regime_trade_pass_rate") or 0.0)
    q_regime_prob = float((q_summary.get("fold_stability") or {}).get("regime_mean_prob") or 0.5)
    s_regime_prob = float((s_summary.get("fold_stability") or {}).get("regime_mean_prob") or 0.5)
    stress_tests: List[Dict] = []
    for mult in cost_stress_multipliers:
        stress_tests.append(
            portfolio_cost_stress_summary(
                idx=idx,
                q_trades=q_trades,
                s_trades=s_trades,
                q_weight=q_weight,
                spy_weight=spy_weight,
                start_capital=args.start_capital,
                cost_multiplier=mult,
            )
        )
    bootstrap = monthly_block_bootstrap_summary(
        monthly=monthly,
        start_capital=args.start_capital,
        n_samples=args.bootstrap_samples,
        block_months=args.bootstrap_block_months,
        seed=args.bootstrap_seed,
    )

    stress_text = "Cost stress: n/a"
    valid_stress = [s for s in stress_tests if ("error" not in s)]
    if valid_stress:
        parts = []
        for s in valid_stress[:2]:
            mult = float(s.get("cost_multiplier") or 1.0)
            avg_s = float(s.get("avg_monthly_pnl") or 0.0)
            calmar_s = float((s.get("dual_perf") or {}).get("calmar") or 0.0)
            parts.append(f"x{mult:.2f} avg ${avg_s:,.0f}, calmar {calmar_s:.2f}")
        stress_text = "Cost stress: " + " | ".join(parts)

    boot_text = "Bootstrap: n/a"
    if bool(bootstrap.get("enabled")):
        avg_ci = bootstrap.get("avg_monthly_pnl", {})
        calmar_ci = bootstrap.get("calmar", {})
        boot_text = (
            f"Bootstrap P10/P50/P90 avg monthly PnL: "
            f"${float(avg_ci.get('p10') or 0.0):,.0f}/"
            f"${float(avg_ci.get('p50') or 0.0):,.0f}/"
            f"${float(avg_ci.get('p90') or 0.0):,.0f}"
            f" | Calmar P10/P50/P90: "
            f"{float(calmar_ci.get('p10') or 0.0):.2f}/"
            f"{float(calmar_ci.get('p50') or 0.0):.2f}/"
            f"{float(calmar_ci.get('p90') or 0.0):.2f}"
        )

    stats_text = "\n".join(
        [
            f"End equity: ${float(perf_dual['end_equity']):,.0f}",
            f"CAGR: {100.0 * float(perf_dual['cagr']):.2f}%  |  Max DD: {100.0 * float(perf_dual['max_drawdown']):.2f}%",
            f"Calmar: {float(perf_dual['calmar']) if perf_dual['calmar'] is not None else float('nan'):.2f}",
            (
                f"Allocator: {allocator_name}  |  {symbol_1_label} weight avg/min/max: "
                f"{q_weight_mean:.2f}/{q_weight_min:.2f}/{q_weight_max:.2f}"
            ),
            f"{symbol_2_label} guard avg/min/max: {spy_guard_mean:.2f}/{spy_guard_min:.2f}/{spy_guard_max:.2f}",
            f"Avg monthly PnL: ${avg_monthly_pnl:,.0f}  |  Median: ${median_monthly_pnl:,.0f}",
            f"Positive months: {100.0 * monthly_positive_rate:.1f}%  |  Negative months: {100.0 * monthly_negative_rate:.1f}%",
            f"Top-5 positive month share: {100.0 * top5_positive_pnl_share:.1f}%  |  Zero-trade months: {100.0 * zero_trade_month_rate:.1f}%",
            f"Avg trades/month: {avg_monthly_trades:.1f}  |  Aggressive rate: {100.0 * aggr_rate_all:.1f}%",
            f"Best month: {best_month['month_end'].strftime('%Y-%m')} ${float(best_month['pnl']):,.0f}",
            f"Worst month: {worst_month['month_end'].strftime('%Y-%m')} ${float(worst_month['pnl']):,.0f}",
            f"Best day: {best_day.strftime('%Y-%m-%d')} ${best_day_amt:,.0f}",
            f"Worst day: {worst_day.strftime('%Y-%m-%d')} ${worst_day_amt:,.0f}",
            (
                f"Pattern aid active ({symbol_1_label}/{symbol_2_label} folds): {100.0 * q_pat_rate:.1f}%/{100.0 * s_pat_rate:.1f}%  |  "
                f"Mean |p shift|: {q_pat_prob:.4f}/{s_pat_prob:.4f}"
                if args.pattern_aid_enable
                else "Pattern aid: disabled"
            ),
            (
                f"Regime model pass-rate ({symbol_1_label}/{symbol_2_label}): {100.0 * q_regime_rate:.1f}%/{100.0 * s_regime_rate:.1f}%  |  "
                f"Mean regime prob: {q_regime_prob:.3f}/{s_regime_prob:.3f}"
                if args.regime_model_enable
                else "Regime model: disabled"
            ),
            stress_text,
            boot_text,
        ]
    )

    fig_path = os.path.join(backtest_dir, "step3_dual_portfolio_curve.png")
    plot_dual(
        eq_dual=eq_dual,
        eq_q=eq_q,
        eq_s=eq_s,
        q_weight=q_weight,
        monthly=monthly,
        out_path=fig_path,
        start_capital=args.start_capital,
        stats_text=stats_text,
        symbol_1_label=symbol_1_label,
        symbol_2_label=symbol_2_label,
    )

    summary = {
        "meta": {
            "script": "step3_train_and_backtest.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "display_symbol_1": symbol_1_label,
            "display_symbol_2": symbol_2_label,
            "train_lookback_days": args.train_lookback_days,
            "embargo_days": args.embargo_days,
            "min_train_events": args.min_train_events,
            "min_val_events": args.min_val_events,
            "min_test_events": args.min_test_events,
            "mix_struct_weight": args.mix_struct_weight,
            "policy_profile": args.policy_profile,
            "max_aggressive_size": args.max_aggressive_size,
            "portfolio_allocator": args.portfolio_allocator,
            "portfolio_objective": args.portfolio_objective,
            "portfolio_train_ratio": args.portfolio_train_ratio,
            "portfolio_weight_step": args.portfolio_weight_step,
            "portfolio_lookback_days": args.portfolio_lookback_days,
            "portfolio_min_train_days": args.portfolio_min_train_days,
            "portfolio_turnover_penalty": args.portfolio_turnover_penalty,
            "portfolio_weight_smoothing": args.portfolio_weight_smoothing,
            "portfolio_momentum_days": args.portfolio_momentum_days,
            "portfolio_vol_days": args.portfolio_vol_days,
            "portfolio_vol_penalty": args.portfolio_vol_penalty,
            "portfolio_max_tilt": args.portfolio_max_tilt,
            "portfolio_min_weight": args.portfolio_min_weight,
            "portfolio_max_weight": args.portfolio_max_weight,
            "portfolio_spy_guard_enabled": (not args.portfolio_no_spy_guard),
            "portfolio_spy_guard_lookback_days": args.portfolio_spy_guard_lookback_days,
            "portfolio_spy_guard_drift_lookback_days": args.portfolio_spy_guard_drift_lookback_days,
            "portfolio_spy_guard_min_mult": args.portfolio_spy_guard_min_mult,
            "portfolio_spy_guard_dd_penalty": args.portfolio_spy_guard_dd_penalty,
            "spy_drift_kill_switch": args.spy_drift_kill_switch,
            "spy_drift_feature_z_cap": args.spy_drift_feature_z_cap,
            "drought_relief_enable": args.drought_relief_enable,
            "drought_relief_symbol": args.drought_relief_symbol,
            "drought_target_trades_per_month": args.drought_target_trades_per_month,
            "drought_p_cut_relax": args.drought_p_cut_relax,
            "drought_ev_relax": args.drought_ev_relax,
            "drought_size_boost": args.drought_size_boost,
            "pattern_aid_enable": args.pattern_aid_enable,
            "pattern_n_clusters": args.pattern_n_clusters,
            "pattern_min_cluster_samples": args.pattern_min_cluster_samples,
            "pattern_prior_strength": args.pattern_prior_strength,
            "pattern_consistency_tol": args.pattern_consistency_tol,
            "pattern_prob_strength": args.pattern_prob_strength,
            "pattern_ret_strength": args.pattern_ret_strength,
            "pattern_prob_max_abs_delta": args.pattern_prob_max_abs_delta,
            "pattern_ret_max_abs_delta": args.pattern_ret_max_abs_delta,
            "exclude_leaky_features": bool(args.exclude_leaky_features),
            "regime_model_enable": bool(args.regime_model_enable),
            "regime_lookahead_events": args.regime_lookahead_events,
            "regime_label_quantile": args.regime_label_quantile,
            "regime_min_train_samples": args.regime_min_train_samples,
            "regime_p_cut": args.regime_p_cut,
            "regime_agg_p_cut": args.regime_agg_p_cut,
            "regime_ev_scale": args.regime_ev_scale,
            "regime_size_scale": args.regime_size_scale,
            "regime_size_min_mult": args.regime_size_min_mult,
            "regime_size_max_mult": args.regime_size_max_mult,
            "tail_q10_cut": args.tail_q10_cut,
            "tail_agg_q10_cut": args.tail_agg_q10_cut,
            "cost_stress_multipliers": cost_stress_multipliers,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_block_months": args.bootstrap_block_months,
            "bootstrap_seed": args.bootstrap_seed,
            "retune_every_folds": args.retune_every_folds,
            "model_stack": (
                "LightGBM classifier/regressor/quantiles + ridge ensemble + calibrated probabilities + confidence sizing + q10 tail gate"
                + (" + cross-asset pattern context aid" if args.pattern_aid_enable else "")
                + (" + regime classifier gate/sizer" if args.regime_model_enable else "")
            ),
            "note": (
                "Nested walk-forward tuning with time+label_end purge/embargo, drift-aware safety guard, and monthly out-of-sample testing."
                + (" Pattern context aid enabled." if args.pattern_aid_enable else "")
                + (" Regime classifier enabled." if args.regime_model_enable else "")
                + (" Leaky outcome features excluded." if args.exclude_leaky_features else "")
            ),
        },
        "symbol_summaries": {
            "qqq": q_summary,
            "spy": s_summary,
        },
        "portfolio": {
            "dual_perf": perf_dual,
            "qqq_standalone_perf": perf_q,
            "spy_standalone_perf": perf_s,
            "allocator": {
                "selected": allocator_name,
                "holdout_split_index": int(split_i),
                "holdout_start_utc": str(idx[split_i]),
                "holdout_score_static": static_eval_score,
                "holdout_score_dynamic": dynamic_eval_score,
                "qqq_weight_mean": q_weight_mean,
                "qqq_weight_min": q_weight_min,
                "qqq_weight_max": q_weight_max,
                "qqq_weight_turnover_abs": q_weight_turnover,
                "spy_weight_mean_pre_guard": spy_weight_pre_guard_mean,
                "spy_guard_mean": spy_guard_mean,
                "spy_guard_min": spy_guard_min,
                "spy_guard_max": spy_guard_max,
            },
            "avg_monthly_pnl": avg_monthly_pnl,
            "median_monthly_pnl": median_monthly_pnl,
            "avg_monthly_trades": avg_monthly_trades,
            "monthly_positive_rate": monthly_positive_rate,
            "monthly_negative_rate": monthly_negative_rate,
            "top5_positive_pnl_share": top5_positive_pnl_share,
            "zero_trade_month_rate": zero_trade_month_rate,
            "total_trades": int(len(all_trades)),
            "aggressive_trade_rate": aggr_rate_all,
            "best_month": {
                "month_end": str(best_month["month_end"]),
                "pnl": float(best_month["pnl"]),
                "ret": float(best_month["ret"]),
                "trades": float(best_month["trades"]),
            },
            "worst_month": {
                "month_end": str(worst_month["month_end"]),
                "pnl": float(worst_month["pnl"]),
                "ret": float(worst_month["ret"]),
                "trades": float(worst_month["trades"]),
            },
            "best_day": {
                "date_utc": str(best_day),
                "pnl": best_day_amt,
            },
            "worst_day": {
                "date_utc": str(worst_day),
                "pnl": worst_day_amt,
            },
            "cost_stress_tests": stress_tests,
            "bootstrap": bootstrap,
        },
        "outputs": {
            "qqq_scored_events": q_scored_path,
            "spy_scored_events": s_scored_path,
            "qqq_trades": q_trades_path,
            "spy_trades": s_trades_path,
            "qqq_fold_summary": q_folds_path,
            "spy_fold_summary": s_folds_path,
            "dual_plot": fig_path,
        },
    }
    summary = round_obj(summary, 6)
    summary_path = os.path.join(backtest_dir, "step3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    monthly_path = os.path.join(backtest_dir, "step3_monthly_table.parquet")
    monthly.to_parquet(monthly_path, index=False)
    print(f"[STEP3-TRAIN] Wrote: {summary_path}")
    print(f"[STEP3-TRAIN] Wrote: {fig_path}")
    print(f"[STEP3-TRAIN] Wrote: {monthly_path}")


if __name__ == "__main__":
    main()
