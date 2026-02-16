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
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


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
) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    out["p_pred"] = np.clip(p_pred, 1e-4, 1.0 - 1e-4)
    out["ret_pred"] = np.clip(ret_pred, -0.03, 0.03)
    out["ev_struct"] = (
        out["p_pred"] * out["a_tp"] - (1.0 - out["p_pred"]) * out["b_sl"] - out["cost_rt"]
    )
    mix = float(params["mix_struct_weight"])
    out["ev_final"] = mix * out["ev_struct"] + (1.0 - mix) * out["ret_pred"]

    trade = (out["p_pred"] >= float(params["p_cut"])) & (out["ev_final"] >= float(params["ev_cut"]))
    agg = trade & (out["p_pred"] >= float(params["agg_p_cut"])) & (
        out["ev_final"] >= float(params["agg_ev_cut"])
    )

    out["trade"] = trade.astype(int)
    out["mode"] = np.where(agg, "aggressive", np.where(trade, "safe", "flat"))
    out["size_mult"] = 0.0
    out.loc[trade, "size_mult"] = float(params["safe_size"])
    out.loc[agg, "size_mult"] = 1.0
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
    return traded, perf


def select_feature_cols(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    cols = [c for c in candidate_cols if c in df.columns]
    numeric = set(df.select_dtypes(include=[np.number]).columns.tolist())
    cols = [c for c in cols if c in numeric]
    cols = [c for c in cols if df[c].notna().mean() >= 0.35]
    if len(cols) < 8:
        raise RuntimeError("Too few valid feature columns after NaN filtering.")
    return sorted(cols)


def tune_fold(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    alpha_grid: List[float],
    mix_struct_weight: float,
    min_val_trades: int,
) -> Tuple[Dict, Dict, Dict]:
    best_alpha_score = -1e18
    best_prob_model: Optional[Dict] = None
    best_ret_model: Optional[Dict] = None

    for ap in alpha_grid:
        for ar in alpha_grid:
            pm = fit_ridge_linear_model(fit_df, feature_cols, target_col="y", ridge_alpha=ap)
            rm = fit_ridge_linear_model(fit_df, feature_cols, target_col="net_logret", ridge_alpha=ar)
            if pm is None or rm is None:
                continue
            p_val = sigmoid(predict_ridge_linear(val_df, pm))
            r_val = predict_ridge_linear(val_df, rm)
            temp_params = {
                "mix_struct_weight": mix_struct_weight,
                "p_cut": 0.53,
                "agg_p_cut": 0.67,
                "safe_size": 0.60,
                "ev_cut": float(np.quantile(0.65 * (p_val * val_df["a_tp"] - (1.0 - p_val) * val_df["b_sl"] - val_df["cost_rt"]) + 0.35 * np.clip(r_val, -0.03, 0.03), 0.55)),
                "agg_ev_cut": 0.0,
            }
            temp_params["agg_ev_cut"] = max(temp_params["ev_cut"], temp_params["ev_cut"] + 0.0002)
            _, perf = simulate_policy(val_df, p_val, r_val, temp_params)
            s = policy_score(perf, min_trades=min_val_trades)
            if s > best_alpha_score:
                best_alpha_score = s
                best_prob_model = pm
                best_ret_model = rm

    if best_prob_model is None or best_ret_model is None:
        raise RuntimeError("Could not fit any model in alpha tuning.")

    p_val = sigmoid(predict_ridge_linear(val_df, best_prob_model))
    r_val = predict_ridge_linear(val_df, best_ret_model)
    ev_struct = p_val * val_df["a_tp"].to_numpy() - (1.0 - p_val) * val_df["b_sl"].to_numpy() - val_df["cost_rt"].to_numpy()
    ev_final = mix_struct_weight * ev_struct + (1.0 - mix_struct_weight) * np.clip(r_val, -0.03, 0.03)

    best_params: Optional[Dict] = None
    best_threshold_score = -1e18
    for p_cut in [0.50, 0.53, 0.56, 0.60]:
        for ev_q in [0.45, 0.55, 0.65]:
            ev_cut = float(np.quantile(ev_final, ev_q))
            for agg_p in [0.62, 0.68, 0.74]:
                for safe_size in [0.45, 0.60, 0.75]:
                    params = {
                        "mix_struct_weight": mix_struct_weight,
                        "p_cut": float(p_cut),
                        "agg_p_cut": float(agg_p),
                        "safe_size": float(safe_size),
                        "ev_cut": ev_cut,
                        "agg_ev_cut": float(max(ev_cut, ev_cut + 0.0002)),
                    }
                    _, perf = simulate_policy(val_df, p_val, r_val, params)
                    s = policy_score(perf, min_trades=min_val_trades)
                    if s > best_threshold_score:
                        best_threshold_score = s
                        best_params = params
    if best_params is None:
        raise RuntimeError("Could not tune policy thresholds.")

    return best_prob_model, best_ret_model, best_params


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
    alpha_grid: List[float],
    mix_struct_weight: float,
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
        if len(train_df) < min_train_events:
            continue
        train_df = train_df.sort_values("decision_time_utc").reset_index(drop=True)
        split_i = int(len(train_df) * 0.80)
        split_i = min(max(split_i, min_train_events - min_val_events), len(train_df) - min_val_events)
        fit_df = train_df.iloc[:split_i].copy()
        val_df = train_df.iloc[split_i:].copy()
        if len(fit_df) < (min_train_events - min_val_events) or len(val_df) < min_val_events:
            continue

        try:
            prob_model_tuned, ret_model_tuned, best_params = tune_fold(
                fit_df=fit_df,
                val_df=val_df,
                feature_cols=feature_cols,
                alpha_grid=alpha_grid,
                mix_struct_weight=mix_struct_weight,
                min_val_trades=max(6, min_test_events // 2),
            )
        except Exception:
            continue

        prob_model = fit_ridge_linear_model(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="y",
            ridge_alpha=float(prob_model_tuned["ridge_alpha"]),
        )
        ret_model = fit_ridge_linear_model(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="net_logret",
            ridge_alpha=float(ret_model_tuned["ridge_alpha"]),
        )
        if prob_model is None or ret_model is None:
            continue

        p_test = sigmoid(predict_ridge_linear(test_df, prob_model))
        r_test = predict_ridge_linear(test_df, ret_model)
        traded, fold_perf = simulate_policy(test_df, p_test, r_test, best_params)
        fold_score = policy_score(fold_perf, min_trades=max(6, min_test_events // 2))

        fold_id = f"{symbol.lower()}_{test_start.strftime('%Y%m')}"
        scored = test_df.copy()
        scored["p_pred"] = p_test
        scored["ret_pred"] = np.clip(r_test, -0.03, 0.03)
        scored["fold_id"] = fold_id
        scored["fold_test_start_utc"] = test_start
        scored["fold_test_end_utc"] = test_end
        all_scored.append(scored)

        if not traded.empty:
            traded["fold_id"] = fold_id
            traded["fold_test_start_utc"] = test_start
            traded["fold_test_end_utc"] = test_end
            all_trades.append(traded)

        model_blob = {
            "fold_id": fold_id,
            "symbol": symbol,
            "test_start_utc": str(test_start),
            "test_end_utc": str(test_end),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_trades_test": int(fold_perf["n"]),
            "policy_score_test": float(fold_score),
            "best_params": best_params,
            "prob_model": prob_model,
            "ret_model": ret_model,
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
                "policy_score_test": fold_score,
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
        "perf": symbol_perf,
    }
    return scored_df, trades_df, fold_rows, symbol_summary


def daily_equity_from_trades(trades: pd.DataFrame, start_capital: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    x = trades.sort_values("exit_time_utc").copy()
    x["exit_time_utc"] = pd.to_datetime(x["exit_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["exit_time_utc"])
    if x.empty:
        return pd.Series(dtype=float)
    eq = start_capital * np.exp(x["weighted_net_logret"].to_numpy(dtype=float).cumsum())
    return pd.Series(eq, index=x["exit_time_utc"]).resample("D").last()


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
    monthly: pd.DataFrame,
    out_path: str,
    start_capital: float,
    stats_text: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        gridspec_kw={"height_ratios": [3.0, 1.6]},
    )
    ax1.plot(eq_dual.index, eq_dual.values, color="#1d3557", linewidth=2.2, label="Dual Step3")
    ax1.plot(eq_q.index, eq_q.values, color="#457b9d", linewidth=1.2, alpha=0.7, label="QQQ Step3")
    ax1.plot(eq_s.index, eq_s.values, color="#2a9d8f", linewidth=1.2, alpha=0.7, label="SPY Step3")
    ax1.axhline(start_capital, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_title("Step 3 Walk-Forward ML Portfolio")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.25)
    ax1.legend()
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
    ap.add_argument("--dataset-dir", required=True, help="Path to step3_out/dataset")
    ap.add_argument("--out-dir", required=True, help="Path to step3_out")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--train-lookback-days", type=int, default=1095)
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--min-train-events", type=int, default=200)
    ap.add_argument("--min-val-events", type=int, default=40)
    ap.add_argument("--min-test-events", type=int, default=6)
    ap.add_argument("--mix-struct-weight", type=float, default=0.65, help="Weight on structural EV head")
    args = ap.parse_args()

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
    q_feats = select_feature_cols(q_df, feature_candidates)
    s_feats = select_feature_cols(s_df, feature_candidates)

    alpha_grid = [2.0, 6.0, 12.0]
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
        alpha_grid=alpha_grid,
        mix_struct_weight=args.mix_struct_weight,
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
        alpha_grid=alpha_grid,
        mix_struct_weight=args.mix_struct_weight,
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

    half_cap = args.start_capital * 0.5
    eq_q = daily_equity_from_trades(q_trades, half_cap)
    eq_s = daily_equity_from_trades(s_trades, half_cap)
    if eq_q.empty or eq_s.empty:
        raise RuntimeError("One symbol produced empty trade equity; cannot build dual Step 3 portfolio.")
    idx = pd.date_range(
        min(eq_q.index.min(), eq_s.index.min()),
        max(eq_q.index.max(), eq_s.index.max()),
        freq="D",
        tz="UTC",
    )
    eq_q = eq_q.reindex(idx).ffill().fillna(half_cap)
    eq_s = eq_s.reindex(idx).ffill().fillna(half_cap)
    eq_dual = eq_q + eq_s

    all_trades = pd.concat([q_trades, s_trades], ignore_index=True).sort_values("exit_time_utc").reset_index(drop=True)
    monthly = monthly_table(eq_dual, all_trades, args.start_capital)
    perf_dual = perf_from_equity_series(eq_dual, args.start_capital)
    perf_q = perf_from_equity_series(eq_q, half_cap)
    perf_s = perf_from_equity_series(eq_s, half_cap)

    best_month = monthly.loc[monthly["pnl"].idxmax()]
    worst_month = monthly.loc[monthly["pnl"].idxmin()]
    avg_monthly_pnl = float(monthly["pnl"].mean())
    avg_monthly_trades = float(monthly["trades"].mean())
    aggr_rate_all = float((all_trades["mode"] == "aggressive").mean()) if not all_trades.empty else 0.0
    daily_pnl = eq_dual.diff().fillna(0.0)
    best_day = daily_pnl.idxmax()
    worst_day = daily_pnl.idxmin()
    best_day_amt = float(daily_pnl.loc[best_day])
    worst_day_amt = float(daily_pnl.loc[worst_day])

    stats_text = "\n".join(
        [
            f"End equity: ${float(perf_dual['end_equity']):,.0f}",
            f"CAGR: {100.0 * float(perf_dual['cagr']):.2f}%  |  Max DD: {100.0 * float(perf_dual['max_drawdown']):.2f}%",
            f"Calmar: {float(perf_dual['calmar']) if perf_dual['calmar'] is not None else float('nan'):.2f}",
            f"Avg monthly PnL: ${avg_monthly_pnl:,.0f}",
            f"Avg trades/month: {avg_monthly_trades:.1f}  |  Aggressive rate: {100.0 * aggr_rate_all:.1f}%",
            f"Best month: {best_month['month_end'].strftime('%Y-%m')} ${float(best_month['pnl']):,.0f}",
            f"Worst month: {worst_month['month_end'].strftime('%Y-%m')} ${float(worst_month['pnl']):,.0f}",
            f"Best day: {best_day.strftime('%Y-%m-%d')} ${best_day_amt:,.0f}",
            f"Worst day: {worst_day.strftime('%Y-%m-%d')} ${worst_day_amt:,.0f}",
        ]
    )

    fig_path = os.path.join(backtest_dir, "step3_dual_portfolio_curve.png")
    plot_dual(
        eq_dual=eq_dual,
        eq_q=eq_q,
        eq_s=eq_s,
        monthly=monthly,
        out_path=fig_path,
        start_capital=args.start_capital,
        stats_text=stats_text,
    )

    summary = {
        "meta": {
            "script": "step3_train_and_backtest.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "train_lookback_days": args.train_lookback_days,
            "embargo_days": args.embargo_days,
            "min_train_events": args.min_train_events,
            "min_val_events": args.min_val_events,
            "min_test_events": args.min_test_events,
            "mix_struct_weight": args.mix_struct_weight,
            "alpha_grid": alpha_grid,
            "note": "Walk-forward monthly folds with train/val/test separation and purge via exit_time < test_start.",
        },
        "symbol_summaries": {
            "qqq": q_summary,
            "spy": s_summary,
        },
        "portfolio": {
            "dual_perf": perf_dual,
            "qqq_half_cap_perf": perf_q,
            "spy_half_cap_perf": perf_s,
            "avg_monthly_pnl": avg_monthly_pnl,
            "avg_monthly_trades": avg_monthly_trades,
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
