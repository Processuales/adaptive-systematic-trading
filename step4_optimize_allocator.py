#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP = "STEP4-ALLOC"
SCRIPT_VERSION = "1.1.0"


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", flush=True)


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly table: {path}")
    x = pd.read_parquet(path).copy()
    x["month_end"] = pd.to_datetime(x["month_end"], utc=True, errors="coerce")
    x = x.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    for c in ("ret", "equity", "pnl", "trades"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        else:
            x[c] = 0.0
    return x


def max_dd(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    eq = (1.0 + ret.fillna(0.0).astype(float)).cumprod()
    peak = eq.cummax()
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return abs(float(dd.min()))


def perf(ret: pd.Series, start_capital: float) -> Dict[str, float]:
    r = ret.fillna(0.0).astype(float)
    if r.empty:
        return {
            "end_equity": float(start_capital),
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "avg_monthly_pnl": 0.0,
            "median_monthly_pnl": 0.0,
            "positive_month_rate": 0.0,
            "n_months": 0,
        }
    eq = float(start_capital) * (1.0 + r).cumprod()
    pnl = eq.diff().fillna(eq.iloc[0] - float(start_capital))
    years = max(len(r) / 12.0, 1.0 / 12.0)
    end_eq = float(eq.iloc[-1])
    cagr = float((max(end_eq / max(float(start_capital), 1e-9), 1e-9) ** (1.0 / years)) - 1.0)
    mdd = max_dd(r)
    calmar = float(cagr / mdd) if mdd > 1e-9 else 0.0
    return {
        "end_equity": end_eq,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": calmar,
        "avg_monthly_pnl": float(pnl.mean()),
        "median_monthly_pnl": float(pnl.median()),
        "positive_month_rate": float((pnl > 0.0).mean()),
        "n_months": int(len(r)),
    }


def bootstrap_p10_avg_monthly_pnl(ret: pd.Series, start_capital: float, n_samples: int, block_months: int) -> float:
    r = ret.fillna(0.0).astype(float).to_numpy()
    n = len(r)
    if n_samples <= 0 or n < 4:
        return 0.0
    rng = np.random.default_rng(42)
    avg_arr = []
    for _ in range(n_samples):
        idxs = []
        while len(idxs) < n:
            s = int(rng.integers(0, n))
            for j in range(block_months):
                idxs.append((s + j) % n)
                if len(idxs) >= n:
                    break
        rr = r[np.array(idxs[:n], dtype=int)]
        eq = float(start_capital) * np.cumprod(1.0 + rr)
        pnl = np.diff(np.insert(eq, 0, float(start_capital)))
        avg_arr.append(float(np.mean(pnl)))
    return float(np.quantile(np.array(avg_arr, dtype=float), 0.10))


def build_features(rs: pd.Series, rb: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"ret_spyqqq": rs, "ret_btceth": rb})
    for c in ("ret_spyqqq", "ret_btceth"):
        for w in (1, 3, 6, 12):
            df[f"{c}_mom_{w}"] = df[c].rolling(w, min_periods=max(1, w // 2)).sum().shift(1)
        for w in (3, 6, 12):
            df[f"{c}_vol_{w}"] = df[c].rolling(w, min_periods=max(2, w // 2)).std().shift(1)
    df["spread_mom_3"] = df["ret_spyqqq_mom_3"] - df["ret_btceth_mom_3"]
    df["spread_mom_6"] = df["ret_spyqqq_mom_6"] - df["ret_btceth_mom_6"]
    df["spread_vol_6"] = df["ret_spyqqq_vol_6"] - df["ret_btceth_vol_6"]
    return df


def utility_labels(rs_next: np.ndarray, rb_next: np.ndarray, bins: np.ndarray) -> np.ndarray:
    u = np.stack([w * rs_next + (1.0 - w) * rb_next for w in bins], axis=1)
    return np.argmax(u, axis=1).astype(int)


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    static_w: float
    lookback: int
    dd_penalty: float
    min_w: float
    max_w: float
    smoothing: float
    max_step: float
    gross_base: float
    gross_conf_scale: float
    max_gross: float
    target_vol: float
    vol_lb: int
    turn_cost_bps: float
    gross_cost_bps: float


def candidates() -> List[Candidate]:
    out = [
        Candidate("baseline_spy_only", "static", 1.0, 6, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 12, 12.0, 6.0),
        Candidate("baseline_btc_only", "static", 0.0, 6, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 12, 12.0, 6.0),
        Candidate("baseline_equal", "static", 0.5, 6, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 12, 12.0, 6.0),
        Candidate("btc_only_levered_115", "static", 0.0, 6, 0.0, 0.0, 1.0, 0.0, 1.0, 1.15, 0.0, 1.15, 0.0, 12, 12.0, 6.0),
        Candidate("btc_only_levered_130", "static", 0.0, 6, 0.0, 0.0, 1.0, 0.0, 1.0, 1.30, 0.0, 1.30, 0.0, 12, 12.0, 6.0),
    ]
    for lb in (3, 6, 12):
        for max_g in (1.0, 1.15, 1.30):
            out.append(
                Candidate(
                    f"heuristic_lb{lb}_g{str(max_g).replace('.', '_')}",
                    "heuristic",
                    0.5,
                    lb,
                    1.2,
                    0.0,
                    0.85,
                    0.20,
                    0.25,
                    1.0,
                    0.30,
                    max_g,
                    0.045 if max_g > 1.0 else 0.0,
                    12,
                    12.0,
                    6.0,
                )
            )
            out.append(
                Candidate(
                    f"regime_guard_lb{lb}_g{str(max_g).replace('.', '_')}",
                    "regime_guard",
                    0.5,
                    lb,
                    0.0,
                    0.05,
                    0.95,
                    0.35,
                    0.30,
                    1.0,
                    0.25,
                    max_g,
                    0.045 if max_g > 1.0 else 0.0,
                    12,
                    12.0,
                    6.0,
                )
            )
    for method in ("ml_lr", "ml_rf", "regime_rf", "ml_dual_ridge"):
        for lb in (24, 36, 48):
            for max_g in (1.0, 1.15, 1.30):
                out.append(
                    Candidate(
                        f"{method}_lb{lb}_g{str(max_g).replace('.', '_')}",
                        method,
                        0.5,
                        lb,
                        2.5 if method == "ml_dual_ridge" else 0.8,
                        0.0,
                        1.0,
                        0.25,
                        0.25,
                        1.0,
                        0.30,
                        max_g,
                        0.040 if max_g > 1.0 else 0.0,
                        12,
                        12.0,
                        6.0,
                    )
                )
    return out


def heuristic_weight(hist_s: np.ndarray, hist_b: np.ndarray, dd_penalty: float) -> float:
    if len(hist_s) < 2 or len(hist_b) < 2:
        return 0.5
    sh_s = float(np.mean(hist_s)) / max(float(np.std(hist_s)), 1e-9)
    sh_b = float(np.mean(hist_b)) / max(float(np.std(hist_b)), 1e-9)
    sc_s = sh_s - dd_penalty * max_dd(pd.Series(hist_s))
    sc_b = sh_b - dd_penalty * max_dd(pd.Series(hist_b))
    z_s = sc_s / 0.35
    z_b = sc_b / 0.35
    z_m = max(z_s, z_b)
    e_s = np.exp(z_s - z_m)
    e_b = np.exp(z_b - z_m)
    return float(e_s / max(e_s + e_b, 1e-12))


def fit_model(method: str, x: np.ndarray, y: np.ndarray):
    if method == "ml_lr":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        m.fit(x, y)
        return m
    if method == "ml_rf":
        m = RandomForestClassifier(
            n_estimators=220,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced_subsample",
        )
        m.fit(x, y)
        return m
    if method == "regime_rf":
        m = RandomForestClassifier(
            n_estimators=260,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
        )
        m.fit(x, y)
        return m
    raise ValueError(f"unsupported method {method}")


def fit_regressor(method: str, x: np.ndarray, y: np.ndarray):
    if method == "ridge":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("rg", Ridge(alpha=1.5)),
            ]
        )
        m.fit(x, y)
        return m
    if method == "rf":
        m = RandomForestRegressor(
            n_estimators=220,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42,
        )
        m.fit(x, y)
        return m
    raise ValueError(f"unsupported regressor method {method}")


def simulate(df: pd.DataFrame, c: Candidate, bins: np.ndarray, min_history: int) -> pd.DataFrame:
    feat = [k for k in df.columns if k.startswith(("ret_spyqqq_", "ret_btceth_", "spread_"))]
    z = df.dropna(subset=feat).copy().sort_index()
    if len(z) < max(min_history + 12, 36):
        raise RuntimeError("too few rows after feature build")
    rows = []
    w_prev = 0.5
    g_prev = 1.0
    n_bins = len(bins)
    uniform = 1.0 / n_bins
    for i in range(min_history, len(z)):
        hist = z.iloc[:i]
        cur = z.iloc[i : i + 1]
        rs = float(cur["ret_spyqqq"].iloc[0])
        rb = float(cur["ret_btceth"].iloc[0])
        conf = 0.0
        if c.method == "static":
            w_raw = float(c.static_w)
        elif c.method == "heuristic":
            w_raw = heuristic_weight(
                hist["ret_spyqqq"].tail(c.lookback).to_numpy(float),
                hist["ret_btceth"].tail(c.lookback).to_numpy(float),
                c.dd_penalty,
            )
            conf = abs(w_raw - 0.5) * 2.0
        elif c.method == "regime_guard":
            hb = hist["ret_btceth"]
            lb = int(max(3, c.lookback))
            mom_now = float(hb.tail(lb).sum())
            vol_now = float(hb.tail(6).std()) if len(hb) >= 3 else 0.0
            mom_series = hb.rolling(lb, min_periods=max(2, lb // 2)).sum().dropna()
            vol_series = hb.rolling(6, min_periods=3).std().dropna()
            mom_th = float(mom_series.quantile(0.35)) if len(mom_series) >= 6 else 0.0
            vol_th = float(vol_series.quantile(0.60)) if len(vol_series) >= 6 else max(float(vol_now), 0.06)
            risk_off = bool((mom_now < mom_th) and (vol_now > vol_th))
            w_target = 0.75 if risk_off else 0.15
            w_raw = float(np.clip(w_target, c.min_w, c.max_w))
            conf = min(1.0, abs(mom_now - mom_th) / max(vol_th, 1e-6))
        elif c.method in ("ml_lr", "ml_rf"):
            tr = hist.copy()
            tr["s_next"] = tr["ret_spyqqq"].shift(-1)
            tr["b_next"] = tr["ret_btceth"].shift(-1)
            tr = tr.dropna(subset=feat + ["s_next", "b_next"]).tail(max(24, int(c.lookback)))
            if len(tr) < 24:
                w_raw = 0.5
                conf = 0.0
            else:
                y = utility_labels(tr["s_next"].to_numpy(float), tr["b_next"].to_numpy(float), bins)
                x = tr[feat].to_numpy(float)
                model = fit_model(c.method, x, y)
                p = model.predict_proba(cur[feat].to_numpy(float))[0]
                p_full = np.zeros(n_bins, dtype=float)
                for cls, pp in zip(getattr(model, "classes_", np.arange(len(p))), p):
                    ci = int(cls)
                    if 0 <= ci < n_bins:
                        p_full[ci] = float(pp)
                if float(np.sum(p_full)) <= 0.0:
                    p_full[:] = uniform
                else:
                    p_full /= float(np.sum(p_full))
                w_raw = float(np.dot(bins, p_full))
                conf = float(np.max(p_full) - uniform) / max(1.0 - uniform, 1e-9)
                conf = float(np.clip(conf, 0.0, 1.0))
        elif c.method == "regime_rf":
            tr = hist.copy()
            tr["s_next"] = tr["ret_spyqqq"].shift(-1)
            tr["b_next"] = tr["ret_btceth"].shift(-1)
            tr["regime"] = (tr["b_next"] > tr["s_next"]).astype(int)
            tr = tr.dropna(subset=feat + ["regime"]).tail(max(24, int(c.lookback)))
            if len(tr) < 24:
                w_raw = 0.5
                conf = 0.0
            else:
                x = tr[feat].to_numpy(float)
                y = tr["regime"].to_numpy(int)
                model = fit_model("regime_rf", x, y)
                p = model.predict_proba(cur[feat].to_numpy(float))[0]
                p_btc = 0.5
                for cls, pp in zip(getattr(model, "classes_", np.arange(len(p))), p):
                    if int(cls) == 1:
                        p_btc = float(pp)
                w_raw = float(1.0 - p_btc)
                conf = float(np.clip(abs(p_btc - 0.5) * 2.0, 0.0, 1.0))
        elif c.method == "ml_dual_ridge":
            tr = hist.copy()
            tr["s_next"] = tr["ret_spyqqq"].shift(-1)
            tr["b_next"] = tr["ret_btceth"].shift(-1)
            tr = tr.dropna(subset=feat + ["s_next", "b_next"]).tail(max(24, int(c.lookback)))
            if len(tr) < 24:
                w_raw = 0.5
                conf = 0.0
            else:
                x = tr[feat].to_numpy(float)
                ms = fit_regressor("ridge", x, tr["s_next"].to_numpy(float))
                mb = fit_regressor("ridge", x, tr["b_next"].to_numpy(float))
                x_cur = cur[feat].to_numpy(float)
                mu_s = float(ms.predict(x_cur)[0])
                mu_b = float(mb.predict(x_cur)[0])
                hs = hist["ret_spyqqq"].tail(max(12, min(36, int(c.lookback)))).to_numpy(float)
                hb = hist["ret_btceth"].tail(max(12, min(36, int(c.lookback)))).to_numpy(float)
                if len(hs) < 6 or len(hb) < 6:
                    w_raw = 0.5
                else:
                    v_s = float(np.var(hs, ddof=1))
                    v_b = float(np.var(hb, ddof=1))
                    cov = float(np.cov(hs, hb, ddof=1)[0, 1])
                    risk_aversion = max(0.8, float(c.dd_penalty))
                    den = 2.0 * risk_aversion * max(v_s + v_b - 2.0 * cov, 1e-8)
                    num = (mu_s - mu_b) - 2.0 * risk_aversion * (cov - v_b)
                    w_raw = float(np.clip(num / den, 0.0, 1.0))
                spread_mu = abs(mu_s - mu_b)
                spread_sigma = float(np.std(hb - hs)) if len(hb) == len(hs) and len(hs) >= 6 else 0.0
                conf = float(np.clip(np.tanh(spread_mu / max(spread_sigma, 1e-6)), 0.0, 1.0))
        else:
            raise ValueError(f"unsupported candidate method {c.method}")

        w_raw = float(np.clip(w_raw, c.min_w, c.max_w))
        w_sm = float(c.smoothing * w_prev + (1.0 - c.smoothing) * w_raw)
        w = float(np.clip(w_prev + np.clip(w_sm - w_prev, -c.max_step, c.max_step), c.min_w, c.max_w))

        gross = float(c.gross_base + c.gross_conf_scale * conf)
        gross = float(np.clip(gross, 0.85, c.max_gross))
        if c.target_vol > 0.0:
            r_hist = (w * hist["ret_spyqqq"] + (1.0 - w) * hist["ret_btceth"]).tail(c.vol_lb)
            rv = float(r_hist.std()) if len(r_hist) >= 3 else 0.0
            if rv > 1e-6:
                gross = min(gross, float(c.target_vol) / rv)
                gross = float(np.clip(gross, 0.85, c.max_gross))

        r_gross = float(gross * (w * rs + (1.0 - w) * rb))
        turn = abs(w - w_prev)
        gchg = abs(gross - g_prev)
        cost = float(c.turn_cost_bps * 1e-4 * turn + c.gross_cost_bps * 1e-4 * gchg)
        r_net = r_gross - cost
        rows.append(
            {
                "month_end": cur.index[0],
                "ret_spyqqq": rs,
                "ret_btceth": rb,
                "weight_spyqqq": w,
                "weight_btceth": 1.0 - w,
                "gross_leverage": gross,
                "ret_gross": r_gross,
                "cost": cost,
                "ret_net": r_net,
                "turnover_abs": turn,
            }
        )
        w_prev = w
        g_prev = gross
    out = pd.DataFrame(rows)
    out["month_end"] = pd.to_datetime(out["month_end"], utc=True, errors="coerce")
    out = out.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    return out


def score(metrics: Dict, objective: str) -> float:
    t = metrics["test"]
    avg_w = float(metrics.get("avg_weight_spyqqq", 0.5))
    weight_std = float(metrics.get("weight_std_spyqqq", 0.0))
    turnover = float(metrics.get("avg_turnover_abs", 0.0))
    diversity = max(0.0, 1.0 - abs(avg_w - 0.5) * 2.0)
    activity = min(1.0, turnover / 0.08)
    robust_tail = max(0.0, float(metrics.get("bootstrap_p10_avg_monthly_pnl", 0.0)))
    if objective == "return":
        return float(1.0 * t["end_equity"] + 200.0 * t["calmar"] - 600.0 * t["max_drawdown"])
    if objective == "calmar":
        return float(180.0 * t["calmar"] + 0.8 * t["avg_monthly_pnl"] - 60.0 * t["max_drawdown"] + 20.0 * robust_tail / 100.0)
    if objective == "robust":
        return float(
            0.45 * t["end_equity"]
            + 140.0 * t["calmar"]
            + 1.4 * t["avg_monthly_pnl"]
            - 360.0 * t["max_drawdown"]
            + 1500.0 * diversity
            + 700.0 * activity
            + 250.0 * min(1.0, weight_std / 0.20)
            + 25.0 * robust_tail / 100.0
        )
    return float(
        0.55 * t["end_equity"]
        + 120.0 * t["calmar"]
        + 1.5 * t["avg_monthly_pnl"]
        - 220.0 * t["max_drawdown"]
        + 35.0 * diversity
        + 18.0 * activity
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spyqqq-monthly", default="combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet")
    ap.add_argument("--btceth-monthly", default="other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet")
    ap.add_argument("--out-dir", default="step4_out")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--min-history-months", type=int, default=24)
    ap.add_argument("--objective", choices=["return", "balanced", "calmar", "robust"], default="return")
    ap.add_argument("--max-drawdown-cap", type=float, default=0.30)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--bootstrap-block-months", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    opt_dir = out_dir / "optimization"
    bt_dir = out_dir / "backtest"
    ensure_dir(opt_dir)
    ensure_dir(bt_dir)

    spy = read_monthly(Path(args.spyqqq_monthly).resolve())
    btc = read_monthly(Path(args.btceth_monthly).resolve())
    rs = pd.Series(spy["ret"].to_numpy(float), index=pd.DatetimeIndex(spy["month_end"]))
    rb = pd.Series(btc["ret"].to_numpy(float), index=pd.DatetimeIndex(btc["month_end"]))
    idx = pd.DatetimeIndex(sorted(rs.index.intersection(rb.index)))
    if len(idx) < 48:
        raise RuntimeError(f"need >=48 overlap months, got {len(idx)}")
    rs = rs.reindex(idx).fillna(0.0)
    rb = rb.reindex(idx).fillna(0.0)
    split_i = int(max(24, min(len(idx) - 12, round(float(args.train_ratio) * len(idx)))))
    split_month = idx[split_i - 1]
    log(f"overlap={len(idx)} split={split_i} objective={args.objective}")

    df = build_features(rs, rb)
    bins = np.linspace(0.0, 1.0, 11)
    rows = []

    for c in candidates():
        try:
            path = simulate(df, c, bins, args.min_history_months)
            ridx = pd.DatetimeIndex(path["month_end"])
            r = pd.Series(path["ret_net"].to_numpy(float), index=ridx)
            r_train = r[ridx <= split_month]
            r_test = r[ridx > split_month]
            m = {
                "full": perf(r, args.start_capital),
                "train": perf(r_train, args.start_capital),
                "test": perf(r_test, args.start_capital),
                "stress_1_25_avg_monthly_pnl": float(np.mean(path["ret_gross"] - 1.25 * path["cost"])) * args.start_capital,
                "bootstrap_p10_avg_monthly_pnl": bootstrap_p10_avg_monthly_pnl(
                    r_test, args.start_capital, args.bootstrap_samples, args.bootstrap_block_months
                ),
                "avg_turnover_abs": float(path["turnover_abs"].mean()),
                "avg_gross_leverage": float(path["gross_leverage"].mean()),
                "avg_weight_spyqqq": float(path["weight_spyqqq"].mean()),
                "weight_std_spyqqq": float(path["weight_spyqqq"].std(ddof=0)),
            }
            m["train_test_calmar_gap_abs"] = float(abs(m["train"]["calmar"] - m["test"]["calmar"]))
            sc = score(m, args.objective)
            robust = bool(
                (m["test"]["max_drawdown"] <= args.max_drawdown_cap)
                and (m["full"]["max_drawdown"] <= 1.35 * args.max_drawdown_cap)
                and (m["test"]["avg_monthly_pnl"] > 0.0)
                and (m["stress_1_25_avg_monthly_pnl"] > 0.0)
                and (m["bootstrap_p10_avg_monthly_pnl"] > -25.0)
                and (m["train_test_calmar_gap_abs"] <= 2.0)
            )
            row = {
                "name": c.name,
                "config": asdict(c),
                "score": float(sc),
                "robust_pass": robust,
                "metrics": m,
                "path_file": str((opt_dir / f"{c.name}_path.parquet").resolve()),
            }
            path.to_parquet(Path(row["path_file"]), index=False)
            rows.append(row)
            log(
                f"{c.name} score={sc:.2f} robust={robust} "
                f"test_end={m['test']['end_equity']:.0f} test_avg={m['test']['avg_monthly_pnl']:.2f} "
                f"test_dd={m['test']['max_drawdown']:.4f}"
            )
        except Exception as e:
            rows.append({"name": c.name, "config": asdict(c), "status": "error", "error": str(e)})
            log(f"{c.name} error: {e}")

    valid = [r for r in rows if "metrics" in r]
    robust = [r for r in valid if bool(r["robust_pass"])]
    best = max((robust if robust else valid), key=lambda x: float(x["score"]))
    best_path = pd.read_parquet(best["path_file"])
    bidx = pd.DatetimeIndex(pd.to_datetime(best_path["month_end"], utc=True, errors="coerce"))
    r_sel = pd.Series(best_path["ret_net"].to_numpy(float), index=bidx).fillna(0.0)
    eq_sel = args.start_capital * (1.0 + r_sel).cumprod()
    eq_spy = args.start_capital * (1.0 + rs.reindex(bidx).fillna(0.0)).cumprod()
    eq_btc = args.start_capital * (1.0 + rb.reindex(bidx).fillna(0.0)).cumprod()
    eq_50 = args.start_capital * (1.0 + 0.5 * rs.reindex(bidx).fillna(0.0) + 0.5 * rb.reindex(bidx).fillna(0.0)).cumprod()

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 3.0, 1.3, 1.2])
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2]); ax3 = fig.add_subplot(gs[3])
    ax0.axis("off")
    txt = [
        f"Selected: {best['name']} | Objective={args.objective} | Score={float(best['score']):.2f}",
        (
            f"Test End ${float(best['metrics']['test']['end_equity']):,.0f} | "
            f"Avg monthly ${float(best['metrics']['test']['avg_monthly_pnl']):,.1f} | "
            f"DD {100.0*float(best['metrics']['test']['max_drawdown']):.2f}% | "
            f"Calmar {float(best['metrics']['test']['calmar']):.2f}"
        ),
        (
            f"Avg wt SPY+QQQ {100.0*float(best['metrics'].get('avg_weight_spyqqq', 0.5)):.1f}% | "
            f"Weight std {float(best['metrics'].get('weight_std_spyqqq', 0.0)):.3f} | "
            f"Avg turnover {float(best['metrics'].get('avg_turnover_abs', 0.0)):.3f}"
        ),
        (
            f"Benchmarks (same window): SPY+QQQ End ${float(perf(rs.reindex(bidx), args.start_capital)['end_equity']):,.0f}, "
            f"BTC+ETH End ${float(perf(rb.reindex(bidx), args.start_capital)['end_equity']):,.0f}, "
            f"50/50 End ${float(perf(0.5*rs.reindex(bidx)+0.5*rb.reindex(bidx), args.start_capital)['end_equity']):,.0f}"
        ),
        f"Cross-pair monthly corr: {float(rs.corr(rb)):.3f}",
    ]
    ax0.text(0.01, 0.98, "\n".join(txt), va="top", ha="left", fontsize=10, bbox={"boxstyle":"round,pad=0.45","facecolor":"white","alpha":0.90,"edgecolor":"#999999"})
    ax1.plot(bidx, eq_sel.values, color="#1d3557", linewidth=2.3, label="Step 4 selected")
    ax1.plot(bidx, eq_spy.values, color="#2a9d8f", linewidth=1.2, alpha=0.78, label="SPY+QQQ sleeve")
    ax1.plot(bidx, eq_btc.values, color="#e76f51", linewidth=1.2, alpha=0.78, label="BTC+ETH sleeve")
    ax1.plot(bidx, eq_50.values, color="#457b9d", linewidth=1.2, alpha=0.78, label="Static 50/50")
    ax1.set_title("Step 4 Allocation: SPY+QQQ + BTC+ETH")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper left", ncol=2, framealpha=0.92)
    ax2.plot(bidx, best_path["weight_spyqqq"].to_numpy(float), color="#264653", linewidth=1.7, label="SPY+QQQ wt")
    ax2.plot(bidx, best_path["weight_btceth"].to_numpy(float), color="#f4a261", linewidth=1.7, label="BTC+ETH wt")
    ax2.plot(bidx, best_path["gross_leverage"].to_numpy(float), color="#6a4c93", linewidth=1.5, label="Gross lev")
    ax2.set_ylim(0.0, max(1.45, float(best_path["gross_leverage"].max()) + 0.05))
    ax2.set_title("Weights and Leverage")
    ax2.set_ylabel("Level")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", ncol=3, framealpha=0.92)
    pnl = eq_sel.diff().fillna(eq_sel.iloc[0] - args.start_capital).to_numpy(float)
    ax3.bar(bidx, pnl, width=20, color=np.where(pnl >= 0.0, "#2a9d8f", "#d62828"), alpha=0.86)
    ax3.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax3.set_title("Step 4 Monthly PnL")
    ax3.set_ylabel("PnL ($)")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(bt_dir / "step4_portfolio_curve.png", dpi=160)
    plt.close(fig)

    monthly = pd.DataFrame({"month_end": bidx, "ret": r_sel.to_numpy(float), "equity": eq_sel.to_numpy(float)})
    prev = monthly["equity"].shift(1)
    prev.iloc[0] = float(args.start_capital)
    monthly["pnl"] = monthly["equity"] - prev
    monthly["weight_spyqqq"] = best_path["weight_spyqqq"].to_numpy(float)
    monthly["weight_btceth"] = best_path["weight_btceth"].to_numpy(float)
    monthly["gross_leverage"] = best_path["gross_leverage"].to_numpy(float)
    monthly.to_parquet(bt_dir / "step4_monthly_table.parquet", index=False)

    summary = {
        "meta": {
            "script": "step4_optimize_allocator.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "objective": args.objective,
            "start_capital": float(args.start_capital),
            "train_ratio": float(args.train_ratio),
            "split_month_end_utc": str(split_month),
            "n_overlap_months": int(len(idx)),
            "spyqqq_monthly_source": str(Path(args.spyqqq_monthly).resolve()),
            "btceth_monthly_source": str(Path(args.btceth_monthly).resolve()),
            "max_drawdown_cap": float(args.max_drawdown_cap),
        },
        "selected_candidate": best,
        "benchmarks": {
            "spyqqq": perf(rs.reindex(bidx).fillna(0.0), args.start_capital),
            "btceth": perf(rb.reindex(bidx).fillna(0.0), args.start_capital),
            "equal_50_50": perf(0.5 * rs.reindex(bidx).fillna(0.0) + 0.5 * rb.reindex(bidx).fillna(0.0), args.start_capital),
            "cross_pair_corr_monthly": float(rs.corr(rb)),
        },
        "outputs": {
            "step4_plot": str((bt_dir / "step4_portfolio_curve.png").resolve()),
            "step4_monthly": str((bt_dir / "step4_monthly_table.parquet").resolve()),
            "step4_summary": str((bt_dir / "step4_summary.json").resolve()),
            "optimization_report": str((opt_dir / "step4_optimization_report.json").resolve()),
        },
    }
    with (bt_dir / "step4_summary.json").open("w", encoding="utf-8") as f:
        json.dump(round_obj(summary, 6), f, separators=(",", ":"), ensure_ascii=True)
    with (opt_dir / "step4_optimization_report.json").open("w", encoding="utf-8") as f:
        json.dump(round_obj({"meta": summary["meta"], "selected": best["name"], "candidates": rows}, 6), f, separators=(",", ":"), ensure_ascii=True)
    log(f"selected={best['name']}")
    log(f"wrote {bt_dir / 'step4_summary.json'}")
    log(f"wrote {opt_dir / 'step4_optimization_report.json'}")


if __name__ == "__main__":
    main()
