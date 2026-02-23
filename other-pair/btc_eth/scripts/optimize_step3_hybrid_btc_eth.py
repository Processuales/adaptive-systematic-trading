#!/usr/bin/env python3
"""
BTC/ETH hybrid core-satellite optimizer.

It overlays a passive BTC/ETH core on top of the existing Step 3 active sleeve,
searches robust candidates, and can promote the best candidate into step3_out/backtest.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APP = "BTC-ETH-HYBRID"
SCRIPT_VERSION = "1.2.0"


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", flush=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=True)


def round_obj(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, dict):
        return {k: round_obj(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_obj(v, ndigits) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, ndigits)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        t = part.strip()
        if t:
            out.append(float(t))
    return out


def parse_str_list(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        t = part.strip()
        if t:
            out.append(t)
    return out


def normalize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    x = df.copy()
    x["month_end"] = pd.to_datetime(x["month_end"], utc=True, errors="coerce")
    x = x.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    for c in ("equity", "pnl", "ret", "trades"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        else:
            x[c] = 0.0
    return x


def load_alias_monthly_ret(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing alias file: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Alias file missing required columns date/close: {path}")
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["date", "close"])
    x = x[x["close"] > 0.0].sort_values("date").drop_duplicates(subset=["date"])
    if x.empty:
        raise RuntimeError(f"No valid rows in alias file: {path}")
    close_m = x.set_index("date")["close"].resample("ME").last()
    return close_m.pct_change().fillna(0.0).astype(float)


def perf_from_equity(eq: pd.Series, start_capital: float) -> Dict[str, float]:
    if eq.empty:
        return {
            "start_time_utc": None,
            "end_time_utc": None,
            "end_equity": float(start_capital),
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": None,
            "n_days": 0,
        }
    x = pd.Series(eq, copy=True).astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "start_time_utc": None,
            "end_time_utc": None,
            "end_equity": float(start_capital),
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": None,
            "n_days": 0,
        }
    n_days = int(max(1, (x.index.max() - x.index.min()).days))
    years = max(n_days / 365.25, 1.0 / 365.25)
    end_eq = float(x.iloc[-1])
    ratio = max(end_eq / max(float(start_capital), 1e-9), 1e-9)
    cagr = float(ratio ** (1.0 / years) - 1.0)
    roll_max = x.cummax()
    dd = (x / roll_max - 1.0).min()
    mdd = abs(float(dd)) if np.isfinite(dd) else 0.0
    calmar = (cagr / mdd) if mdd > 1e-9 else None
    return {
        "start_time_utc": str(x.index.min()),
        "end_time_utc": str(x.index.max()),
        "end_equity": end_eq,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": calmar,
        "n_days": n_days,
    }


def monthly_table_from_ret(
    ret: pd.Series,
    start_capital: float,
    trades_series: pd.Series | None = None,
) -> pd.DataFrame:
    r = pd.Series(ret, copy=True).fillna(0.0).astype(float)
    idx = pd.to_datetime(r.index, utc=True, errors="coerce")
    r.index = idx
    r = r[~r.index.isna()].sort_index()
    if r.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    eq = float(start_capital) * (1.0 + r).cumprod()
    prev = eq.shift(1)
    prev.iloc[0] = float(start_capital)
    pnl = eq - prev
    if trades_series is None:
        trades = pd.Series(0.0, index=r.index, dtype=float)
    else:
        t = pd.Series(trades_series, copy=True).astype(float)
        t.index = pd.to_datetime(t.index, utc=True, errors="coerce")
        t = t[~t.index.isna()].sort_index()
        trades = t.reindex(r.index).fillna(0.0)
    return pd.DataFrame(
        {
            "month_end": r.index,
            "equity": eq.to_numpy(dtype=float),
            "pnl": pnl.to_numpy(dtype=float),
            "ret": r.to_numpy(dtype=float),
            "trades": trades.to_numpy(dtype=float),
        }
    )


def q(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"p05": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean": float(np.mean(arr)),
    }


def bootstrap_summary(
    monthly: pd.DataFrame,
    start_capital: float,
    n_samples: int,
    block_months: int,
    seed: int,
) -> Dict[str, Any]:
    x = normalize_monthly(monthly)
    if x.empty:
        return {"enabled": False, "reason": "empty_monthly_table"}
    if n_samples <= 0:
        return {"enabled": False, "reason": "disabled"}
    if block_months < 1:
        return {"enabled": False, "reason": "invalid_block_months"}

    rets = x["ret"].to_numpy(dtype=float)
    pnls = x["pnl"].to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"enabled": False, "reason": "too_few_months"}

    rng = np.random.default_rng(seed)
    avg_pnl_arr: List[float] = []
    med_pnl_arr: List[float] = []
    pos_rate_arr: List[float] = []
    end_eq_arr: List[float] = []
    dd_arr: List[float] = []
    calmar_arr: List[float] = []

    for _ in range(n_samples):
        idxs: List[int] = []
        while len(idxs) < n:
            start_i = int(rng.integers(0, n))
            for j in range(block_months):
                idxs.append((start_i + j) % n)
                if len(idxs) >= n:
                    break
        idx = np.array(idxs[:n], dtype=int)
        r_s = rets[idx]
        p_s = pnls[idx]

        eq_path = float(start_capital) * np.cumprod(1.0 + r_s)
        end_eq = float(eq_path[-1]) if len(eq_path) else float(start_capital)
        roll_max = np.maximum.accumulate(eq_path)
        dd = np.min(eq_path / np.maximum(roll_max, 1e-9) - 1.0) if len(eq_path) else 0.0
        mdd = abs(float(dd))
        years = max(n / 12.0, 1.0 / 12.0)
        cagr = float((max(end_eq / max(float(start_capital), 1e-9), 1e-9) ** (1.0 / years)) - 1.0)
        calmar = (cagr / mdd) if mdd > 1e-9 else np.nan

        avg_pnl_arr.append(float(np.mean(p_s)))
        med_pnl_arr.append(float(np.median(p_s)))
        pos_rate_arr.append(float(np.mean(p_s > 0.0)))
        end_eq_arr.append(end_eq)
        dd_arr.append(mdd)
        if np.isfinite(calmar):
            calmar_arr.append(float(calmar))

    return {
        "enabled": True,
        "n_samples": int(n_samples),
        "block_months": int(block_months),
        "seed": int(seed),
        "avg_monthly_pnl": q(np.array(avg_pnl_arr, dtype=float)),
        "median_monthly_pnl": q(np.array(med_pnl_arr, dtype=float)),
        "monthly_positive_rate": q(np.array(pos_rate_arr, dtype=float)),
        "end_equity": q(np.array(end_eq_arr, dtype=float)),
        "max_drawdown": q(np.array(dd_arr, dtype=float)),
        "calmar": q(np.array(calmar_arr, dtype=float)) if calmar_arr else q(np.array([], dtype=float)),
    }


def summarize_portfolio(monthly: pd.DataFrame, start_capital: float) -> Dict[str, Any]:
    x = normalize_monthly(monthly)
    if x.empty:
        return {
            "monthly": x,
            "perf": perf_from_equity(pd.Series(dtype=float), start_capital),
            "avg_monthly_pnl": 0.0,
            "median_monthly_pnl": 0.0,
            "avg_monthly_trades": 0.0,
            "monthly_positive_rate": 0.0,
            "monthly_negative_rate": 0.0,
            "best_month": {},
            "worst_month": {},
        }
    eq = pd.Series(x["equity"].to_numpy(dtype=float), index=x["month_end"], dtype=float)
    perf = perf_from_equity(eq, start_capital)
    avg_pnl = float(x["pnl"].mean())
    med_pnl = float(x["pnl"].median())
    avg_trades = float(x["trades"].mean())
    pos_rate = float((x["pnl"] > 0.0).mean())
    neg_rate = float((x["pnl"] < 0.0).mean())
    i_best = int(x["pnl"].idxmax())
    i_worst = int(x["pnl"].idxmin())
    best = x.iloc[i_best]
    worst = x.iloc[i_worst]
    return {
        "monthly": x,
        "perf": perf,
        "avg_monthly_pnl": avg_pnl,
        "median_monthly_pnl": med_pnl,
        "avg_monthly_trades": avg_trades,
        "monthly_positive_rate": pos_rate,
        "monthly_negative_rate": neg_rate,
        "best_month": {
            "month_end": str(best["month_end"]),
            "pnl": float(best["pnl"]),
            "ret": float(best["ret"]),
            "trades": float(best["trades"]),
        },
        "worst_month": {
            "month_end": str(worst["month_end"]),
            "pnl": float(worst["pnl"]),
            "ret": float(worst["ret"]),
            "trades": float(worst["trades"]),
        },
    }


def metric_snapshot(summary: Dict[str, Any]) -> Dict[str, float]:
    p = summary.get("portfolio") or {}
    perf = p.get("dual_perf") or {}
    s125 = {}
    s150 = {}
    for row in (p.get("cost_stress_tests") or []):
        mult = float(row.get("cost_multiplier") or 0.0)
        if abs(mult - 1.25) <= 0.03:
            s125 = row
        elif abs(mult - 1.50) <= 0.03:
            s150 = row
    boot = p.get("bootstrap") or {}
    boot_avg = boot.get("avg_monthly_pnl") or {}
    return {
        "avg_monthly_pnl": float(p.get("avg_monthly_pnl") or 0.0),
        "calmar": float(perf.get("calmar") or 0.0),
        "max_drawdown": float(perf.get("max_drawdown") or 1.0),
        "end_equity": float(perf.get("end_equity") or 0.0),
        "avg_monthly_trades": float(p.get("avg_monthly_trades") or 0.0),
        "positive_month_rate": float(p.get("monthly_positive_rate") or 0.0),
        "stress_1_25_avg_monthly_pnl": float(s125.get("avg_monthly_pnl") or 0.0),
        "stress_1_50_avg_monthly_pnl": float(s150.get("avg_monthly_pnl") or 0.0),
        "bootstrap_p10_avg_monthly_pnl": float(boot_avg.get("p10") or 0.0),
    }


def score_snapshot(m: Dict[str, float]) -> float:
    return float(
        0.70 * m["avg_monthly_pnl"]
        + 65.0 * m["calmar"]
        - 70.0 * m["max_drawdown"]
        + 0.25 * m["stress_1_25_avg_monthly_pnl"]
        + 0.20 * m["stress_1_50_avg_monthly_pnl"]
        + 0.12 * m["bootstrap_p10_avg_monthly_pnl"]
    )


def derive_active_stress_deltas(
    active_summary: Dict[str, Any],
    start_capital: float,
) -> Dict[float, float]:
    p = active_summary.get("portfolio") or {}
    base_avg = float(p.get("avg_monthly_pnl") or 0.0)
    out: Dict[float, float] = {1.0: 0.0}
    known: List[tuple[float, float]] = []
    for row in (p.get("cost_stress_tests") or []):
        mult = float(row.get("cost_multiplier") or 0.0)
        if mult <= 1.0:
            continue
        avg_m = float(row.get("avg_monthly_pnl") or 0.0)
        delta_ret = (avg_m - base_avg) / max(float(start_capital), 1e-9)
        out[mult] = delta_ret
        known.append((mult, delta_ret))
    if 2.0 not in out:
        if known:
            known_sorted = sorted(known, key=lambda z: z[0])
            m_hi, d_hi = known_sorted[-1]
            slope = d_hi / max(m_hi - 1.0, 1e-9)
            out[2.0] = slope * (2.0 - 1.0)
        else:
            out[2.0] = 0.0
    return out


def build_core_weight_series(
    idx: pd.DatetimeIndex,
    btc_ret: pd.Series,
    eth_ret: pd.Series,
    core_mode: str,
    btc_share: float,
) -> pd.Series:
    base_share = float(np.clip(btc_share, 0.0, 1.0))
    if core_mode == "fixed":
        return pd.Series(base_share, index=idx, dtype=float)
    if core_mode == "vol_parity_6m":
        vb = btc_ret.rolling(6, min_periods=3).std().replace(0.0, np.nan)
        ve = eth_ret.rolling(6, min_periods=3).std().replace(0.0, np.nan)
        inv_b = 1.0 / vb
        inv_e = 1.0 / ve
        w = inv_b / (inv_b + inv_e)
        w = w.reindex(idx).fillna(base_share)
        return w.clip(lower=0.10, upper=0.90).astype(float)
    if core_mode == "trend_guard_6m":
        mom_b = btc_ret.rolling(6, min_periods=3).sum().reindex(idx)
        mom_e = eth_ret.rolling(6, min_periods=3).sum().reindex(idx)
        spread = (mom_b - mom_e).fillna(0.0)
        spread_scale = spread.rolling(12, min_periods=6).std().replace(0.0, np.nan).fillna(0.15)
        z = (spread / spread_scale).clip(lower=-3.0, upper=3.0)
        w = base_share + 0.18 * np.tanh(z)
        vb = btc_ret.rolling(6, min_periods=3).std().reindex(idx).fillna(0.0)
        ve = eth_ret.rolling(6, min_periods=3).std().reindex(idx).fillna(0.0)
        risk_off = (mom_b.fillna(0.0) < 0.0) & (vb > ve)
        w = np.where(risk_off.to_numpy(bool), np.minimum(w, 0.35), w)
        return pd.Series(w, index=idx, dtype=float).clip(lower=0.10, upper=0.90)
    if core_mode == "risk_parity_mom_6m":
        vb = btc_ret.rolling(6, min_periods=3).std().replace(0.0, np.nan)
        ve = eth_ret.rolling(6, min_periods=3).std().replace(0.0, np.nan)
        inv_b = 1.0 / vb
        inv_e = 1.0 / ve
        w_vp = (inv_b / (inv_b + inv_e)).reindex(idx).fillna(base_share).clip(lower=0.10, upper=0.90)
        mom_b = btc_ret.rolling(6, min_periods=3).sum().reindex(idx)
        mom_e = eth_ret.rolling(6, min_periods=3).sum().reindex(idx)
        spread = (mom_b - mom_e).fillna(0.0)
        spread_scale = spread.rolling(12, min_periods=6).std().replace(0.0, np.nan).fillna(0.15)
        z = (spread / spread_scale).clip(lower=-3.0, upper=3.0)
        w_m = (base_share + 0.12 * np.tanh(z)).clip(lower=0.10, upper=0.90)
        w = (0.70 * w_vp + 0.30 * w_m).astype(float)
        risk_off = (mom_b.fillna(0.0) < 0.0) & (vb.reindex(idx).fillna(0.0) > ve.reindex(idx).fillna(0.0))
        w = np.where(risk_off.to_numpy(bool), np.minimum(w, 0.30), w)
        return pd.Series(w, index=idx, dtype=float).clip(lower=0.10, upper=0.90)
    raise ValueError(f"Unsupported core mode: {core_mode}")


@dataclass
class Candidate:
    name: str
    core_fraction: float
    btc_core_share: float
    active_scale: float
    core_mode: str


def make_candidates(
    core_fractions: List[float],
    core_btc_shares: List[float],
    active_scales: List[float],
    core_modes: List[str],
) -> List[Candidate]:
    out: List[Candidate] = []
    for c in core_fractions:
        for b in core_btc_shares:
            for a in active_scales:
                for m in core_modes:
                    name = f"hybrid_core{c:.2f}_btc{b:.2f}_active{a:.2f}_{m}"
                    out.append(
                        Candidate(
                            name=name.replace(".", "_"),
                            core_fraction=float(c),
                            btc_core_share=float(b),
                            active_scale=float(a),
                            core_mode=m,
                        )
                    )
    return out


def build_hybrid_monthly(
    active_ret: pd.Series,
    active_trades: pd.Series,
    btc_ret: pd.Series,
    eth_ret: pd.Series,
    c: Candidate,
    start_capital: float,
    delta_active_ret: float = 0.0,
) -> Dict[str, Any]:
    idx = active_ret.index.intersection(btc_ret.index).intersection(eth_ret.index)
    idx = pd.DatetimeIndex(sorted(idx))
    if len(idx) < 8:
        raise RuntimeError("Insufficient overlap months for hybrid evaluation.")
    a = active_ret.reindex(idx).fillna(0.0)
    t = active_trades.reindex(idx).fillna(0.0)
    b = btc_ret.reindex(idx).fillna(0.0)
    e = eth_ret.reindex(idx).fillna(0.0)
    core_w_btc = build_core_weight_series(
        idx=idx,
        btc_ret=b,
        eth_ret=e,
        core_mode=c.core_mode,
        btc_share=c.btc_core_share,
    )
    core_ret = core_w_btc * b + (1.0 - core_w_btc) * e
    active_sleeve = float(np.clip((1.0 - c.core_fraction) * c.active_scale, 0.0, 1.0))
    core_sleeve = float(np.clip(c.core_fraction, 0.0, 1.0))
    total_ret = core_sleeve * core_ret + active_sleeve * (a + float(delta_active_ret))
    monthly = monthly_table_from_ret(total_ret, start_capital=start_capital, trades_series=t)
    stats = summarize_portfolio(monthly, start_capital=start_capital)
    stats["active_sleeve_share"] = active_sleeve
    stats["core_sleeve_share"] = core_sleeve
    stats["core_btc_weight_mean"] = float(core_w_btc.mean())
    stats["core_btc_weight_min"] = float(core_w_btc.min())
    stats["core_btc_weight_max"] = float(core_w_btc.max())
    return stats


def robust_pass(
    metrics: Dict[str, float],
    active_sleeve_share: float,
    min_stress125_avg_monthly_pnl: float,
    min_stress150_avg_monthly_pnl: float,
    min_bootstrap_p10_monthly_pnl: float,
    max_drawdown_cap: float,
    min_active_sleeve_share: float,
) -> tuple[bool, List[str]]:
    fails: List[str] = []
    if metrics["stress_1_25_avg_monthly_pnl"] < min_stress125_avg_monthly_pnl:
        fails.append(
            f"stress_1_25_avg_monthly_pnl {metrics['stress_1_25_avg_monthly_pnl']:.2f} "
            f"< {min_stress125_avg_monthly_pnl:.2f}"
        )
    if metrics["stress_1_50_avg_monthly_pnl"] < min_stress150_avg_monthly_pnl:
        fails.append(
            f"stress_1_50_avg_monthly_pnl {metrics['stress_1_50_avg_monthly_pnl']:.2f} "
            f"< {min_stress150_avg_monthly_pnl:.2f}"
        )
    if metrics["bootstrap_p10_avg_monthly_pnl"] < min_bootstrap_p10_monthly_pnl:
        fails.append(
            f"bootstrap_p10_avg_monthly_pnl {metrics['bootstrap_p10_avg_monthly_pnl']:.2f} "
            f"< {min_bootstrap_p10_monthly_pnl:.2f}"
        )
    if metrics["max_drawdown"] > max_drawdown_cap:
        fails.append(f"max_drawdown {metrics['max_drawdown']:.4f} > {max_drawdown_cap:.4f}")
    if active_sleeve_share < min_active_sleeve_share:
        fails.append(f"active_sleeve_share {active_sleeve_share:.3f} < {min_active_sleeve_share:.3f}")
    return (len(fails) == 0), fails


def should_promote(
    best: Dict[str, float],
    baseline: Dict[str, float],
    avg_monthly_pnl_margin: float,
    calmar_tolerance: float,
    drawdown_tolerance: float,
    absolute_max_drawdown: float,
) -> bool:
    pnl_gain = float(best["avg_monthly_pnl"] - baseline["avg_monthly_pnl"])
    if pnl_gain <= 0.0:
        return False

    abs_dd_cap = float(max(0.01, absolute_max_drawdown))
    conservative_dd_gate = min(abs_dd_cap, baseline["max_drawdown"] + drawdown_tolerance)
    high_gain_dd_gate = min(abs_dd_cap, baseline["max_drawdown"] + max(drawdown_tolerance, 0.08))

    if (
        pnl_gain >= avg_monthly_pnl_margin
        and best["calmar"] >= baseline["calmar"] - calmar_tolerance
        and best["max_drawdown"] <= conservative_dd_gate
    ):
        return True

    # If baseline is weak/negative, permit measured drawdown expansion only when
    # gains are large and robustness tails are still positive.
    if (
        pnl_gain >= max(3.0 * avg_monthly_pnl_margin, 40.0)
        and best["calmar"] >= baseline["calmar"] - max(calmar_tolerance, 0.0)
        and best["max_drawdown"] <= high_gain_dd_gate
        and best["stress_1_25_avg_monthly_pnl"] > 0.0
        and best["bootstrap_p10_avg_monthly_pnl"] > 0.0
    ):
        return True

    if (
        pnl_gain >= max(6.0 * avg_monthly_pnl_margin, 80.0)
        and best["calmar"] >= baseline["calmar"]
        and best["max_drawdown"] <= abs_dd_cap
    ):
        return True
    return False


def pick_best_promotable_candidate(
    robust_rows: List[Dict[str, Any]],
    baseline: Dict[str, float],
    avg_monthly_pnl_margin: float,
    calmar_tolerance: float,
    drawdown_tolerance: float,
    absolute_max_drawdown: float,
) -> Dict[str, Any] | None:
    if not robust_rows:
        return None
    ranked = sorted(robust_rows, key=lambda r: float(r.get("score") or -1e18), reverse=True)
    for row in ranked:
        metrics = row.get("metrics") or {}
        if should_promote(
            best=metrics,
            baseline=baseline,
            avg_monthly_pnl_margin=avg_monthly_pnl_margin,
            calmar_tolerance=calmar_tolerance,
            drawdown_tolerance=drawdown_tolerance,
            absolute_max_drawdown=absolute_max_drawdown,
        ):
            return row
    return None


def make_hybrid_plot(
    monthly: pd.DataFrame,
    out_path: Path,
    start_capital: float,
    title: str,
    subtitle: str,
    stats_lines: List[str],
) -> None:
    m = normalize_monthly(monthly)
    if m.empty:
        raise RuntimeError("Cannot draw hybrid chart from empty monthly table.")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [3.2, 1.4, 1.1]},
    )
    ax1.plot(m["month_end"], m["equity"], color="#1d3557", linewidth=2.2, label="Hybrid equity")
    ax1.axhline(start_capital, color="black", linewidth=0.8, linestyle="--", alpha=0.7, label="Start capital")
    ax1.set_title(f"{title}\n{subtitle}")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="lower right", framealpha=0.92)
    ax1.text(
        0.01,
        0.99,
        "\n".join(stats_lines),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#999999"},
    )

    pnl = m["pnl"].to_numpy(dtype=float)
    colors = np.where(pnl >= 0.0, "#2a9d8f", "#d62828")
    ax2.bar(m["month_end"], pnl, width=20, color=colors, alpha=0.86)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_title("Monthly Profit / Loss")
    ax2.set_ylabel("Monthly PnL ($)")
    ax2.grid(alpha=0.2)

    trades = m["trades"].to_numpy(dtype=float)
    ax3.bar(m["month_end"], trades, width=20, color="#264653", alpha=0.74, label="Trades/month")
    ax3.plot(
        m["month_end"],
        pd.Series(trades).rolling(3, min_periods=1).mean().to_numpy(dtype=float),
        color="#e76f51",
        linewidth=1.5,
        label="3-month avg trades",
    )
    ax3.set_title("Trading Activity")
    ax3.set_ylabel("# Trades")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def promote_into_backtest(
    step3_out_dir: Path,
    active_summary: Dict[str, Any],
    best_row: Dict[str, Any],
    promoted_monthly: pd.DataFrame,
    start_capital: float,
) -> None:
    backtest_dir = step3_out_dir / "backtest"
    if not backtest_dir.exists():
        raise FileNotFoundError(f"Missing step3 backtest dir: {backtest_dir}")

    summary_path = backtest_dir / "step3_summary.json"
    monthly_path = backtest_dir / "step3_monthly_table.parquet"
    fig_path = backtest_dir / "step3_dual_portfolio_curve.png"

    backup_dir = step3_out_dir / "backtest_active_baseline"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_summary = backup_dir / "step3_summary_active.json"
    backup_monthly = backup_dir / "step3_monthly_table_active.parquet"
    backup_plot = backup_dir / "step3_dual_portfolio_curve_active.png"
    # Preserve first active baseline snapshot so repeated hybrid promotions
    # do not recursively redefine what "active baseline" means.
    if summary_path.exists() and (not backup_summary.exists()):
        shutil.copy2(summary_path, backup_dir / "step3_summary_active.json")
    if monthly_path.exists() and (not backup_monthly.exists()):
        shutil.copy2(monthly_path, backup_dir / "step3_monthly_table_active.parquet")
    if fig_path.exists() and (not backup_plot.exists()):
        shutil.copy2(fig_path, backup_dir / "step3_dual_portfolio_curve_active.png")

    s = deepcopy(active_summary)
    p = s.setdefault("portfolio", {})
    best_metrics = best_row["metrics"]
    best_perf = best_row["portfolio_dual_perf"]
    p["dual_perf"] = best_perf
    p["avg_monthly_pnl"] = best_metrics["avg_monthly_pnl"]
    p["median_monthly_pnl"] = best_metrics["median_monthly_pnl"]
    p["avg_monthly_trades"] = best_metrics["avg_monthly_trades"]
    p["monthly_positive_rate"] = best_metrics["monthly_positive_rate"]
    p["monthly_negative_rate"] = best_metrics["monthly_negative_rate"]
    p["best_month"] = best_row["best_month"]
    p["worst_month"] = best_row["worst_month"]
    p["cost_stress_tests"] = best_row["cost_stress_tests"]
    p["bootstrap"] = best_row["bootstrap"]

    allocator = p.setdefault("allocator", {})
    allocator["selected"] = "btc_eth_hybrid_core_satellite"
    allocator["core_fraction"] = best_row["config"]["core_fraction"]
    allocator["core_mode"] = best_row["config"]["core_mode"]
    allocator["core_btc_share_input"] = best_row["config"]["btc_core_share"]
    allocator["core_btc_weight_mean"] = best_row["core_btc_weight_mean"]
    allocator["core_btc_weight_min"] = best_row["core_btc_weight_min"]
    allocator["core_btc_weight_max"] = best_row["core_btc_weight_max"]
    allocator["active_scale"] = best_row["config"]["active_scale"]
    allocator["active_sleeve_share"] = best_row["active_sleeve_share"]
    allocator["hybrid_overlay_promoted"] = True

    m = s.setdefault("meta", {})
    m["btc_eth_hybrid_overlay"] = {
        "enabled": True,
        "script": "optimize_step3_hybrid_btc_eth.py",
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": best_row["name"],
        "note": "Core-satellite BTC/ETH overlay promoted over active-only baseline.",
    }

    stats_lines = [
        f"End equity: ${best_metrics['end_equity']:,.0f}",
        f"CAGR: {100.0 * best_metrics['cagr']:.2f}%  |  Max drawdown: {100.0 * best_metrics['max_drawdown']:.2f}%",
        f"Calmar: {best_metrics['calmar']:.2f}",
        f"Avg monthly PnL: ${best_metrics['avg_monthly_pnl']:,.0f}  |  Median: ${best_metrics['median_monthly_pnl']:,.0f}",
        f"Positive months: {100.0 * best_metrics['monthly_positive_rate']:.1f}%  |  Avg trades/month: {best_metrics['avg_monthly_trades']:.1f}",
        (
            "Core mode: "
            f"{best_row['config']['core_mode']} | core={best_row['config']['core_fraction']:.2f} "
            f"| btc core weight avg/min/max={best_row['core_btc_weight_mean']:.2f}/"
            f"{best_row['core_btc_weight_min']:.2f}/{best_row['core_btc_weight_max']:.2f}"
        ),
        f"Stress x1.25 avg monthly PnL: ${best_metrics['stress_1_25_avg_monthly_pnl']:,.0f}",
        f"Stress x1.50 avg monthly PnL: ${best_metrics['stress_1_50_avg_monthly_pnl']:,.0f}",
        f"Bootstrap p10 avg monthly PnL: ${best_metrics['bootstrap_p10_avg_monthly_pnl']:,.0f}",
    ]
    make_hybrid_plot(
        monthly=promoted_monthly,
        out_path=fig_path,
        start_capital=start_capital,
        title="Step 3 Real ML Backtest (BTC + ETH)",
        subtitle="Hybrid core-satellite overlay (passive core + active Step 3 sleeve)",
        stats_lines=stats_lines,
    )

    promoted_monthly.to_parquet(monthly_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(round_obj(s, 6), f, separators=(",", ":"), ensure_ascii=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step3-out-dir", required=True)
    ap.add_argument("--alias-dir", required=True)
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--core-fractions", default="0.20,0.30,0.50,0.70,0.85")
    ap.add_argument("--core-btc-shares", default="0.30,0.50,0.70,0.85")
    ap.add_argument("--active-scales", default="1.00,0.80,0.60,0.40,0.25,0.15")
    ap.add_argument("--core-modes", default="fixed,vol_parity_6m,trend_guard_6m,risk_parity_mom_6m")
    ap.add_argument("--bootstrap-samples", type=int, default=800)
    ap.add_argument("--bootstrap-block-months", type=int, default=6)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--min-stress125-avg-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--min-stress150-avg-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--min-bootstrap-p10-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--max-drawdown-cap", type=float, default=0.22)
    ap.add_argument("--min-active-sleeve-share", type=float, default=0.05)
    ap.add_argument("--promote-avg-monthly-pnl-margin", type=float, default=5.0)
    ap.add_argument("--promote-calmar-tolerance", type=float, default=0.05)
    ap.add_argument("--promote-drawdown-tolerance", type=float, default=0.02)
    ap.add_argument("--promote-absolute-max-drawdown", type=float, default=0.20)
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    step3_out_dir = Path(args.step3_out_dir).resolve()
    alias_dir = Path(args.alias_dir).resolve()
    backtest_dir = step3_out_dir / "backtest"
    active_baseline_dir = step3_out_dir / "backtest_active_baseline"
    tilt_report_path = step3_out_dir / "optimization" / "btc_eth_tilt_search_report.json"
    baseline_summary_path = backtest_dir / "step3_summary.json"
    baseline_monthly_path = backtest_dir / "step3_monthly_table.parquet"
    baseline_source = "backtest"

    # Preferred source: tuned active-only run from BTC/ETH tilt report.
    if tilt_report_path.exists():
        try:
            tilt_report = read_json(tilt_report_path)
            best = tilt_report.get("best_candidate") or {}
            s_path = Path(str(best.get("summary_path") or "")).resolve()
            m_path = s_path.parent / "step3_monthly_table.parquet"
            if s_path.exists() and m_path.exists():
                baseline_summary_path = s_path
                baseline_monthly_path = m_path
                baseline_source = "tilt_best_candidate"
        except Exception:
            pass

    # Fallback source: preserved active baseline snapshot.
    if baseline_source == "backtest":
        if (
            (active_baseline_dir / "step3_summary_active.json").exists()
            and (active_baseline_dir / "step3_monthly_table_active.parquet").exists()
        ):
            baseline_summary_path = active_baseline_dir / "step3_summary_active.json"
            baseline_monthly_path = active_baseline_dir / "step3_monthly_table_active.parquet"
            baseline_source = "backtest_active_baseline"
    if not baseline_summary_path.exists() or not baseline_monthly_path.exists():
        raise FileNotFoundError("Missing baseline Step 3 summary/monthly table for hybrid overlay.")

    active_summary = read_json(baseline_summary_path)
    active_monthly = normalize_monthly(pd.read_parquet(baseline_monthly_path))
    if active_monthly.empty:
        raise RuntimeError("Active monthly table is empty.")

    idx = pd.DatetimeIndex(active_monthly["month_end"])
    active_ret = pd.Series(active_monthly["ret"].to_numpy(dtype=float), index=idx, dtype=float)
    active_trades = pd.Series(active_monthly["trades"].to_numpy(dtype=float), index=idx, dtype=float)

    btc_ret = load_alias_monthly_ret(alias_dir / "qqq_1h_rth_clean.parquet")
    eth_ret = load_alias_monthly_ret(alias_dir / "spy_1h_rth_clean.parquet")
    stress_deltas = derive_active_stress_deltas(active_summary, start_capital=args.start_capital)

    baseline_metrics = metric_snapshot(active_summary)
    baseline_score = score_snapshot(baseline_metrics)
    log(
        "baseline "
        f"avg_pnl={baseline_metrics['avg_monthly_pnl']:.2f} "
        f"calmar={baseline_metrics['calmar']:.4f} "
        f"dd={baseline_metrics['max_drawdown']:.4f} "
        f"score={baseline_score:.4f}"
    )

    core_fractions = parse_float_list(args.core_fractions)
    core_btc_shares = parse_float_list(args.core_btc_shares)
    active_scales = parse_float_list(args.active_scales)
    core_modes = parse_str_list(args.core_modes)
    if not core_fractions or not core_btc_shares or not active_scales or not core_modes:
        raise ValueError("Candidate grids must not be empty.")

    cands = make_candidates(core_fractions, core_btc_shares, active_scales, core_modes)
    rows: List[Dict[str, Any]] = []

    for c in cands:
        base_stats = build_hybrid_monthly(
            active_ret=active_ret,
            active_trades=active_trades,
            btc_ret=btc_ret,
            eth_ret=eth_ret,
            c=c,
            start_capital=args.start_capital,
            delta_active_ret=0.0,
        )
        monthly = base_stats["monthly"]
        bootstrap = bootstrap_summary(
            monthly=monthly,
            start_capital=args.start_capital,
            n_samples=args.bootstrap_samples,
            block_months=args.bootstrap_block_months,
            seed=args.bootstrap_seed,
        )

        stress_rows: List[Dict[str, Any]] = []
        stress_125_avg = 0.0
        stress_150_avg = 0.0
        for mult in (1.25, 1.50, 2.00):
            delta = float(stress_deltas.get(mult, 0.0))
            st = build_hybrid_monthly(
                active_ret=active_ret,
                active_trades=active_trades,
                btc_ret=btc_ret,
                eth_ret=eth_ret,
                c=c,
                start_capital=args.start_capital,
                delta_active_ret=delta,
            )
            stress_row = {
                "cost_multiplier": float(mult),
                "dual_perf": st["perf"],
                "avg_monthly_pnl": st["avg_monthly_pnl"],
                "median_monthly_pnl": st["median_monthly_pnl"],
                "monthly_positive_rate": st["monthly_positive_rate"],
            }
            stress_rows.append(stress_row)
            if abs(mult - 1.25) <= 0.02:
                stress_125_avg = float(st["avg_monthly_pnl"])
            if abs(mult - 1.50) <= 0.02:
                stress_150_avg = float(st["avg_monthly_pnl"])

        boot_avg = bootstrap.get("avg_monthly_pnl") or {}
        metrics = {
            "avg_monthly_pnl": float(base_stats["avg_monthly_pnl"]),
            "median_monthly_pnl": float(base_stats["median_monthly_pnl"]),
            "avg_monthly_trades": float(base_stats["avg_monthly_trades"]),
            "monthly_positive_rate": float(base_stats["monthly_positive_rate"]),
            "monthly_negative_rate": float(base_stats["monthly_negative_rate"]),
            "end_equity": float(base_stats["perf"]["end_equity"]),
            "cagr": float(base_stats["perf"]["cagr"]),
            "max_drawdown": float(base_stats["perf"]["max_drawdown"]),
            "calmar": float(base_stats["perf"]["calmar"] or 0.0),
            "stress_1_25_avg_monthly_pnl": float(stress_125_avg),
            "stress_1_50_avg_monthly_pnl": float(stress_150_avg),
            "bootstrap_p10_avg_monthly_pnl": float(boot_avg.get("p10") or 0.0),
        }
        score = score_snapshot(metrics)
        pass_robust, robust_failures = robust_pass(
            metrics=metrics,
            active_sleeve_share=float(base_stats["active_sleeve_share"]),
            min_stress125_avg_monthly_pnl=args.min_stress125_avg_monthly_pnl,
            min_stress150_avg_monthly_pnl=args.min_stress150_avg_monthly_pnl,
            min_bootstrap_p10_monthly_pnl=args.min_bootstrap_p10_monthly_pnl,
            max_drawdown_cap=args.max_drawdown_cap,
            min_active_sleeve_share=args.min_active_sleeve_share,
        )
        row = {
            "name": c.name,
            "config": {
                "core_fraction": c.core_fraction,
                "btc_core_share": c.btc_core_share,
                "active_scale": c.active_scale,
                "core_mode": c.core_mode,
            },
            "metrics": metrics,
            "score": score,
            "robust_pass": bool(pass_robust),
            "robust_failures": robust_failures,
            "portfolio_dual_perf": base_stats["perf"],
            "cost_stress_tests": stress_rows,
            "bootstrap": bootstrap,
            "best_month": base_stats["best_month"],
            "worst_month": base_stats["worst_month"],
            "active_sleeve_share": float(base_stats["active_sleeve_share"]),
            "core_sleeve_share": float(base_stats["core_sleeve_share"]),
            "core_btc_weight_mean": float(base_stats["core_btc_weight_mean"]),
            "core_btc_weight_min": float(base_stats["core_btc_weight_min"]),
            "core_btc_weight_max": float(base_stats["core_btc_weight_max"]),
            "monthly_table_path": str(step3_out_dir / "hybrid" / f"{c.name}_monthly.parquet"),
        }
        Path(row["monthly_table_path"]).parent.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(Path(row["monthly_table_path"]), index=False)
        rows.append(row)
        log(
            f"candidate={c.name} avg_pnl={metrics['avg_monthly_pnl']:.2f} "
            f"calmar={metrics['calmar']:.4f} dd={metrics['max_drawdown']:.4f} "
            f"stress125={metrics['stress_1_25_avg_monthly_pnl']:.2f} "
            f"boot_p10={metrics['bootstrap_p10_avg_monthly_pnl']:.2f} "
            f"score={score:.4f} robust={pass_robust}"
        )

    robust_rows = [r for r in rows if r["robust_pass"]]
    if robust_rows:
        best = max(robust_rows, key=lambda r: float(r["score"]))
    else:
        best = max(rows, key=lambda r: float(r["score"]))
    best_metrics = best["metrics"]

    promote_row: Dict[str, Any] | None = None
    if (not args.no_promote) and robust_rows:
        promote_row = pick_best_promotable_candidate(
            robust_rows=robust_rows,
            baseline=baseline_metrics,
            avg_monthly_pnl_margin=args.promote_avg_monthly_pnl_margin,
            calmar_tolerance=args.promote_calmar_tolerance,
            drawdown_tolerance=args.promote_drawdown_tolerance,
            absolute_max_drawdown=args.promote_absolute_max_drawdown,
        )
        if promote_row is not None and promote_row["name"] != best["name"]:
            log(
                "top-score candidate failed promotion gate; "
                f"using best promotable candidate {promote_row['name']} instead"
            )
    promote = promote_row is not None

    promoted_from = "active_baseline"
    if promote:
        row = promote_row or best
        promoted_monthly = normalize_monthly(pd.read_parquet(row["monthly_table_path"]))
        promote_into_backtest(
            step3_out_dir=step3_out_dir,
            active_summary=active_summary,
            best_row=row,
            promoted_monthly=promoted_monthly,
            start_capital=args.start_capital,
        )
        promoted_from = str(row["name"])
        log(f"promoted candidate -> step3_out/backtest: {row['name']}")
    else:
        log("no promotion: active baseline remains selected")

    report = {
        "meta": {
            "script": "optimize_step3_hybrid_btc_eth.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "step3_out_dir": str(step3_out_dir),
            "alias_dir": str(alias_dir),
            "start_capital": float(args.start_capital),
            "baseline_source": baseline_source,
            "gates": {
                "min_stress125_avg_monthly_pnl": float(args.min_stress125_avg_monthly_pnl),
                "min_stress150_avg_monthly_pnl": float(args.min_stress150_avg_monthly_pnl),
                "min_bootstrap_p10_monthly_pnl": float(args.min_bootstrap_p10_monthly_pnl),
                "max_drawdown_cap": float(args.max_drawdown_cap),
                "min_active_sleeve_share": float(args.min_active_sleeve_share),
                "promote_absolute_max_drawdown": float(args.promote_absolute_max_drawdown),
            },
            "promotion_rule": (
                "Promote only if robust gates pass; allow measured drawdown expansion for large "
                "PnL gains, but never above promote_absolute_max_drawdown."
            ),
        },
        "baseline": {
            "summary_path": str(baseline_summary_path),
            "monthly_path": str(baseline_monthly_path),
            "metrics": baseline_metrics,
            "score": baseline_score,
        },
        "best_candidate": best,
        "best_promotable_candidate": promote_row,
        "promoted": bool(promote),
        "promoted_from": promoted_from,
        "candidates": rows,
    }
    report_path = step3_out_dir / "optimization" / "btc_eth_hybrid_overlay_report.json"
    write_json(report_path, round_obj(report, 6))
    log(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
