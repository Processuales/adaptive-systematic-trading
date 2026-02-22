#!/usr/bin/env python3
"""
Build overfitting diagnostics for BTC/ETH Step 3 active baseline vs promoted hybrid.

Outputs a JSON report with:
- headline summary deltas
- time-split stability checks
- concentration checks
- rolling-window fragility checks
- sign-flip randomization significance test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def metric_snapshot(summary: Dict[str, Any]) -> Dict[str, float]:
    p = summary.get("portfolio") or {}
    perf = p.get("dual_perf") or {}
    stress = p.get("cost_stress_tests") or []
    s125 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.25) <= 0.03), {})
    s150 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.50) <= 0.03), {})
    boot = p.get("bootstrap") or {}
    bavg = boot.get("avg_monthly_pnl") or {}
    return {
        "avg_monthly_pnl": float(p.get("avg_monthly_pnl") or 0.0),
        "end_equity": float(perf.get("end_equity") or 0.0),
        "max_drawdown": float(perf.get("max_drawdown") or 0.0),
        "calmar": float(perf.get("calmar") or 0.0),
        "stress_1_25_avg_monthly_pnl": float(s125.get("avg_monthly_pnl") or 0.0),
        "stress_1_50_avg_monthly_pnl": float(s150.get("avg_monthly_pnl") or 0.0),
        "bootstrap_p10_avg_monthly_pnl": float(bavg.get("p10") or 0.0),
        "bootstrap_p90_max_drawdown": float((boot.get("max_drawdown") or {}).get("p90") or 0.0),
    }


def _window_stats(monthly: pd.DataFrame, window_months: int) -> Dict[str, float]:
    if len(monthly) < window_months:
        return {}
    out: Dict[str, float] = {}
    pnl_roll = monthly["pnl"].rolling(window_months)
    out["rolling_avg_pnl_min"] = float((pnl_roll.mean()).min())
    out["rolling_avg_pnl_p10"] = float((pnl_roll.mean()).quantile(0.10))

    dd_values = []
    for end_idx in range(window_months - 1, len(monthly)):
        chunk = monthly.iloc[end_idx - window_months + 1 : end_idx + 1]
        eq = chunk["equity"].to_numpy(dtype=float)
        if eq.size == 0:
            continue
        peak = np.maximum.accumulate(eq)
        dd = np.min(eq / np.maximum(peak, 1e-12) - 1.0)
        dd_values.append(abs(float(dd)))
    if dd_values:
        arr = np.array(dd_values, dtype=float)
        out["rolling_max_dd_max"] = float(np.max(arr))
        out["rolling_max_dd_p90"] = float(np.quantile(arr, 0.90))
    return out


def _sign_flip_pvalue(monthly: pd.DataFrame, n_iter: int = 6000, seed: int = 42) -> Dict[str, float]:
    pnls = monthly["pnl"].to_numpy(dtype=float)
    if pnls.size == 0:
        return {"observed_mean": 0.0, "null_mean": 0.0, "p_value_ge_observed": 1.0}
    rng = np.random.default_rng(seed)
    observed = float(np.mean(pnls))
    mags = np.abs(pnls)
    sims = []
    for _ in range(n_iter):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=mags.size)
        sims.append(float(np.mean(mags * signs)))
    sims_arr = np.array(sims, dtype=float)
    pval = float(np.mean(sims_arr >= observed))
    return {
        "observed_mean": observed,
        "null_mean": float(np.mean(sims_arr)),
        "p_value_ge_observed": pval,
    }


def timesplit_stats(monthly: pd.DataFrame) -> Dict[str, Any]:
    x = normalize_monthly(monthly)
    if x.empty:
        return {}

    out: Dict[str, Any] = {
        "n_months": int(len(x)),
        "avg_monthly_pnl": float(x["pnl"].mean()),
        "median_monthly_pnl": float(x["pnl"].median()),
        "positive_rate": float((x["pnl"] > 0.0).mean()),
        "top5_positive_pnl_share": float(
            np.sort(np.maximum(x["pnl"].to_numpy(dtype=float), 0.0))[-5:].sum()
            / max(np.maximum(x["pnl"].to_numpy(dtype=float), 0.0).sum(), 1e-12)
        ),
        "top10_positive_pnl_share": float(
            np.sort(np.maximum(x["pnl"].to_numpy(dtype=float), 0.0))[-10:].sum()
            / max(np.maximum(x["pnl"].to_numpy(dtype=float), 0.0).sum(), 1e-12)
        ),
        "zero_trade_month_rate": float((x["trades"] <= 0.0).mean()),
    }

    mid = len(x) // 2
    a = x.iloc[:mid]
    b = x.iloc[mid:]
    if not a.empty and not b.empty:
        out["first_half_avg_pnl"] = float(a["pnl"].mean())
        out["second_half_avg_pnl"] = float(b["pnl"].mean())
        out["first_half_positive_rate"] = float((a["pnl"] > 0.0).mean())
        out["second_half_positive_rate"] = float((b["pnl"] > 0.0).mean())
        out["second_to_first_avg_ratio"] = float(
            out["second_half_avg_pnl"] / max(abs(out["first_half_avg_pnl"]), 1e-9)
        )

    tail = x.tail(24)
    if not tail.empty:
        out["last_24m_avg_pnl"] = float(tail["pnl"].mean())
        out["last_24m_positive_rate"] = float((tail["pnl"] > 0.0).mean())

    out["rolling_24m"] = _window_stats(x, 24)
    out["rolling_36m"] = _window_stats(x, 36)
    out["sign_flip_test"] = _sign_flip_pvalue(x, n_iter=6000, seed=42)
    return out


def assess_risk(summary: Dict[str, float], ts: Dict[str, Any]) -> Dict[str, Any]:
    checks = []

    top5 = float(ts.get("top5_positive_pnl_share") or 0.0)
    checks.append(
        {
            "name": "concentration_top5",
            "value": top5,
            "threshold": 0.45,
            "pass": bool(top5 <= 0.45),
            "note": "Top-5 months should not dominate total positive PnL.",
        }
    )

    ratio = float(ts.get("second_to_first_avg_ratio") or 0.0)
    checks.append(
        {
            "name": "time_split_stability",
            "value": ratio,
            "threshold": 0.5,
            "pass": bool(ratio >= 0.5),
            "note": "Second-half average monthly PnL should retain at least 50% of first half.",
        }
    )

    tail_ratio = float(ts.get("last_24m_avg_pnl") or 0.0) / max(float(ts.get("avg_monthly_pnl") or 0.0), 1e-9)
    checks.append(
        {
            "name": "recent_regime_retention",
            "value": tail_ratio,
            "threshold": 0.4,
            "pass": bool(tail_ratio >= 0.4),
            "note": "Last-24m average PnL should retain at least 40% of full-period average.",
        }
    )

    pval = float(((ts.get("sign_flip_test") or {}).get("p_value_ge_observed")) or 1.0)
    checks.append(
        {
            "name": "significance_sign_flip",
            "value": pval,
            "threshold": 0.10,
            "pass": bool(pval <= 0.10),
            "note": "Lower p-value indicates less chance that mean PnL is random sign noise.",
        }
    )

    p90_dd = float(summary.get("bootstrap_p90_max_drawdown") or 0.0)
    checks.append(
        {
            "name": "bootstrap_drawdown_tail",
            "value": p90_dd,
            "threshold": 0.45,
            "pass": bool(p90_dd <= 0.45),
            "note": "Bootstrap p90 drawdown should remain below 45%.",
        }
    )

    failed = [c for c in checks if not bool(c["pass"])]
    score = max(0.0, 100.0 - 20.0 * float(len(failed)))
    return {
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
        "risk_score_0_to_100": score,
        "risk_label": "low" if score >= 80 else ("medium" if score >= 60 else "high"),
    }


def resolve_paths(step3_out_dir: Path) -> Tuple[Path, Path]:
    hybrid_report_path = step3_out_dir / "optimization" / "btc_eth_hybrid_overlay_report.json"
    if not hybrid_report_path.exists():
        raise FileNotFoundError(f"Missing hybrid report: {hybrid_report_path}")
    hybrid_report = read_json(hybrid_report_path)
    baseline_summary = Path(str((hybrid_report.get("baseline") or {}).get("summary_path") or "")).resolve()
    promoted_summary = step3_out_dir / "backtest" / "step3_summary.json"
    if not baseline_summary.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_summary}")
    if not promoted_summary.exists():
        raise FileNotFoundError(f"Missing promoted summary: {promoted_summary}")
    return baseline_summary, promoted_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step3-out-dir", default="other-pair/btc_eth/step3_out")
    ap.add_argument("--active-summary-path", default=None)
    ap.add_argument("--promoted-summary-path", default=None)
    ap.add_argument(
        "--out-path",
        default="other-pair/btc_eth/step3_out/optimization/overfit_diagnostics_report.json",
    )
    args = ap.parse_args()

    step3_out_dir = Path(args.step3_out_dir).resolve()
    if args.active_summary_path and args.promoted_summary_path:
        active_summary_path = Path(args.active_summary_path).resolve()
        promoted_summary_path = Path(args.promoted_summary_path).resolve()
    else:
        active_summary_path, promoted_summary_path = resolve_paths(step3_out_dir)

    active_summary = read_json(active_summary_path)
    promoted_summary = read_json(promoted_summary_path)
    active_monthly = normalize_monthly(pd.read_parquet(active_summary_path.parent / "step3_monthly_table.parquet"))
    promoted_monthly = normalize_monthly(
        pd.read_parquet(promoted_summary_path.parent / "step3_monthly_table.parquet")
    )

    active_snap = metric_snapshot(active_summary)
    promoted_snap = metric_snapshot(promoted_summary)
    active_ts = timesplit_stats(active_monthly)
    promoted_ts = timesplit_stats(promoted_monthly)
    active_risk = assess_risk(active_snap, active_ts)
    promoted_risk = assess_risk(promoted_snap, promoted_ts)

    delta = {
        k: float(promoted_snap.get(k, 0.0) - active_snap.get(k, 0.0))
        for k in (
            "avg_monthly_pnl",
            "end_equity",
            "max_drawdown",
            "calmar",
            "stress_1_25_avg_monthly_pnl",
            "stress_1_50_avg_monthly_pnl",
            "bootstrap_p10_avg_monthly_pnl",
            "bootstrap_p90_max_drawdown",
        )
    }

    report = {
        "meta": {
            "script": "build_overfit_diagnostics.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "active_summary_path": str(active_summary_path),
            "promoted_summary_path": str(promoted_summary_path),
        },
        "active": {"summary": active_snap, "timesplit": active_ts, "overfit_risk": active_risk},
        "promoted": {"summary": promoted_snap, "timesplit": promoted_ts, "overfit_risk": promoted_risk},
        "delta_promoted_minus_active": delta,
    }

    write_json(Path(args.out_path).resolve(), report)
    print(f"[OVERFIT-DIAG] wrote {Path(args.out_path).resolve()}", flush=True)


if __name__ == "__main__":
    main()

