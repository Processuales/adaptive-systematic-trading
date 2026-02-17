#!/usr/bin/env python3
"""
Run a focused experiment suite for the Step 3 pattern-aid ML idea.

Outputs:
- final output/dual ml/01_pattern_ml_dual_comparison.png
- final output/dual ml/01_pattern_ml_dual_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cmd(
    py: str,
    repo_root: Path,
    dataset_dir: Path,
    out_dir: Path,
    start_capital: float,
    cfg: Dict,
) -> List[str]:
    key_map = {
        "train_lookback_days": "--train-lookback-days",
        "embargo_days": "--embargo-days",
        "min_train_events": "--min-train-events",
        "min_val_events": "--min-val-events",
        "min_test_events": "--min-test-events",
        "mix_struct_weight": "--mix-struct-weight",
        "policy_profile": "--policy-profile",
        "max_aggressive_size": "--max-aggressive-size",
        "retune_every_folds": "--retune-every-folds",
        "portfolio_allocator": "--portfolio-allocator",
        "portfolio_objective": "--portfolio-objective",
        "portfolio_train_ratio": "--portfolio-train-ratio",
        "portfolio_weight_step": "--portfolio-weight-step",
        "portfolio_lookback_days": "--portfolio-lookback-days",
        "portfolio_min_train_days": "--portfolio-min-train-days",
        "portfolio_turnover_penalty": "--portfolio-turnover-penalty",
        "portfolio_weight_smoothing": "--portfolio-weight-smoothing",
        "portfolio_momentum_days": "--portfolio-momentum-days",
        "portfolio_vol_days": "--portfolio-vol-days",
        "portfolio_vol_penalty": "--portfolio-vol-penalty",
        "portfolio_max_tilt": "--portfolio-max-tilt",
        "portfolio_min_weight": "--portfolio-min-weight",
        "portfolio_max_weight": "--portfolio-max-weight",
        "portfolio_spy_guard_lookback_days": "--portfolio-spy-guard-lookback-days",
        "portfolio_spy_guard_drift_lookback_days": "--portfolio-spy-guard-drift-lookback-days",
        "portfolio_spy_guard_min_mult": "--portfolio-spy-guard-min-mult",
        "portfolio_spy_guard_dd_penalty": "--portfolio-spy-guard-dd-penalty",
        "spy_drift_kill_switch": "--spy-drift-kill-switch",
        "spy_drift_feature_z_cap": "--spy-drift-feature-z-cap",
        "drought_relief_symbol": "--drought-relief-symbol",
        "drought_target_trades_per_month": "--drought-target-trades-per-month",
        "drought_p_cut_relax": "--drought-p-cut-relax",
        "drought_ev_relax": "--drought-ev-relax",
        "drought_size_boost": "--drought-size-boost",
        "pattern_n_clusters": "--pattern-n-clusters",
        "pattern_min_cluster_samples": "--pattern-min-cluster-samples",
        "pattern_prior_strength": "--pattern-prior-strength",
        "pattern_consistency_tol": "--pattern-consistency-tol",
        "pattern_prob_strength": "--pattern-prob-strength",
        "pattern_ret_strength": "--pattern-ret-strength",
        "pattern_prob_max_abs_delta": "--pattern-prob-max-abs-delta",
        "pattern_ret_max_abs_delta": "--pattern-ret-max-abs-delta",
        "cost_stress_multipliers": "--cost-stress-multipliers",
        "bootstrap_samples": "--bootstrap-samples",
        "bootstrap_block_months": "--bootstrap-block-months",
        "bootstrap_seed": "--bootstrap-seed",
    }
    bool_flags = {
        "portfolio_no_spy_guard": "--portfolio-no-spy-guard",
        "drought_relief_enable": "--drought-relief-enable",
        "pattern_aid_enable": "--pattern-aid-enable",
    }

    cmd = [
        py,
        str(repo_root / "step3_train_and_backtest.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(out_dir),
        "--start-capital",
        str(start_capital),
    ]
    for k, flag in key_map.items():
        if k in cfg and cfg[k] is not None:
            cmd.extend([flag, str(cfg[k])])
    for k, flag in bool_flags.items():
        if bool(cfg.get(k, False)):
            cmd.append(flag)
    return cmd


def run_cfg(
    py: str,
    repo_root: Path,
    dataset_dir: Path,
    runs_root: Path,
    start_capital: float,
    name: str,
    cfg: Dict,
    reuse_existing: bool,
) -> Dict:
    out_dir = runs_root / name
    summary_path = out_dir / "backtest" / "step3_summary.json"
    monthly_path = out_dir / "backtest" / "step3_monthly_table.parquet"
    if reuse_existing and summary_path.exists() and monthly_path.exists():
        summary = load_json(summary_path)
        monthly = pd.read_parquet(monthly_path)
        return {"name": name, "config": cfg, "summary": summary, "summary_path": str(summary_path), "monthly": monthly}

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    ensure_dir(out_dir)
    cmd = build_cmd(py, repo_root, dataset_dir, out_dir, start_capital, cfg)
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    summary = load_json(summary_path)
    monthly = pd.read_parquet(monthly_path)
    return {"name": name, "config": cfg, "summary": summary, "summary_path": str(summary_path), "monthly": monthly}


def metrics_from_summary(summary: Dict) -> Dict:
    p = summary.get("portfolio", {})
    perf = p.get("dual_perf", {})
    stress_rows = p.get("cost_stress_tests") or []
    s125 = next((r for r in stress_rows if abs(float(r.get("cost_multiplier") or 0.0) - 1.25) <= 0.03), {})
    s150 = next((r for r in stress_rows if abs(float(r.get("cost_multiplier") or 0.0) - 1.50) <= 0.03), {})
    boot = p.get("bootstrap") or {}
    b_avg = boot.get("avg_monthly_pnl") or {}
    return {
        "avg_monthly_pnl": float(p.get("avg_monthly_pnl") or 0.0),
        "avg_monthly_trades": float(p.get("avg_monthly_trades") or 0.0),
        "end_equity": float(perf.get("end_equity") or 0.0),
        "max_drawdown": float(perf.get("max_drawdown") or 1.0),
        "calmar": float(perf.get("calmar") or 0.0),
        "stress_125_avg_monthly_pnl": float(s125.get("avg_monthly_pnl") or 0.0),
        "stress_150_avg_monthly_pnl": float(s150.get("avg_monthly_pnl") or 0.0),
        "bootstrap_p10_avg_monthly_pnl": float(b_avg.get("p10") or 0.0),
        "pattern_enabled": bool(summary.get("meta", {}).get("pattern_aid_enable", False)),
    }


def select_winner(rows: List[Dict]) -> Dict:
    for r in rows:
        r["metrics"] = metrics_from_summary(r["summary"])
    baseline = next(r for r in rows if r["name"] == "baseline")
    b = baseline["metrics"]

    for r in rows:
        m = r["metrics"]
        r["score"] = (
            0.09 * m["avg_monthly_pnl"]
            + 0.45 * m["calmar"]
            - 3.8 * m["max_drawdown"]
            + 0.03 * m["stress_125_avg_monthly_pnl"]
            + 0.01 * m["stress_150_avg_monthly_pnl"]
            + 0.01 * m["bootstrap_p10_avg_monthly_pnl"]
        )

    best = max(rows, key=lambda x: float(x["score"]))
    bm = best["metrics"]
    improved_return = (
        bm["avg_monthly_pnl"] >= b["avg_monthly_pnl"] * 1.01
        and bm["stress_125_avg_monthly_pnl"] >= b["stress_125_avg_monthly_pnl"] * 0.99
    )
    improved_risk = (
        bm["avg_monthly_pnl"] >= b["avg_monthly_pnl"] * 0.995
        and bm["max_drawdown"] <= b["max_drawdown"] * 0.95
        and bm["calmar"] >= b["calmar"] * 1.02
    )
    keep_pattern = bool(best["metrics"]["pattern_enabled"]) and (improved_return or improved_risk)
    verdict = "keep_pattern_ml" if keep_pattern else "remove_pattern_ml"
    rationale = (
        "Pattern ML improved return or preserved return with better risk."
        if keep_pattern
        else "Pattern ML did not beat baseline after robustness checks."
    )
    return {
        "baseline_name": baseline["name"],
        "best_name": best["name"],
        "best_score": float(best["score"]),
        "keep_pattern_ml": keep_pattern,
        "verdict": verdict,
        "rationale": rationale,
    }


def plot_comparison(rows: List[Dict], decision: Dict, out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2.8, 1.6]})

    for r in rows:
        m = r["monthly"].copy()
        m["month_end"] = pd.to_datetime(m["month_end"], utc=True, errors="coerce")
        m = m.dropna(subset=["month_end"]).sort_values("month_end")
        lbl = r["name"] + (" (pattern)" if r["metrics"]["pattern_enabled"] else " (baseline)")
        ax1.plot(m["month_end"], m["equity"], linewidth=1.8, label=lbl)
    ax1.set_title("Dual ML Experiment: Baseline vs Pattern-Aid Variants")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper left")

    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    pnl = [r["metrics"]["avg_monthly_pnl"] for r in rows]
    mdd = [100.0 * r["metrics"]["max_drawdown"] for r in rows]
    cal = [r["metrics"]["calmar"] for r in rows]
    bars = ax2.bar(x, pnl, color=["#1d3557" if n == "baseline" else "#2a9d8f" for n in names], alpha=0.85)
    ax2.set_ylabel("Avg monthly PnL ($)")
    ax2.set_xticks(x, names, rotation=0)
    ax2.grid(axis="y", alpha=0.2)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax2b = ax2.twinx()
    ax2b.plot(x, mdd, color="#d62828", marker="o", linewidth=1.6, label="Max drawdown %")
    ax2b.plot(x, cal, color="#6a994e", marker="s", linewidth=1.6, label="Calmar")
    ax2b.set_ylabel("Drawdown % / Calmar")
    ax2b.legend(loc="upper right")

    verdict = decision["verdict"]
    rationale = decision["rationale"]
    ax1.text(
        0.01,
        0.02,
        f"Decision: {verdict}\nBest run: {decision['best_name']}\n{rationale}",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.86, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="step3_out/dataset")
    ap.add_argument("--step3-out-dir", default="step3_out")
    ap.add_argument("--best-config-path", default="step3_out/optimization/step3_best_config.json")
    ap.add_argument("--final-dir", default="final output/dual ml")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--reuse-existing-runs", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    step3_out = (repo_root / args.step3_out_dir).resolve()
    best_cfg_path = (repo_root / args.best_config_path).resolve()
    final_dir = (repo_root / args.final_dir).resolve()
    runs_root = step3_out / "pattern_aid_runs"
    ensure_dir(runs_root)
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    ensure_dir(final_dir)

    base_cfg = load_json(best_cfg_path)
    base_cfg.pop("name", None)

    experiments = [
        ("baseline", {**base_cfg, "pattern_aid_enable": False}),
        (
            "pattern_conservative",
            {
                **base_cfg,
                "pattern_aid_enable": True,
                "pattern_n_clusters": 5,
                "pattern_min_cluster_samples": 48,
                "pattern_prior_strength": 160.0,
                "pattern_consistency_tol": 0.0035,
                "pattern_prob_strength": 0.35,
                "pattern_ret_strength": 0.40,
                "pattern_prob_max_abs_delta": 0.025,
                "pattern_ret_max_abs_delta": 0.0020,
            },
        ),
        (
            "pattern_balanced",
            {
                **base_cfg,
                "pattern_aid_enable": True,
                "pattern_n_clusters": 6,
                "pattern_min_cluster_samples": 40,
                "pattern_prior_strength": 120.0,
                "pattern_consistency_tol": 0.0040,
                "pattern_prob_strength": 0.55,
                "pattern_ret_strength": 0.65,
                "pattern_prob_max_abs_delta": 0.040,
                "pattern_ret_max_abs_delta": 0.0035,
            },
        ),
        (
            "pattern_strict",
            {
                **base_cfg,
                "pattern_aid_enable": True,
                "pattern_n_clusters": 4,
                "pattern_min_cluster_samples": 56,
                "pattern_prior_strength": 180.0,
                "pattern_consistency_tol": 0.0030,
                "pattern_prob_strength": 0.45,
                "pattern_ret_strength": 0.50,
                "pattern_prob_max_abs_delta": 0.030,
                "pattern_ret_max_abs_delta": 0.0025,
            },
        ),
    ]

    rows: List[Dict] = []
    for name, cfg in experiments:
        print(f"[PATTERN-EXP] running={name}")
        row = run_cfg(
            py=py,
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            runs_root=runs_root,
            start_capital=args.start_capital,
            name=name,
            cfg=cfg,
            reuse_existing=args.reuse_existing_runs,
        )
        rows.append(row)
        m = metrics_from_summary(row["summary"])
        print(
            f"[PATTERN-EXP] {name}: avg_pnl={m['avg_monthly_pnl']:.2f} "
            f"mdd={m['max_drawdown']:.4f} calmar={m['calmar']:.4f}"
        )

    decision = select_winner(rows)

    png_path = final_dir / "01_pattern_ml_dual_comparison.png"
    json_path = final_dir / "01_pattern_ml_dual_comparison.json"
    plot_comparison(rows=rows, decision=decision, out_png=png_path)

    out = {
        "meta": {
            "script": "step3_pattern_aid_experiment.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Pattern-aid ML experiment over baseline Step 3 configuration.",
        },
        "decision": decision,
        "experiments": [
            {
                "name": r["name"],
                "metrics": r["metrics"],
                "summary_path": r["summary_path"],
                "pattern_enabled": bool(r["metrics"]["pattern_enabled"]),
            }
            for r in rows
        ],
        "paths": {
            "comparison_png": str(png_path),
            "comparison_json": str(json_path),
            "runs_root": str(runs_root),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(out, 6), f, separators=(",", ":"), ensure_ascii=True)

    print(f"[PATTERN-EXP] Wrote: {png_path}")
    print(f"[PATTERN-EXP] Wrote: {json_path}")
    print(f"[PATTERN-EXP] Decision: {decision['verdict']} (best={decision['best_name']})")


if __name__ == "__main__":
    main()
