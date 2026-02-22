#!/usr/bin/env python3
"""
Step 3 optimizer.

Runs multiple Step 3 training/backtest configurations and selects the best
risk-adjusted candidate, then re-runs best config into the main step3_out.
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

import numpy as np

SCRIPT_VERSION = "2.5.0"


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


def candidate_grid() -> List[Dict]:
    return [
        {
            "name": "ultimate_high_gain_baseline",
            "train_lookback_days": 1460,
            "embargo_days": 10,
            "min_train_events": 220,
            "min_val_events": 45,
            "min_test_events": 6,
            "mix_struct_weight": 0.55,
            "policy_profile": "growth",
            "max_aggressive_size": 1.35,
            "portfolio_allocator": "dynamic_regime",
            "retune_every_folds": 2,
            "portfolio_objective": "end_equity",
            "portfolio_train_ratio": 0.55,
            "portfolio_lookback_days": 504,
            "portfolio_weight_smoothing": 0.15,
            "portfolio_min_weight": 0.25,
            "portfolio_max_weight": 0.98,
        },
        {
            "name": "ultimate_high_gain_guarded_soft",
            "train_lookback_days": 1460,
            "embargo_days": 10,
            "min_train_events": 220,
            "min_val_events": 45,
            "min_test_events": 6,
            "mix_struct_weight": 0.55,
            "policy_profile": "growth",
            "max_aggressive_size": 1.35,
            "portfolio_allocator": "dynamic_regime",
            "retune_every_folds": 2,
            "portfolio_objective": "end_equity",
            "portfolio_train_ratio": 0.55,
            "portfolio_lookback_days": 504,
            "portfolio_weight_smoothing": 0.15,
            "portfolio_min_weight": 0.25,
            "portfolio_max_weight": 0.98,
            "spy_drift_kill_switch": "soft",
            "spy_drift_feature_z_cap": 1.25,
            "drought_relief_enable": True,
            "drought_relief_symbol": "QQQ",
            "drought_target_trades_per_month": 5.5,
            "drought_p_cut_relax": 0.015,
            "drought_ev_relax": 0.00018,
            "drought_size_boost": 0.08,
            "portfolio_spy_guard_lookback_days": 63,
            "portfolio_spy_guard_drift_lookback_days": 42,
            "portfolio_spy_guard_min_mult": 0.25,
            "portfolio_spy_guard_dd_penalty": 1.2,
        },
        {
            "name": "ultimate_high_gain_pattern_balanced",
            "train_lookback_days": 1460,
            "embargo_days": 10,
            "min_train_events": 220,
            "min_val_events": 45,
            "min_test_events": 6,
            "mix_struct_weight": 0.55,
            "policy_profile": "growth",
            "max_aggressive_size": 1.35,
            "portfolio_allocator": "dynamic_regime",
            "retune_every_folds": 2,
            "portfolio_objective": "end_equity",
            "portfolio_train_ratio": 0.55,
            "portfolio_lookback_days": 504,
            "portfolio_weight_smoothing": 0.15,
            "portfolio_min_weight": 0.25,
            "portfolio_max_weight": 0.98,
            "spy_drift_kill_switch": "soft",
            "spy_drift_feature_z_cap": 1.25,
            "drought_relief_enable": True,
            "drought_relief_symbol": "QQQ",
            "drought_target_trades_per_month": 5.5,
            "drought_p_cut_relax": 0.015,
            "drought_ev_relax": 0.00018,
            "drought_size_boost": 0.08,
            "portfolio_spy_guard_lookback_days": 63,
            "portfolio_spy_guard_drift_lookback_days": 42,
            "portfolio_spy_guard_min_mult": 0.25,
            "portfolio_spy_guard_dd_penalty": 1.2,
            "pattern_aid_enable": True,
            "pattern_n_clusters": 6,
            "pattern_min_cluster_samples": 40,
            "pattern_prior_strength": 120.0,
            "pattern_consistency_tol": 0.0040,
            "pattern_prob_strength": 0.55,
            "pattern_ret_strength": 0.65,
            "pattern_prob_max_abs_delta": 0.04,
            "pattern_ret_max_abs_delta": 0.0035,
        },
        {
            "name": "ultimate_high_gain_guarded_fast_retune",
            "train_lookback_days": 1460,
            "embargo_days": 10,
            "min_train_events": 220,
            "min_val_events": 45,
            "min_test_events": 6,
            "mix_struct_weight": 0.52,
            "policy_profile": "growth",
            "max_aggressive_size": 1.40,
            "portfolio_allocator": "dynamic_regime",
            "retune_every_folds": 1,
            "portfolio_objective": "end_equity",
            "portfolio_train_ratio": 0.55,
            "portfolio_lookback_days": 504,
            "portfolio_weight_smoothing": 0.15,
            "portfolio_min_weight": 0.35,
            "portfolio_max_weight": 0.98,
            "spy_drift_kill_switch": "soft",
            "spy_drift_feature_z_cap": 1.25,
            "drought_relief_enable": True,
            "drought_relief_symbol": "QQQ",
            "drought_target_trades_per_month": 5.8,
            "drought_p_cut_relax": 0.018,
            "drought_ev_relax": 0.00022,
            "drought_size_boost": 0.10,
            "portfolio_spy_guard_lookback_days": 63,
            "portfolio_spy_guard_drift_lookback_days": 42,
            "portfolio_spy_guard_min_mult": 0.20,
            "portfolio_spy_guard_dd_penalty": 1.2,
        },
        {
            "name": "growth_equal_split_baseline",
            "train_lookback_days": 730,
            "embargo_days": 7,
            "min_train_events": 170,
            "min_val_events": 30,
            "min_test_events": 5,
            "mix_struct_weight": 0.60,
            "policy_profile": "growth",
            "max_aggressive_size": 1.20,
            "portfolio_allocator": "equal_split",
            "retune_every_folds": 3,
            "portfolio_objective": "calmar",
            "portfolio_no_spy_guard": True,
            "spy_drift_kill_switch": "none",
            "drought_relief_enable": False,
        },
        {
            "name": "balanced_equal_split_baseline",
            "train_lookback_days": 1095,
            "embargo_days": 7,
            "min_train_events": 180,
            "min_val_events": 35,
            "min_test_events": 5,
            "mix_struct_weight": 0.65,
            "policy_profile": "balanced",
            "max_aggressive_size": 1.10,
            "portfolio_allocator": "equal_split",
            "retune_every_folds": 4,
            "portfolio_objective": "calmar",
            "portfolio_no_spy_guard": True,
            "spy_drift_kill_switch": "none",
            "drought_relief_enable": False,
        },
    ]


def _stress_row(portfolio: Dict, target_mult: float, tol: float = 0.03) -> Dict:
    rows = portfolio.get("cost_stress_tests") or []
    for r in rows:
        m = float(r.get("cost_multiplier") or 0.0)
        if abs(m - target_mult) <= tol:
            return r
    return {}


def _bootstrap_quantile(portfolio: Dict, metric: str, key: str, default: float) -> float:
    boot = portfolio.get("bootstrap") or {}
    if not boot or not bool(boot.get("enabled")):
        return float(default)
    bucket = boot.get(metric) or {}
    return float(bucket.get(key) or default)


def objective_from_summary(
    summary: Dict,
    dd_cap: float,
    min_trades_per_month: float,
    target_avg_monthly_pnl: float,
) -> float:
    p = summary["portfolio"]
    perf = p["dual_perf"]
    calmar = float(perf["calmar"]) if perf.get("calmar") is not None else -1.0
    dd = float(perf["max_drawdown"]) if perf.get("max_drawdown") is not None else 1.0
    avg_pnl = float(p.get("avg_monthly_pnl") or 0.0)
    med_pnl = float(p.get("median_monthly_pnl") or 0.0)
    avg_tpm = float(p.get("avg_monthly_trades") or 0.0)
    pos_rate = float(p.get("monthly_positive_rate") or 0.0)
    neg_rate = float(p.get("monthly_negative_rate") or 1.0)
    top5_pos_share = float(p.get("top5_positive_pnl_share") or 1.0)
    zero_trade_rate = float(p.get("zero_trade_month_rate") or 1.0)
    aggr = float(p.get("aggressive_trade_rate") or 0.0)
    cagr = float(perf.get("cagr") or -1.0)
    q_stability = summary.get("symbol_summaries", {}).get("qqq", {}).get("fold_stability", {})
    s_stability = summary.get("symbol_summaries", {}).get("spy", {}).get("fold_stability", {})
    q_p25 = float(q_stability.get("p25_test_calmar") or -1.0)
    s_p25 = float(s_stability.get("p25_test_calmar") or -1.0)
    q_neg = float(q_stability.get("negative_test_calmar_rate") or 1.0)
    s_neg = float(s_stability.get("negative_test_calmar_rate") or 1.0)
    allocator = p.get("allocator", {})
    q_turnover = float(allocator.get("qqq_weight_turnover_abs") or 0.0)
    s125 = _stress_row(p, 1.25)
    s150 = _stress_row(p, 1.50)
    s125_perf = s125.get("dual_perf") if s125 else {}
    s150_perf = s150.get("dual_perf") if s150 else {}
    s125_avg = float(s125.get("avg_monthly_pnl") or 0.0)
    s150_avg = float(s150.get("avg_monthly_pnl") or 0.0)
    s125_calmar = float(s125_perf.get("calmar") or 0.0) if s125_perf else 0.0
    s150_calmar = float(s150_perf.get("calmar") or 0.0) if s150_perf else 0.0
    boot_p10_avg = _bootstrap_quantile(p, "avg_monthly_pnl", "p10", default=0.0)
    boot_p10_calmar = _bootstrap_quantile(p, "calmar", "p10", default=-1.0)

    score = (
        1.6 * calmar
        + 0.080 * avg_pnl
        + 0.010 * med_pnl
        + 0.10 * min(avg_tpm, 12.0)
        + 0.45 * cagr
        + 0.75 * pos_rate
        + 0.12 * aggr
        + 0.18 * q_p25
        + 0.08 * s_p25
        + 0.035 * s125_avg
        + 0.015 * s150_avg
        + 0.50 * s125_calmar
        + 0.20 * s150_calmar
        + 0.012 * boot_p10_avg
        + 0.20 * boot_p10_calmar
    )
    score -= 4.0 * max(0.0, dd - dd_cap)
    score -= 0.80 * max(0.0, min_trades_per_month - avg_tpm)
    score -= 0.45 * (q_neg + s_neg)
    score -= 0.80 * max(0.0, top5_pos_share - 0.50)
    score -= 0.45 * max(0.0, zero_trade_rate - 0.22)
    score -= 0.20 * max(0.0, neg_rate - 0.40)
    score -= 0.005 * max(0.0, q_turnover - 80.0)
    score -= 0.015 * max(0.0, target_avg_monthly_pnl - avg_pnl)
    score -= 0.040 * max(0.0, 50.0 - s125_avg)
    score -= 0.020 * max(0.0, 20.0 - s150_avg)
    score -= 0.30 * max(0.0, 0.25 - s125_calmar)
    score -= 0.15 * max(0.0, 0.05 - s150_calmar)
    score -= 0.030 * max(0.0, 20.0 - boot_p10_avg)
    return float(score)


def robustness_check(
    summary: Dict,
    hard_dd_cap: float,
    max_negative_fold_rate: float,
    min_p25_calmar: float,
    min_avg_monthly_pnl: float,
    min_positive_month_rate: float,
    max_top5_positive_pnl_share: float,
    max_zero_trade_month_rate: float,
    min_stress125_avg_monthly_pnl: float,
    min_stress150_avg_monthly_pnl: float,
    min_bootstrap_p10_monthly_pnl: float,
) -> Dict:
    p = summary["portfolio"]
    perf = p["dual_perf"]
    dd = float(perf.get("max_drawdown") or 1.0)
    avg_pnl = float(p.get("avg_monthly_pnl") or 0.0)
    pos_rate = float(p.get("monthly_positive_rate") or 0.0)
    top5_pos_share = float(p.get("top5_positive_pnl_share") or 1.0)
    zero_trade_rate = float(p.get("zero_trade_month_rate") or 1.0)
    q_stability = summary.get("symbol_summaries", {}).get("qqq", {}).get("fold_stability", {})
    s_stability = summary.get("symbol_summaries", {}).get("spy", {}).get("fold_stability", {})
    q_p25 = float(q_stability.get("p25_test_calmar") or -1e9)
    s_p25 = float(s_stability.get("p25_test_calmar") or -1e9)
    q_neg = float(q_stability.get("negative_test_calmar_rate") or 1.0)
    s_neg = float(s_stability.get("negative_test_calmar_rate") or 1.0)
    s125 = _stress_row(p, 1.25)
    s150 = _stress_row(p, 1.50)
    s125_avg = float(s125.get("avg_monthly_pnl") or -1e9) if s125 else -1e9
    s150_avg = float(s150.get("avg_monthly_pnl") or -1e9) if s150 else -1e9
    s125_dd = float((s125.get("dual_perf") or {}).get("max_drawdown") or 1.0) if s125 else 1.0
    boot_p10_avg = _bootstrap_quantile(p, "avg_monthly_pnl", "p10", default=-1e9)

    failures: List[str] = []
    if dd > hard_dd_cap:
        failures.append(f"max_drawdown {dd:.4f} > hard_dd_cap {hard_dd_cap:.4f}")
    if q_neg > max_negative_fold_rate:
        failures.append(f"qqq negative_fold_rate {q_neg:.4f} > {max_negative_fold_rate:.4f}")
    if s_neg > max_negative_fold_rate:
        failures.append(f"spy negative_fold_rate {s_neg:.4f} > {max_negative_fold_rate:.4f}")
    if q_p25 < min_p25_calmar:
        failures.append(f"qqq p25_calmar {q_p25:.4f} < {min_p25_calmar:.4f}")
    if s_p25 < min_p25_calmar:
        failures.append(f"spy p25_calmar {s_p25:.4f} < {min_p25_calmar:.4f}")
    if avg_pnl < min_avg_monthly_pnl:
        failures.append(f"avg_monthly_pnl {avg_pnl:.2f} < {min_avg_monthly_pnl:.2f}")
    if pos_rate < min_positive_month_rate:
        failures.append(f"monthly_positive_rate {pos_rate:.4f} < {min_positive_month_rate:.4f}")
    if top5_pos_share > max_top5_positive_pnl_share:
        failures.append(f"top5_positive_pnl_share {top5_pos_share:.4f} > {max_top5_positive_pnl_share:.4f}")
    if zero_trade_rate > max_zero_trade_month_rate:
        failures.append(f"zero_trade_month_rate {zero_trade_rate:.4f} > {max_zero_trade_month_rate:.4f}")
    if s125_avg < min_stress125_avg_monthly_pnl:
        failures.append(f"stress1.25 avg_monthly_pnl {s125_avg:.2f} < {min_stress125_avg_monthly_pnl:.2f}")
    if s150_avg < min_stress150_avg_monthly_pnl:
        failures.append(f"stress1.50 avg_monthly_pnl {s150_avg:.2f} < {min_stress150_avg_monthly_pnl:.2f}")
    if s125_dd > hard_dd_cap * 1.10:
        failures.append(f"stress1.25 max_drawdown {s125_dd:.4f} > {hard_dd_cap * 1.10:.4f}")
    if boot_p10_avg < min_bootstrap_p10_monthly_pnl:
        failures.append(f"bootstrap avg_monthly_pnl p10 {boot_p10_avg:.2f} < {min_bootstrap_p10_monthly_pnl:.2f}")

    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "snapshot": {
            "max_drawdown": dd,
            "avg_monthly_pnl": avg_pnl,
            "monthly_positive_rate": pos_rate,
            "top5_positive_pnl_share": top5_pos_share,
            "zero_trade_month_rate": zero_trade_rate,
            "qqq_negative_fold_rate": q_neg,
            "spy_negative_fold_rate": s_neg,
            "qqq_p25_test_calmar": q_p25,
            "spy_p25_test_calmar": s_p25,
            "stress_1.25_avg_monthly_pnl": s125_avg,
            "stress_1.25_max_drawdown": s125_dd,
            "stress_1.50_avg_monthly_pnl": s150_avg,
            "bootstrap_p10_avg_monthly_pnl": boot_p10_avg,
        },
    }


def run_cfg(
    py: str,
    repo_root: Path,
    dataset_dir: Path,
    out_dir: Path,
    start_capital: float,
    cfg: Dict,
    crypto_mode: bool = False,
) -> Dict:
    cmd = [
        py,
        str(repo_root / "step3_train_and_backtest.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(out_dir),
        "--start-capital",
        str(start_capital),
        "--train-lookback-days",
        str(cfg["train_lookback_days"]),
        "--embargo-days",
        str(cfg["embargo_days"]),
        "--min-train-events",
        str(cfg["min_train_events"]),
        "--min-val-events",
        str(cfg["min_val_events"]),
        "--min-test-events",
        str(cfg["min_test_events"]),
        "--mix-struct-weight",
        str(cfg["mix_struct_weight"]),
        "--policy-profile",
        str(cfg.get("policy_profile", "growth")),
        "--max-aggressive-size",
        str(cfg.get("max_aggressive_size", 1.20)),
        "--portfolio-allocator",
        str(cfg.get("portfolio_allocator", "dynamic_regime")),
        "--portfolio-objective",
        str(cfg.get("portfolio_objective", "calmar")),
        "--portfolio-train-ratio",
        str(cfg.get("portfolio_train_ratio", 0.60)),
        "--portfolio-weight-step",
        str(cfg.get("portfolio_weight_step", 0.05)),
        "--portfolio-lookback-days",
        str(cfg.get("portfolio_lookback_days", 756)),
        "--portfolio-min-train-days",
        str(cfg.get("portfolio_min_train_days", 252)),
        "--portfolio-turnover-penalty",
        str(cfg.get("portfolio_turnover_penalty", 0.04)),
        "--portfolio-weight-smoothing",
        str(cfg.get("portfolio_weight_smoothing", 0.30)),
        "--portfolio-momentum-days",
        str(cfg.get("portfolio_momentum_days", 63)),
        "--portfolio-vol-days",
        str(cfg.get("portfolio_vol_days", 21)),
        "--portfolio-vol-penalty",
        str(cfg.get("portfolio_vol_penalty", 1.4)),
        "--portfolio-max-tilt",
        str(cfg.get("portfolio_max_tilt", 0.2)),
        "--portfolio-min-weight",
        str(cfg.get("portfolio_min_weight", 0.05)),
        "--portfolio-max-weight",
        str(cfg.get("portfolio_max_weight", 0.95)),
        "--portfolio-spy-guard-lookback-days",
        str(cfg.get("portfolio_spy_guard_lookback_days", 63)),
        "--portfolio-spy-guard-drift-lookback-days",
        str(cfg.get("portfolio_spy_guard_drift_lookback_days", 42)),
        "--portfolio-spy-guard-min-mult",
        str(cfg.get("portfolio_spy_guard_min_mult", 0.25)),
        "--portfolio-spy-guard-dd-penalty",
        str(cfg.get("portfolio_spy_guard_dd_penalty", 1.2)),
        "--spy-drift-kill-switch",
        str(cfg.get("spy_drift_kill_switch", "soft")),
        "--spy-drift-feature-z-cap",
        str(cfg.get("spy_drift_feature_z_cap", 1.25)),
        "--drought-relief-symbol",
        str(cfg.get("drought_relief_symbol", "QQQ")),
        "--drought-target-trades-per-month",
        str(cfg.get("drought_target_trades_per_month", 5.5)),
        "--drought-p-cut-relax",
        str(cfg.get("drought_p_cut_relax", 0.015)),
        "--drought-ev-relax",
        str(cfg.get("drought_ev_relax", 0.00018)),
        "--drought-size-boost",
        str(cfg.get("drought_size_boost", 0.08)),
        "--pattern-n-clusters",
        str(cfg.get("pattern_n_clusters", 6)),
        "--pattern-min-cluster-samples",
        str(cfg.get("pattern_min_cluster_samples", 40)),
        "--pattern-prior-strength",
        str(cfg.get("pattern_prior_strength", 120.0)),
        "--pattern-consistency-tol",
        str(cfg.get("pattern_consistency_tol", 0.004)),
        "--pattern-prob-strength",
        str(cfg.get("pattern_prob_strength", 0.55)),
        "--pattern-ret-strength",
        str(cfg.get("pattern_ret_strength", 0.65)),
        "--pattern-prob-max-abs-delta",
        str(cfg.get("pattern_prob_max_abs_delta", 0.04)),
        "--pattern-ret-max-abs-delta",
        str(cfg.get("pattern_ret_max_abs_delta", 0.0035)),
        "--cost-stress-multipliers",
        str(cfg.get("cost_stress_multipliers", "1.25,1.50")),
        "--bootstrap-samples",
        str(cfg.get("bootstrap_samples", 800)),
        "--bootstrap-block-months",
        str(cfg.get("bootstrap_block_months", 6)),
        "--bootstrap-seed",
        str(cfg.get("bootstrap_seed", 42)),
        "--retune-every-folds",
        str(cfg.get("retune_every_folds", 3)),
    ]
    if bool(cfg.get("portfolio_no_spy_guard", False)):
        cmd.append("--portfolio-no-spy-guard")
    if bool(cfg.get("drought_relief_enable", False)):
        cmd.append("--drought-relief-enable")
    if bool(cfg.get("pattern_aid_enable", False)):
        cmd.append("--pattern-aid-enable")
    if bool(crypto_mode):
        cmd += [
            "--exclude-leaky-features",
            "--display-symbol-1",
            "BTC",
            "--display-symbol-2",
            "ETH",
        ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    s_path = out_dir / "backtest" / "step3_summary.json"
    if not s_path.exists():
        raise FileNotFoundError(f"Expected summary not found: {s_path}")
    with open(s_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_final_artifacts(
    repo_root: Path,
    out_root: Path,
    opt_root: Path,
    final_dir_input: str,
    best_cfg: Dict,
    best_row: Dict,
) -> str:
    final_dir = Path(final_dir_input)
    if not final_dir.is_absolute():
        final_dir = (repo_root / final_dir).resolve()
    if final_dir.exists():
        for p in final_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    ensure_dir(str(final_dir))

    backtest_dir = out_root / "backtest"
    to_copy = [
        (backtest_dir / "step3_summary.json", final_dir / "step3_summary.json"),
        (backtest_dir / "step3_dual_portfolio_curve.png", final_dir / "step3_dual_portfolio_curve.png"),
        (backtest_dir / "step3_monthly_table.parquet", final_dir / "step3_monthly_table.parquet"),
        (opt_root / "step3_optimization_report.json", final_dir / "step3_optimization_report.json"),
        (opt_root / "step3_best_config.json", final_dir / "step3_best_config.json"),
    ]
    for src, dst in to_copy:
        if src.exists():
            shutil.copy2(src, dst)

    snapshot = {
        "meta": {
            "script": "step3_optimize_model.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Consolidated final Step 3 artifacts.",
        },
        "best_config": best_cfg,
        "best_run": best_row,
        "paths": {
            "step3_summary_json": str(final_dir / "step3_summary.json"),
            "step3_plot_png": str(final_dir / "step3_dual_portfolio_curve.png"),
            "step3_monthly_table_parquet": str(final_dir / "step3_monthly_table.parquet"),
            "step3_optimization_report_json": str(final_dir / "step3_optimization_report.json"),
            "step3_best_config_json": str(final_dir / "step3_best_config.json"),
        },
    }
    snapshot_path = final_dir / "step3_final_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(snapshot, 6), f, separators=(",", ":"), ensure_ascii=True)
    return str(final_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to step3_out/dataset")
    ap.add_argument("--out-dir", required=True, help="Path to step3_out root")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--dd-cap", type=float, default=0.12)
    ap.add_argument("--min-trades-per-month", type=float, default=6.0)
    ap.add_argument("--target-avg-monthly-pnl", type=float, default=105.0)
    ap.add_argument("--hard-dd-cap", type=float, default=0.18)
    ap.add_argument("--max-negative-fold-rate", type=float, default=0.75)
    ap.add_argument("--min-p25-calmar", type=float, default=-7.0)
    ap.add_argument("--min-avg-monthly-pnl", type=float, default=80.0)
    ap.add_argument("--min-positive-month-rate", type=float, default=0.50)
    ap.add_argument("--max-top5-positive-pnl-share", type=float, default=0.50)
    ap.add_argument("--max-zero-trade-month-rate", type=float, default=0.22)
    ap.add_argument("--min-stress125-avg-monthly-pnl", type=float, default=50.0)
    ap.add_argument("--min-stress150-avg-monthly-pnl", type=float, default=20.0)
    ap.add_argument("--min-bootstrap-p10-monthly-pnl", type=float, default=20.0)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument(
        "--crypto-mode",
        action="store_true",
        help="Pass crypto safety flags into step3_train_and_backtest (e.g., leaky-feature exclusion).",
    )
    ap.add_argument("--final-dir", default="final output/step3", help="Consolidated final Step 3 artifacts path.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    py = sys.executable

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset-dir: {dataset_dir}")

    opt_root = out_root / "optimization"
    runs_dir = opt_root / "runs"
    ensure_dir(str(runs_dir))
    for p in runs_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

    grid = candidate_grid()[: max(1, args.max_candidates)]
    rows: List[Dict] = []
    for i, cfg in enumerate(grid, start=1):
        run_dir = runs_dir / f"{i:02d}_{cfg['name']}"
        ensure_dir(str(run_dir))
        status = "ok"
        err = None
        summary = None
        score = -1e18
        robust = None
        try:
            summary = run_cfg(
                py=py,
                repo_root=repo_root,
                dataset_dir=dataset_dir,
                out_dir=run_dir,
                start_capital=args.start_capital,
                cfg=cfg,
                crypto_mode=bool(args.crypto_mode),
            )
            score = objective_from_summary(
                summary=summary,
                dd_cap=args.dd_cap,
                min_trades_per_month=args.min_trades_per_month,
                target_avg_monthly_pnl=args.target_avg_monthly_pnl,
            )
            robust = robustness_check(
                summary=summary,
                hard_dd_cap=args.hard_dd_cap,
                max_negative_fold_rate=args.max_negative_fold_rate,
                min_p25_calmar=args.min_p25_calmar,
                min_avg_monthly_pnl=args.min_avg_monthly_pnl,
                min_positive_month_rate=args.min_positive_month_rate,
                max_top5_positive_pnl_share=args.max_top5_positive_pnl_share,
                max_zero_trade_month_rate=args.max_zero_trade_month_rate,
                min_stress125_avg_monthly_pnl=args.min_stress125_avg_monthly_pnl,
                min_stress150_avg_monthly_pnl=args.min_stress150_avg_monthly_pnl,
                min_bootstrap_p10_monthly_pnl=args.min_bootstrap_p10_monthly_pnl,
            )
        except Exception as e:
            status = "error"
            err = str(e)
        row = {
            "run_id": i,
            "name": cfg["name"],
            "config": cfg,
            "status": status,
            "error": err,
            "objective_score": score,
            "robustness": robust,
            "summary_path": str(run_dir / "backtest" / "step3_summary.json"),
            "portfolio_snapshot": (summary.get("portfolio") if summary else None),
        }
        rows.append(row)
        robust_text = "n/a"
        if robust is not None:
            robust_text = "pass" if robust.get("pass") else "fail"
        print(
            f"[STEP3-OPT] run={i}/{len(grid)} name={cfg['name']} status={status} "
            f"score={score:.4f} robust={robust_text}"
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("All optimization runs failed.")
    robust_rows = [r for r in ok_rows if bool((r.get("robustness") or {}).get("pass"))]
    fallback_used = False
    if robust_rows:
        best = max(robust_rows, key=lambda r: float(r["objective_score"]))
    else:
        fallback_used = True
        best = max(ok_rows, key=lambda r: float(r["objective_score"]))

    # Promote best run by rerunning directly into main out-dir.
    best_cfg = best["config"]
    run_cfg(
        py=py,
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        out_dir=out_root,
        start_capital=args.start_capital,
        cfg=best_cfg,
        crypto_mode=bool(args.crypto_mode),
    )

    best_report = {
        "meta": {
            "script": "step3_optimize_model.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_candidates": len(grid),
            "dd_cap": args.dd_cap,
            "min_trades_per_month": args.min_trades_per_month,
            "target_avg_monthly_pnl": args.target_avg_monthly_pnl,
            "hard_dd_cap": args.hard_dd_cap,
            "max_negative_fold_rate": args.max_negative_fold_rate,
            "min_p25_calmar": args.min_p25_calmar,
            "min_avg_monthly_pnl": args.min_avg_monthly_pnl,
            "min_positive_month_rate": args.min_positive_month_rate,
            "max_top5_positive_pnl_share": args.max_top5_positive_pnl_share,
            "max_zero_trade_month_rate": args.max_zero_trade_month_rate,
            "min_stress125_avg_monthly_pnl": args.min_stress125_avg_monthly_pnl,
            "min_stress150_avg_monthly_pnl": args.min_stress150_avg_monthly_pnl,
            "min_bootstrap_p10_monthly_pnl": args.min_bootstrap_p10_monthly_pnl,
            "n_robust_pass": len(robust_rows),
            "fallback_used": fallback_used,
            "crypto_mode": bool(args.crypto_mode),
            "note": "Best candidate re-run into main --out-dir.",
        },
        "best": best,
        "all_runs": rows,
    }
    best_report = round_obj(best_report, 6)
    report_path = opt_root / "step3_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(best_report, f, separators=(",", ":"), ensure_ascii=True)

    best_cfg_path = opt_root / "step3_best_config.json"
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(best_cfg, 6), f, separators=(",", ":"), ensure_ascii=True)

    # Convenience copy of best image.
    best_img = out_root / "backtest" / "step3_dual_portfolio_curve.png"
    if best_img.exists():
        ensure_dir(str(opt_root / "best"))
        shutil.copy2(best_img, opt_root / "best" / "step3_dual_portfolio_curve_best.png")

    final_dir = export_final_artifacts(
        repo_root=repo_root,
        out_root=out_root,
        opt_root=opt_root,
        final_dir_input=args.final_dir,
        best_cfg=best_cfg,
        best_row=best,
    )

    print(f"[STEP3-OPT] Wrote: {report_path}")
    print(f"[STEP3-OPT] Wrote: {best_cfg_path}")
    print(f"[STEP3-OPT] Exported final artifacts to: {final_dir}")
    print(f"[STEP3-OPT] Best run: {best['name']} score={best['objective_score']:.4f}")


if __name__ == "__main__":
    main()
