#!/usr/bin/env python3
"""
BTC/ETH-specific Step 3 optimizer.

Runs a small, high-value candidate set tuned for the observed BTC/ETH behavior:
- BTC leg tends to be less stable in this sample.
- ETH leg tends to carry more consistent edge.

This optimizer keeps strict walk-forward training and robustness stats, but
searches allocation constraints that can down-weight the weaker leg.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

APP = "BTC-ETH-STEP3-OPT"


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def run_cmd(cmd: List[str], cwd: Path, heartbeat_seconds: float = 20.0) -> None:
    log("$ " + " ".join(cmd))
    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    next_hb = start + max(1.0, heartbeat_seconds)
    while True:
        code = proc.poll()
        now = time.monotonic()
        if code is not None:
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)
            log(f"done elapsed={fmt_duration(now - start)}")
            return
        if now >= next_hb:
            log(f"heartbeat elapsed={fmt_duration(now - start)}")
            next_hb = now + max(1.0, heartbeat_seconds)
        time.sleep(1.0)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def stress_row(portfolio: Dict[str, Any], target_mult: float, tol: float = 0.02) -> Dict[str, Any]:
    for row in (portfolio.get("cost_stress_tests") or []):
        mult = float(row.get("cost_multiplier") or 0.0)
        if abs(mult - target_mult) <= tol:
            return row
    return {}


def metric_snapshot(summary: Dict[str, Any]) -> Dict[str, float]:
    p = summary.get("portfolio") or {}
    perf = p.get("dual_perf") or {}
    s125 = stress_row(p, 1.25)
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
        "bootstrap_p10_avg_monthly_pnl": float(boot_avg.get("p10") or 0.0),
    }


def score_snapshot(m: Dict[str, float]) -> float:
    return float(
        0.70 * m["avg_monthly_pnl"]
        + 65.0 * m["calmar"]
        - 70.0 * m["max_drawdown"]
        + 0.25 * m["stress_1_25_avg_monthly_pnl"]
        + 0.12 * m["bootstrap_p10_avg_monthly_pnl"]
    )


def should_promote(best: Dict[str, float], baseline: Dict[str, float]) -> bool:
    if best["avg_monthly_pnl"] >= baseline["avg_monthly_pnl"] + 10.0:
        return True
    if (
        best["avg_monthly_pnl"] >= baseline["avg_monthly_pnl"]
        and best["calmar"] >= baseline["calmar"]
        and best["max_drawdown"] <= baseline["max_drawdown"] + 0.02
    ):
        return True
    return False


@dataclass
class Candidate:
    name: str
    mix_struct_weight: float
    max_aggressive_size: float
    profile: str = "tilt_dynamic"
    regime_enable: bool = False
    regime_lookahead_events: int = 18
    regime_label_quantile: float = 0.55
    regime_p_cut: float = 0.52
    regime_agg_p_cut: float = 0.62
    regime_ev_scale: float = 0.80
    regime_size_scale: float = 0.30
    regime_size_min_mult: float = 0.60
    regime_size_max_mult: float = 1.20


def candidates() -> List[Candidate]:
    return [
        Candidate("balanced_equal_base", 0.65, 1.1, profile="balanced_equal"),
        Candidate(
            "balanced_equal_regime_size",
            0.65,
            1.1,
            profile="balanced_equal",
            regime_enable=True,
            regime_lookahead_events=24,
            regime_label_quantile=0.50,
            regime_p_cut=0.45,
            regime_agg_p_cut=0.50,
            regime_ev_scale=0.20,
            regime_size_scale=0.30,
            regime_size_min_mult=0.70,
            regime_size_max_mult=1.20,
        ),
        Candidate(
            "balanced_equal_regime_defensive",
            0.65,
            1.1,
            profile="balanced_equal",
            regime_enable=True,
            regime_lookahead_events=24,
            regime_label_quantile=0.58,
            regime_p_cut=0.56,
            regime_agg_p_cut=0.66,
            regime_ev_scale=0.60,
            regime_size_scale=0.35,
            regime_size_min_mult=0.60,
            regime_size_max_mult=1.10,
        ),
        Candidate("crypto_tilt_baseline_mix_0_75", 0.75, 1.0, profile="tilt_dynamic"),
        Candidate(
            "crypto_regime_soft_mix_0_75",
            0.75,
            1.0,
            profile="tilt_dynamic",
            regime_enable=True,
            regime_lookahead_events=18,
            regime_label_quantile=0.55,
            regime_p_cut=0.52,
            regime_agg_p_cut=0.62,
            regime_ev_scale=0.80,
            regime_size_scale=0.28,
            regime_size_min_mult=0.65,
            regime_size_max_mult=1.18,
        ),
        Candidate(
            "crypto_regime_balanced_mix_0_70",
            0.70,
            1.0,
            profile="tilt_dynamic",
            regime_enable=True,
            regime_lookahead_events=18,
            regime_label_quantile=0.57,
            regime_p_cut=0.54,
            regime_agg_p_cut=0.65,
            regime_ev_scale=0.95,
            regime_size_scale=0.35,
            regime_size_min_mult=0.60,
            regime_size_max_mult=1.20,
        ),
        Candidate(
            "crypto_regime_defensive_mix_0_80",
            0.80,
            1.0,
            profile="tilt_dynamic",
            regime_enable=True,
            regime_lookahead_events=24,
            regime_label_quantile=0.60,
            regime_p_cut=0.56,
            regime_agg_p_cut=0.68,
            regime_ev_scale=1.10,
            regime_size_scale=0.42,
            regime_size_min_mult=0.55,
            regime_size_max_mult=1.15,
        ),
        Candidate(
            "crypto_regime_fast_mix_0_65",
            0.65,
            1.0,
            profile="tilt_dynamic",
            regime_enable=True,
            regime_lookahead_events=12,
            regime_label_quantile=0.55,
            regime_p_cut=0.52,
            regime_agg_p_cut=0.63,
            regime_ev_scale=0.75,
            regime_size_scale=0.25,
            regime_size_min_mult=0.70,
            regime_size_max_mult=1.18,
        ),
    ]


def build_step3_cmd(
    py: str,
    repo_root: Path,
    dataset_dir: Path,
    out_dir: Path,
    start_capital: float,
    c: Candidate,
) -> List[str]:
    cmd = [
        py,
        str(repo_root / "step3_train_and_backtest.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(out_dir),
        "--start-capital",
        str(start_capital),
        "--mix-struct-weight",
        str(c.mix_struct_weight),
        "--max-aggressive-size",
        str(c.max_aggressive_size),
        "--display-symbol-1",
        "BTC",
        "--display-symbol-2",
        "ETH",
        "--exclude-leaky-features",
        "--portfolio-no-spy-guard",
        "--spy-drift-kill-switch",
        "none",
    ]
    if c.profile == "balanced_equal":
        cmd += [
            "--train-lookback-days",
            "1095",
            "--embargo-days",
            "7",
            "--min-train-events",
            "180",
            "--min-val-events",
            "35",
            "--min-test-events",
            "5",
            "--retune-every-folds",
            "4",
            "--policy-profile",
            "balanced",
            "--portfolio-allocator",
            "equal_split",
            "--portfolio-objective",
            "calmar",
        ]
    else:
        cmd += [
            "--train-lookback-days",
            "1460",
            "--embargo-days",
            "10",
            "--min-train-events",
            "220",
            "--min-val-events",
            "45",
            "--min-test-events",
            "6",
            "--retune-every-folds",
            "2",
            "--policy-profile",
            "growth",
            "--portfolio-allocator",
            "dynamic_regime_forced",
            "--portfolio-objective",
            "end_equity",
            "--portfolio-train-ratio",
            "0.55",
            "--portfolio-weight-step",
            "0.005",
            "--portfolio-lookback-days",
            "504",
            "--portfolio-min-train-days",
            "252",
            "--portfolio-turnover-penalty",
            "0.04",
            "--portfolio-weight-smoothing",
            "0.15",
            "--portfolio-momentum-days",
            "63",
            "--portfolio-vol-days",
            "21",
            "--portfolio-vol-penalty",
            "1.4",
            "--portfolio-max-tilt",
            "0.2",
            "--portfolio-min-weight",
            "0.0",
            "--portfolio-max-weight",
            "0.005",
        ]
    if c.regime_enable:
        cmd += [
            "--regime-model-enable",
            "--regime-lookahead-events",
            str(c.regime_lookahead_events),
            "--regime-label-quantile",
            str(c.regime_label_quantile),
            "--regime-p-cut",
            str(c.regime_p_cut),
            "--regime-agg-p-cut",
            str(c.regime_agg_p_cut),
            "--regime-ev-scale",
            str(c.regime_ev_scale),
            "--regime-size-scale",
            str(c.regime_size_scale),
            "--regime-size-min-mult",
            str(c.regime_size_min_mult),
            "--regime-size-max-mult",
            str(c.regime_size_max_mult),
        ]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--step3-out-dir", required=True)
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    py = sys.executable

    dataset_dir = Path(args.dataset_dir).resolve()
    step3_out_dir = Path(args.step3_out_dir).resolve()
    runs_root = step3_out_dir / "btc_eth_tilt_search_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = step3_out_dir / "backtest" / "step3_summary.json"
    if not baseline_summary_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_summary_path}")

    baseline_summary = read_json(baseline_summary_path)
    baseline_metrics = metric_snapshot(baseline_summary)
    baseline_score = score_snapshot(baseline_metrics)
    log(
        "baseline "
        f"avg_pnl={baseline_metrics['avg_monthly_pnl']:.2f} "
        f"calmar={baseline_metrics['calmar']:.4f} "
        f"dd={baseline_metrics['max_drawdown']:.4f} "
        f"score={baseline_score:.4f}"
    )

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    for idx, c in enumerate(candidates(), start=1):
        run_dir = runs_root / f"{idx:02d}_{c.name}"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        cmd = build_step3_cmd(
            py=py,
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            out_dir=run_dir,
            start_capital=args.start_capital,
            c=c,
        )
        run_cmd(cmd, cwd=repo_root, heartbeat_seconds=args.heartbeat_seconds)

        summary_path = run_dir / "backtest" / "step3_summary.json"
        summary = read_json(summary_path)
        metrics = metric_snapshot(summary)
        score = score_snapshot(metrics)

        row = {
            "name": c.name,
            "config": {
                "profile": c.profile,
                "mix_struct_weight": c.mix_struct_weight,
                "max_aggressive_size": c.max_aggressive_size,
                "portfolio_allocator": (
                    "equal_split" if c.profile == "balanced_equal" else "dynamic_regime_forced"
                ),
                "portfolio_min_weight": (None if c.profile == "balanced_equal" else 0.0),
                "portfolio_max_weight": (None if c.profile == "balanced_equal" else 0.005),
                "exclude_leaky_features": True,
                "regime_model_enable": bool(c.regime_enable),
                "regime_lookahead_events": int(c.regime_lookahead_events),
                "regime_label_quantile": float(c.regime_label_quantile),
                "regime_p_cut": float(c.regime_p_cut),
                "regime_agg_p_cut": float(c.regime_agg_p_cut),
                "regime_ev_scale": float(c.regime_ev_scale),
                "regime_size_scale": float(c.regime_size_scale),
                "regime_size_min_mult": float(c.regime_size_min_mult),
                "regime_size_max_mult": float(c.regime_size_max_mult),
            },
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "metrics": metrics,
            "score": score,
        }
        rows.append(row)
        if best_row is None or float(row["score"]) > float(best_row["score"]):
            best_row = row

        log(
            f"candidate={c.name} "
            f"profile={c.profile} "
            f"regime={'on' if c.regime_enable else 'off'} "
            f"avg_pnl={metrics['avg_monthly_pnl']:.2f} "
            f"calmar={metrics['calmar']:.4f} dd={metrics['max_drawdown']:.4f} "
            f"score={score:.4f}"
        )

    if best_row is None:
        raise RuntimeError("No candidate results produced.")

    best_metrics = best_row["metrics"]
    promote = should_promote(best_metrics, baseline_metrics)

    promoted_from = "baseline"
    if promote:
        best_backtest_dir = Path(best_row["run_dir"]) / "backtest"
        target_backtest_dir = step3_out_dir / "backtest"
        if target_backtest_dir.exists():
            shutil.rmtree(target_backtest_dir, ignore_errors=True)
        shutil.copytree(best_backtest_dir, target_backtest_dir)
        promoted_from = str(best_row["name"])
        log(f"promoted candidate -> step3_out/backtest: {best_row['name']}")
    else:
        log("no promotion: baseline remains best under promotion rules")

    report = {
        "meta": {
            "script": "optimize_step3_btc_eth.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(dataset_dir),
            "step3_out_dir": str(step3_out_dir),
            "start_capital": float(args.start_capital),
            "promotion_rule": "promote if avg_monthly_pnl improves by >=10 OR "
            "avg_pnl+calmar improve with no material drawdown increase",
        },
        "baseline": {
            "summary_path": str(baseline_summary_path),
            "metrics": baseline_metrics,
            "score": baseline_score,
        },
        "best_candidate": best_row,
        "promoted_from": promoted_from,
        "promoted": bool(promote),
        "candidates": rows,
    }
    report_path = step3_out_dir / "optimization" / "btc_eth_tilt_search_report.json"
    write_json(report_path, report)
    log(f"wrote report: {report_path}")

    best_cfg_path = step3_out_dir / "optimization" / "step3_best_config.json"
    if best_cfg_path.exists() and promote:
        cfg = read_json(best_cfg_path)
        cfg["btc_eth_override"] = {
            "enabled": True,
            "source": "optimize_step3_btc_eth.py",
            "selected_candidate": best_row["name"],
            "selected_metrics": best_metrics,
        }
        write_json(best_cfg_path, cfg)
        log(f"annotated best config: {best_cfg_path}")


if __name__ == "__main__":
    main()
