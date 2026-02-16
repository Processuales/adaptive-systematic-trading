#!/usr/bin/env python3
"""
Unified runner:
1) Run Step 2 (including dual SPY+QQQ ML simulation)
2) Build + optimize Step 3 ML
3) Export two final images into "final output"
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

import numpy as np

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
    return obj


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"[FINAL-RUN] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_clean")
    ap.add_argument("--step2-out-dir", default="step2_out")
    ap.add_argument("--step3-out-dir", default="step3_out")
    ap.add_argument("--final-dir", default="final output")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--step2-n-trials", type=int, default=36)
    ap.add_argument("--step2-seed", type=int, default=42)
    ap.add_argument("--step3-max-candidates", type=int, default=6)
    ap.add_argument("--step3-dd-cap", type=float, default=0.12)
    ap.add_argument("--step3-min-trades-month", type=float, default=6.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    step2_out = (repo_root / args.step2_out_dir).resolve()
    step3_out = (repo_root / args.step3_out_dir).resolve()
    final_dir = (repo_root / args.final_dir).resolve()
    ensure_dir(str(final_dir))

    print(f"[FINAL-RUN] script_version={SCRIPT_VERSION}")
    print(f"[FINAL-RUN] generated_utc={datetime.now(timezone.utc).isoformat()}")

    step2_cmd = [
        py,
        str(repo_root / "step2_run_all.py"),
        "--n-trials",
        str(args.step2_n_trials),
        "--seed",
        str(args.step2_seed),
        "--start-capital",
        str(args.start_capital),
        "--symbols",
        "SPY",
        "QQQ",
        "--trade-symbol",
        "QQQ",
    ]
    try:
        run_cmd(step2_cmd, cwd=str(repo_root))
    except subprocess.CalledProcessError:
        fallback_trials = max(20, args.step2_n_trials)
        if fallback_trials == args.step2_n_trials:
            raise
        print(
            f"[FINAL-RUN] Step2 failed with n-trials={args.step2_n_trials}. "
            f"Retrying with n-trials={fallback_trials}."
        )
        step2_cmd_fallback = step2_cmd.copy()
        step2_cmd_fallback[step2_cmd_fallback.index("--n-trials") + 1] = str(fallback_trials)
        run_cmd(step2_cmd_fallback, cwd=str(repo_root))

    run_cmd(
        [
            py,
            str(repo_root / "step3_build_training_dataset.py"),
            "--data-dir",
            str((repo_root / args.data_dir).resolve()),
            "--out-dir",
            str(step3_out),
        ],
        cwd=str(repo_root),
    )

    run_cmd(
        [
            py,
            str(repo_root / "step3_optimize_model.py"),
            "--dataset-dir",
            str(step3_out / "dataset"),
            "--out-dir",
            str(step3_out),
            "--start-capital",
            str(args.start_capital),
            "--dd-cap",
            str(args.step3_dd_cap),
            "--min-trades-per-month",
            str(args.step3_min_trades_month),
            "--max-candidates",
            str(args.step3_max_candidates),
        ],
        cwd=str(repo_root),
    )

    step2_img_src = step2_out / "selection" / "dual_portfolio_ml" / "dual_symbol_portfolio_curve.png"
    step3_img_src = step3_out / "backtest" / "step3_dual_portfolio_curve.png"
    if not step2_img_src.exists():
        raise FileNotFoundError(f"Missing Step 2 image: {step2_img_src}")
    if not step3_img_src.exists():
        raise FileNotFoundError(f"Missing Step 3 image: {step3_img_src}")

    step2_img_dst = final_dir / "01_step2_ml_simulation_spy_qqq.png"
    step3_img_dst = final_dir / "02_step3_real_ml_spy_qqq.png"
    shutil.copy2(step2_img_src, step2_img_dst)
    shutil.copy2(step3_img_src, step3_img_dst)

    s2 = load_json(step2_out / "selection" / "dual_portfolio_ml" / "dual_symbol_portfolio_summary.json")
    s3 = load_json(step3_out / "backtest" / "step3_summary.json")
    opt = load_json(step3_out / "optimization" / "step3_optimization_report.json")

    report = {
        "meta": {
            "script": "run_step2_step3_final.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "start_capital": args.start_capital,
        },
        "final_outputs": {
            "step2_ml_simulation_image": str(step2_img_dst),
            "step3_real_ml_image": str(step3_img_dst),
        },
        "step2_ml_simulation_snapshot": {
            "performance": s2.get("performance"),
            "pnl_stats": s2.get("pnl_stats"),
            "trades": s2.get("trades"),
            "selected_candidates": s2.get("selected_candidates"),
        },
        "step3_real_ml_snapshot": s3.get("portfolio"),
        "step3_optimization_best": opt.get("best"),
    }
    report = round_obj(report, 6)
    report_path = final_dir / "final_output_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[FINAL-RUN] Wrote: {step2_img_dst}")
    print(f"[FINAL-RUN] Wrote: {step3_img_dst}")
    print(f"[FINAL-RUN] Wrote: {report_path}")


if __name__ == "__main__":
    main()
