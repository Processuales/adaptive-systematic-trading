#!/usr/bin/env python3
"""
One-click Step 2 pipeline runner.

Run with no arguments:
    python step2_run_all.py

Default behavior:
1) Rebuild Step 2 datasets (bar_features/events/meta)
2) Run Step 2.5 analysis + figures
3) Run Step 2 comparison/selection (multi-scenario Step2b sweeps)
4) Build candidate capital curve visual (no-ML vs ML)
5) Build optimized dual-symbol portfolio curves (no-ML + ML-sim)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_VERSION = "1.0.0"


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"[STEP2-RUN] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20, help="Trials per scenario in step2_compare_and_select")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sweeps")
    ap.add_argument("--start-capital", type=float, default=10000.0, help="Start capital for visuals")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols for Step 2 build")
    ap.add_argument("--trade-symbol", default="QQQ", help="Primary trade symbol for Step 2 build outputs")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    data_dir = repo_root / "data_clean"
    step2_out = repo_root / "step2_out"
    selection_dir = step2_out / "selection"
    step25_dir = step2_out / "step2_5"

    print(f"[STEP2-RUN] script_version={SCRIPT_VERSION}")
    print(f"[STEP2-RUN] generated_utc={datetime.now(timezone.utc).isoformat()}")

    # 1) Step 2 dataset build
    run_cmd(
        [
            py,
            str(repo_root / "step2_build_events_dataset.py"),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(step2_out),
            "--symbols",
            *[s.upper() for s in args.symbols],
            "--trade-symbol",
            args.trade_symbol.upper(),
        ],
        cwd=str(repo_root),
    )

    # 2) Step 2.5 analysis
    run_cmd(
        [
            py,
            str(repo_root / "step2_5_analyze_events.py"),
            "--events-path",
            str(step2_out / "events" / f"{args.trade_symbol.lower()}_events.parquet"),
            "--bar-features-path",
            str(step2_out / "bar_features" / f"{args.trade_symbol.lower()}_bar_features.parquet"),
            "--out-dir",
            str(step25_dir),
        ],
        cwd=str(repo_root),
    )

    # 3) Candidate selection (runs Step2b scenarios internally)
    run_cmd(
        [
            py,
            str(repo_root / "step2_compare_and_select.py"),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(selection_dir),
            "--n-trials",
            str(args.n_trials),
            "--seed",
            str(args.seed),
        ],
        cwd=str(repo_root),
    )

    # 4) Candidate capital curve visual
    run_cmd(
        [
            py,
            str(repo_root / "step2_capital_curve.py"),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(selection_dir / "visual"),
            "--candidate-non-ml",
            str(selection_dir / "best_candidate_non_ml.json"),
            "--candidate-ml",
            str(selection_dir / "best_candidate_ml.json"),
            "--start-capital",
            str(args.start_capital),
        ],
        cwd=str(repo_root),
    )

    # 5) Dual-symbol optimized portfolio (no-ML)
    run_cmd(
        [
            py,
            str(repo_root / "step2_dual_symbol_portfolio_test.py"),
            "--data-dir",
            str(data_dir),
            "--selection-dir",
            str(selection_dir),
            "--out-dir",
            str(selection_dir / "dual_portfolio"),
            "--mode",
            "no_ml",
            "--start-capital",
            str(args.start_capital),
            "--objective",
            "calmar",
            "--train-ratio",
            "0.6",
            "--weight-step",
            "0.05",
        ],
        cwd=str(repo_root),
    )

    # 6) Dual-symbol optimized portfolio (ML-sim)
    run_cmd(
        [
            py,
            str(repo_root / "step2_dual_symbol_portfolio_test.py"),
            "--data-dir",
            str(data_dir),
            "--selection-dir",
            str(selection_dir),
            "--out-dir",
            str(selection_dir / "dual_portfolio_ml"),
            "--mode",
            "ml_sim",
            "--start-capital",
            str(args.start_capital),
            "--objective",
            "calmar",
            "--train-ratio",
            "0.6",
            "--weight-step",
            "0.05",
        ],
        cwd=str(repo_root),
    )

    print("[STEP2-RUN] COMPLETE")
    print(f"[STEP2-RUN] Main outputs in: {step2_out}")
    print(f"[STEP2-RUN] Candidate files in: {selection_dir}")


if __name__ == "__main__":
    main()
