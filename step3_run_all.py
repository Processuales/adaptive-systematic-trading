#!/usr/bin/env python3
"""
One-click Step 3 runner.

Outputs only under step3_out.
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
    print(f"[STEP3-RUN] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_clean")
    ap.add_argument("--out-dir", default="step3_out")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--train-lookback-days", type=int, default=1095)
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--min-train-events", type=int, default=200)
    ap.add_argument("--min-val-events", type=int, default=40)
    ap.add_argument("--min-test-events", type=int, default=6)
    ap.add_argument("--mix-struct-weight", type=float, default=0.65)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    print(f"[STEP3-RUN] script_version={SCRIPT_VERSION}")
    print(f"[STEP3-RUN] generated_utc={datetime.now(timezone.utc).isoformat()}")

    out_dir = (repo_root / args.out_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)

    run_cmd(
        [
            py,
            str(repo_root / "step3_build_training_dataset.py"),
            "--data-dir",
            str((repo_root / args.data_dir).resolve()),
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(repo_root),
    )

    run_cmd(
        [
            py,
            str(repo_root / "step3_train_and_backtest.py"),
            "--dataset-dir",
            str((out_dir / "dataset").resolve()),
            "--out-dir",
            str(out_dir),
            "--start-capital",
            str(args.start_capital),
            "--train-lookback-days",
            str(args.train_lookback_days),
            "--embargo-days",
            str(args.embargo_days),
            "--min-train-events",
            str(args.min_train_events),
            "--min-val-events",
            str(args.min_val_events),
            "--min-test-events",
            str(args.min_test_events),
            "--mix-struct-weight",
            str(args.mix_struct_weight),
        ],
        cwd=str(repo_root),
    )

    print(f"[STEP3-RUN] COMPLETE. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
