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


def candidate_grid() -> List[Dict]:
    return [
        {
            "name": "balanced_default",
            "train_lookback_days": 1095,
            "embargo_days": 7,
            "min_train_events": 200,
            "min_val_events": 40,
            "min_test_events": 6,
            "mix_struct_weight": 0.65,
        },
        {
            "name": "balanced_more_ret_head",
            "train_lookback_days": 1095,
            "embargo_days": 7,
            "min_train_events": 180,
            "min_val_events": 35,
            "min_test_events": 5,
            "mix_struct_weight": 0.55,
        },
        {
            "name": "balanced_more_struct_head",
            "train_lookback_days": 1095,
            "embargo_days": 7,
            "min_train_events": 220,
            "min_val_events": 45,
            "min_test_events": 6,
            "mix_struct_weight": 0.75,
        },
        {
            "name": "shorter_memory",
            "train_lookback_days": 730,
            "embargo_days": 7,
            "min_train_events": 170,
            "min_val_events": 30,
            "min_test_events": 5,
            "mix_struct_weight": 0.60,
        },
        {
            "name": "longer_memory",
            "train_lookback_days": 1460,
            "embargo_days": 10,
            "min_train_events": 240,
            "min_val_events": 50,
            "min_test_events": 6,
            "mix_struct_weight": 0.70,
        },
        {
            "name": "low_embargo",
            "train_lookback_days": 1095,
            "embargo_days": 3,
            "min_train_events": 180,
            "min_val_events": 35,
            "min_test_events": 5,
            "mix_struct_weight": 0.65,
        },
    ]


def objective_from_summary(
    summary: Dict,
    dd_cap: float,
    min_trades_per_month: float,
) -> float:
    p = summary["portfolio"]
    perf = p["dual_perf"]
    calmar = float(perf["calmar"]) if perf.get("calmar") is not None else -1.0
    dd = float(perf["max_drawdown"]) if perf.get("max_drawdown") is not None else 1.0
    avg_pnl = float(p.get("avg_monthly_pnl") or 0.0)
    avg_tpm = float(p.get("avg_monthly_trades") or 0.0)
    aggr = float(p.get("aggressive_trade_rate") or 0.0)
    cagr = float(perf.get("cagr") or -1.0)

    score = (
        1.8 * calmar
        + 0.02 * avg_pnl
        + 0.10 * min(avg_tpm, 12.0)
        + 0.50 * cagr
        + 0.20 * aggr
    )
    score -= 4.0 * max(0.0, dd - dd_cap)
    score -= 0.80 * max(0.0, min_trades_per_month - avg_tpm)
    return float(score)


def run_cfg(
    py: str,
    repo_root: Path,
    dataset_dir: Path,
    out_dir: Path,
    start_capital: float,
    cfg: Dict,
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
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    s_path = out_dir / "backtest" / "step3_summary.json"
    if not s_path.exists():
        raise FileNotFoundError(f"Expected summary not found: {s_path}")
    with open(s_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to step3_out/dataset")
    ap.add_argument("--out-dir", required=True, help="Path to step3_out root")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--dd-cap", type=float, default=0.12)
    ap.add_argument("--min-trades-per-month", type=float, default=6.0)
    ap.add_argument("--max-candidates", type=int, default=6)
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

    grid = candidate_grid()[: max(1, args.max_candidates)]
    rows: List[Dict] = []
    for i, cfg in enumerate(grid, start=1):
        run_dir = runs_dir / f"{i:02d}_{cfg['name']}"
        ensure_dir(str(run_dir))
        status = "ok"
        err = None
        summary = None
        score = -1e18
        try:
            summary = run_cfg(
                py=py,
                repo_root=repo_root,
                dataset_dir=dataset_dir,
                out_dir=run_dir,
                start_capital=args.start_capital,
                cfg=cfg,
            )
            score = objective_from_summary(
                summary=summary,
                dd_cap=args.dd_cap,
                min_trades_per_month=args.min_trades_per_month,
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
            "summary_path": str(run_dir / "backtest" / "step3_summary.json"),
            "portfolio_snapshot": (summary.get("portfolio") if summary else None),
        }
        rows.append(row)
        print(f"[STEP3-OPT] run={i}/{len(grid)} name={cfg['name']} status={status} score={score:.4f}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("All optimization runs failed.")
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
    )

    best_report = {
        "meta": {
            "script": "step3_optimize_model.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_candidates": len(grid),
            "dd_cap": args.dd_cap,
            "min_trades_per_month": args.min_trades_per_month,
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

    print(f"[STEP3-OPT] Wrote: {report_path}")
    print(f"[STEP3-OPT] Wrote: {best_cfg_path}")
    print(f"[STEP3-OPT] Best run: {best['name']} score={best['objective_score']:.4f}")


if __name__ == "__main__":
    main()
