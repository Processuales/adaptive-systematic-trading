#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[STEP4-RUN] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def score_for_promotion(summary: Dict) -> float:
    sel = (summary.get("selected_candidate") or {})
    m = sel.get("metrics") or {}
    t = m.get("test") or {}
    avg_w = float(m.get("avg_weight_spyqqq") or 0.5)
    weight_std = float(m.get("weight_std_spyqqq") or 0.0)
    turnover = float(m.get("avg_turnover_abs") or 0.0)
    p10 = float(m.get("bootstrap_p10_avg_monthly_pnl") or 0.0)
    diversity = max(0.0, 1.0 - abs(avg_w - 0.5) * 2.0)
    activity = min(1.0, turnover / 0.08)
    return float(
        0.70 * float(t.get("end_equity") or 0.0)
        + 175.0 * float(t.get("calmar") or 0.0)
        + 1.5 * float(t.get("avg_monthly_pnl") or 0.0)
        - 300.0 * float(t.get("max_drawdown") or 0.0)
        + 120.0 * diversity
        + 60.0 * activity
        + 30.0 * min(1.0, weight_std / 0.20)
        + 0.6 * max(0.0, p10)
    )


def dynamic_score(summary: Dict) -> float:
    sel = (summary.get("selected_candidate") or {})
    m = sel.get("metrics") or {}
    t = m.get("test") or {}
    avg_w = float(m.get("avg_weight_spyqqq") or 0.5)
    turnover = float(m.get("avg_turnover_abs") or 0.0)
    diversity = max(0.0, 1.0 - abs(avg_w - 0.5) * 2.0)
    return float(
        0.45 * float(t.get("end_equity") or 0.0)
        + 165.0 * float(t.get("calmar") or 0.0)
        + 1.2 * float(t.get("avg_monthly_pnl") or 0.0)
        - 340.0 * float(t.get("max_drawdown") or 0.0)
        + 130.0 * diversity
        + 80.0 * min(1.0, turnover / 0.08)
    )


def is_dynamic_candidate(summary: Dict) -> bool:
    sel = (summary.get("selected_candidate") or {})
    cfg = sel.get("config") or {}
    m = sel.get("metrics") or {}
    avg_w = float(m.get("avg_weight_spyqqq") or 0.5)
    turnover = float(m.get("avg_turnover_abs") or 0.0)
    return bool(
        str(cfg.get("method") or "") != "static"
        and turnover >= 0.02
        and 0.08 <= avg_w <= 0.92
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spyqqq-monthly", default="combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet")
    ap.add_argument("--btceth-monthly", default="other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet")
    ap.add_argument("--out-dir", default="step4_out")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--min-history-months", type=int, default=24)
    ap.add_argument("--bootstrap-samples", type=int, default=1200)
    ap.add_argument("--bootstrap-block-months", type=int, default=6)
    ap.add_argument("--promote-dd-cap", type=float, default=0.30)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs = [
        ("return", 0.32),
        ("balanced", 0.24),
        ("calmar", 0.20),
        ("robust", 0.18),
    ]
    summaries: List[Dict] = []

    for objective, dd_cap in runs:
        run_dir = out_root / objective
        cmd = [
            sys.executable,
            str(repo / "step4_optimize_allocator.py"),
            "--spyqqq-monthly",
            str(Path(args.spyqqq_monthly).resolve()),
            "--btceth-monthly",
            str(Path(args.btceth_monthly).resolve()),
            "--out-dir",
            str(run_dir),
            "--objective",
            objective,
            "--max-drawdown-cap",
            str(dd_cap),
            "--start-capital",
            str(args.start_capital),
            "--train-ratio",
            str(args.train_ratio),
            "--min-history-months",
            str(args.min_history_months),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-block-months",
            str(args.bootstrap_block_months),
        ]
        run_cmd(cmd, cwd=repo)
        s_path = run_dir / "backtest" / "step4_summary.json"
        s = read_json(s_path)
        s["run_objective"] = objective
        s["run_dd_cap"] = dd_cap
        summaries.append(s)

    eligible = []
    for s in summaries:
        test = (((s.get("selected_candidate") or {}).get("metrics") or {}).get("test") or {})
        if float(test.get("max_drawdown") or 1.0) <= float(args.promote_dd_cap):
            eligible.append(s)
    pick_pool = eligible if eligible else summaries
    promoted = max(pick_pool, key=score_for_promotion)
    promoted_obj = str(promoted.get("run_objective") or "")
    promoted_dir = out_root / promoted_obj

    dyn_pool = [s for s in pick_pool if is_dynamic_candidate(s)]
    if not dyn_pool:
        dyn_pool = [s for s in summaries if is_dynamic_candidate(s)]
    promoted_dyn = max(dyn_pool, key=dynamic_score) if dyn_pool else None
    promoted_dyn_obj = str((promoted_dyn or {}).get("run_objective") or "")
    promoted_dyn_dir = out_root / promoted_dyn_obj if promoted_dyn_obj else None

    final_dir = out_root / "final"
    (final_dir / "backtest").mkdir(parents=True, exist_ok=True)
    (final_dir / "optimization").mkdir(parents=True, exist_ok=True)
    copy_if_exists(promoted_dir / "backtest" / "step4_portfolio_curve.png", final_dir / "backtest" / "step4_portfolio_curve.png")
    copy_if_exists(promoted_dir / "backtest" / "step4_monthly_table.parquet", final_dir / "backtest" / "step4_monthly_table.parquet")
    copy_if_exists(promoted_dir / "backtest" / "step4_summary.json", final_dir / "backtest" / "step4_summary.json")
    copy_if_exists(
        promoted_dir / "optimization" / "step4_optimization_report.json",
        final_dir / "optimization" / "step4_optimization_report.json",
    )

    final_dyn_dir = out_root / "final_dynamic"
    if promoted_dyn_dir is not None:
        (final_dyn_dir / "backtest").mkdir(parents=True, exist_ok=True)
        (final_dyn_dir / "optimization").mkdir(parents=True, exist_ok=True)
        copy_if_exists(promoted_dyn_dir / "backtest" / "step4_portfolio_curve.png", final_dyn_dir / "backtest" / "step4_portfolio_curve.png")
        copy_if_exists(promoted_dyn_dir / "backtest" / "step4_monthly_table.parquet", final_dyn_dir / "backtest" / "step4_monthly_table.parquet")
        copy_if_exists(promoted_dyn_dir / "backtest" / "step4_summary.json", final_dyn_dir / "backtest" / "step4_summary.json")
        copy_if_exists(
            promoted_dyn_dir / "optimization" / "step4_optimization_report.json",
            final_dyn_dir / "optimization" / "step4_optimization_report.json",
        )

    report = {
        "meta": {
            "script": "step4_run_all.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(out_root),
            "promote_dd_cap": float(args.promote_dd_cap),
        },
        "runs": summaries,
        "promoted_objective": promoted_obj,
        "promoted_score": score_for_promotion(promoted),
        "promoted_dynamic_objective": promoted_dyn_obj,
        "promoted_dynamic_score": dynamic_score(promoted_dyn) if promoted_dyn else None,
        "final_outputs": {
            "step4_plot": str((final_dir / "backtest" / "step4_portfolio_curve.png").resolve()),
            "step4_monthly": str((final_dir / "backtest" / "step4_monthly_table.parquet").resolve()),
            "step4_summary": str((final_dir / "backtest" / "step4_summary.json").resolve()),
            "step4_optimization_report": str((final_dir / "optimization" / "step4_optimization_report.json").resolve()),
        },
        "final_dynamic_outputs": {
            "step4_plot": str((final_dyn_dir / "backtest" / "step4_portfolio_curve.png").resolve()) if promoted_dyn else None,
            "step4_monthly": str((final_dyn_dir / "backtest" / "step4_monthly_table.parquet").resolve()) if promoted_dyn else None,
            "step4_summary": str((final_dyn_dir / "backtest" / "step4_summary.json").resolve()) if promoted_dyn else None,
            "step4_optimization_report": str((final_dyn_dir / "optimization" / "step4_optimization_report.json").resolve()) if promoted_dyn else None,
        },
    }
    with (final_dir / "step4_run_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[STEP4-RUN] promoted objective: {promoted_obj}", flush=True)
    if promoted_dyn:
        print(f"[STEP4-RUN] promoted dynamic objective: {promoted_dyn_obj}", flush=True)
    print(f"[STEP4-RUN] wrote: {final_dir / 'step4_run_report.json'}", flush=True)


if __name__ == "__main__":
    main()
