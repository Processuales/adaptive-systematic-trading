#!/usr/bin/env python3
"""
Unified runner:
1) Run Step 2 (including dual SPY+QQQ ML simulation)
2) Build + optimize Step 3 ML
3) Export organized final artifacts + two explainer charts for non-traders
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

SCRIPT_VERSION = "1.6.0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
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


def run_cmd(cmd: List[str], cwd: str) -> None:
    print(f"[FINAL-RUN] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(float(x)):,.0f}"


def safe_tag(text: str, fallback: str = "pair") -> str:
    s = str(text or "").strip().lower()
    if not s:
        return fallback
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_"):
            out.append("_")
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or fallback


def normalize_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    x = monthly.copy()
    x["month_end"] = pd.to_datetime(x["month_end"], utc=True, errors="coerce")
    x = x.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    for c in ["equity", "pnl", "ret", "trades"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        else:
            x[c] = 0.0
    return x


def plot_explainer(
    out_path: Path,
    title: str,
    subtitle: str,
    monthly: pd.DataFrame,
    start_capital: float,
    stats_lines: List[str],
) -> None:
    m = normalize_monthly(monthly)
    if m.empty:
        raise RuntimeError(f"Monthly table is empty; cannot plot {out_path}")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [3.2, 1.4, 1.1]},
    )

    ax1.plot(m["month_end"], m["equity"], color="#1d3557", linewidth=2.2, label="Portfolio equity")
    ax1.axhline(start_capital, color="black", linewidth=0.9, linestyle="--", alpha=0.65, label="Start capital")
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
    ax3.bar(m["month_end"], trades, width=20, color="#264653", alpha=0.7, label="Trades/month")
    trades_roll = pd.Series(trades).rolling(3, min_periods=1).mean().to_numpy(dtype=float)
    ax3.plot(m["month_end"], trades_roll, color="#e76f51", linewidth=1.6, label="3-month avg trades")
    ax3.set_title("Trading Activity")
    ax3.set_ylabel("# Trades")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def remove_known_temp_dirs(step3_out: Path) -> List[str]:
    removed: List[str] = []
    for name in ("forced_dynamic_test", "ultimate_single_test", "high_gain_test"):
        p = step3_out / name
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    return removed


def make_step2_chart(
    final_chart_path: Path,
    s2_summary: Dict,
    start_capital: float,
    display_symbols: tuple[str, str],
) -> Dict:
    perf = s2_summary.get("performance", {})
    trades = s2_summary.get("trades", {})
    pnl_stats = s2_summary.get("pnl_stats", {})
    monthly = pd.DataFrame(s2_summary.get("monthly_table", []))

    avg_pnl = float(pnl_stats.get("avg_monthly_pnl") or 0.0)
    med_pnl = float(pnl_stats.get("median_monthly_pnl") or 0.0)
    avg_tpm = float(trades.get("avg_monthly_trades") or 0.0)
    total_trades = int(trades.get("total_trades") or 0)
    end_eq = float(perf.get("end_equity") or start_capital)
    cagr = float(perf.get("cagr") or 0.0)
    mdd = float(perf.get("max_drawdown") or 0.0)
    calmar = perf.get("calmar")
    if (not monthly.empty) and ("pnl" in monthly.columns):
        pnl_series = pd.to_numeric(monthly["pnl"], errors="coerce").fillna(0.0)
        pos_rate = float((pnl_series > 0.0).mean())
    else:
        pos_rate = 0.0

    best_month = pnl_stats.get("best_month", {})
    worst_month = pnl_stats.get("worst_month", {})
    best_day = pnl_stats.get("best_day", {})
    worst_day = pnl_stats.get("worst_day", {})

    stats_lines = [
        f"End equity: {as_money(end_eq)}",
        f"CAGR: {100.0 * cagr:.2f}%  |  Max drawdown: {100.0 * mdd:.2f}%",
        f"Calmar: {float(calmar):.2f}" if calmar is not None else "Calmar: n/a",
        f"Average monthly PnL: {as_money(avg_pnl)}  |  Median: {as_money(med_pnl)}",
        f"Positive months: {100.0 * pos_rate:.1f}%  |  Avg trades/month: {avg_tpm:.1f}",
        (
            f"Best month: {str(best_month.get('month_end', 'n/a'))[:7]} {as_money(best_month.get('pnl', 0.0))}"
            if best_month
            else "Best month: n/a"
        ),
        (
            f"Worst month: {str(worst_month.get('month_end', 'n/a'))[:7]} {as_money(worst_month.get('pnl', 0.0))}"
            if worst_month
            else "Worst month: n/a"
        ),
        (
            f"Best day: {str(best_day.get('date_utc', 'n/a'))[:10]} {as_money(best_day.get('pnl', 0.0))}"
            if best_day
            else "Best day: n/a"
        ),
        (
            f"Worst day: {str(worst_day.get('date_utc', 'n/a'))[:10]} {as_money(worst_day.get('pnl', 0.0))}"
            if worst_day
            else "Worst day: n/a"
        ),
    ]
    plot_explainer(
        out_path=final_chart_path,
        title=f"Step 2 ML Simulation ({display_symbols[0]} + {display_symbols[1]})",
        subtitle="Candidate/policy simulation before full walk-forward ML",
        monthly=monthly,
        start_capital=start_capital,
        stats_lines=stats_lines,
    )
    return {
        "avg_monthly_pnl": avg_pnl,
        "median_monthly_pnl": med_pnl,
        "avg_monthly_trades": avg_tpm,
        "positive_month_rate": pos_rate,
        "end_equity": end_eq,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": (float(calmar) if calmar is not None else None),
        "total_trades": total_trades,
    }


def make_step3_chart(
    final_chart_path: Path,
    s3_summary: Dict,
    monthly_path: Path,
    start_capital: float,
    display_symbols: tuple[str, str],
) -> Dict:
    p = s3_summary.get("portfolio", {})
    perf = p.get("dual_perf", {})
    monthly = pd.read_parquet(monthly_path)

    avg_pnl = float(p.get("avg_monthly_pnl") or 0.0)
    med_pnl = float(p.get("median_monthly_pnl") or 0.0)
    avg_tpm = float(p.get("avg_monthly_trades") or 0.0)
    total_trades = int(p.get("total_trades") or 0)
    if "monthly_positive_rate" in p:
        pos_rate = float(p.get("monthly_positive_rate") or 0.0)
    else:
        m_tmp = normalize_monthly(monthly)
        pos_rate = float((m_tmp["pnl"] > 0.0).mean()) if not m_tmp.empty else 0.0
    end_eq = float(perf.get("end_equity") or start_capital)
    cagr = float(perf.get("cagr") or 0.0)
    mdd = float(perf.get("max_drawdown") or 0.0)
    calmar = perf.get("calmar")
    allocator = p.get("allocator", {})

    best_month = p.get("best_month", {})
    worst_month = p.get("worst_month", {})
    best_day = p.get("best_day", {})
    worst_day = p.get("worst_day", {})
    stress = p.get("cost_stress_tests") or []
    boot = p.get("bootstrap") or {}
    stress_125 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.25) <= 0.03), None)
    stress_150 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.50) <= 0.03), None)
    if stress_125:
        s125_pnl = float(stress_125.get("avg_monthly_pnl") or 0.0)
        s125_calmar = float((stress_125.get("dual_perf") or {}).get("calmar") or 0.0)
        stress_125_line = f"Cost stress x1.25: avg monthly {as_money(s125_pnl)}, calmar {s125_calmar:.2f}"
    else:
        stress_125_line = "Cost stress x1.25: n/a"
    if stress_150:
        s150_pnl = float(stress_150.get("avg_monthly_pnl") or 0.0)
        s150_calmar = float((stress_150.get("dual_perf") or {}).get("calmar") or 0.0)
        stress_150_line = f"Cost stress x1.50: avg monthly {as_money(s150_pnl)}, calmar {s150_calmar:.2f}"
    else:
        stress_150_line = "Cost stress x1.50: n/a"

    boot_line = "Bootstrap P10/P50/P90 monthly PnL: n/a"
    if bool(boot.get("enabled")):
        b = boot.get("avg_monthly_pnl") or {}
        boot_line = (
            "Bootstrap P10/P50/P90 monthly PnL: "
            f"{as_money(b.get('p10', 0.0))}/{as_money(b.get('p50', 0.0))}/{as_money(b.get('p90', 0.0))}"
        )

    stats_lines = [
        f"End equity: {as_money(end_eq)}",
        f"CAGR: {100.0 * cagr:.2f}%  |  Max drawdown: {100.0 * mdd:.2f}%",
        f"Calmar: {float(calmar):.2f}" if calmar is not None else "Calmar: n/a",
        f"Average monthly PnL: {as_money(avg_pnl)}  |  Median: {as_money(med_pnl)}",
        f"Positive months: {100.0 * pos_rate:.1f}%  |  Avg trades/month: {avg_tpm:.1f}",
        (
            f"Allocator: {allocator.get('selected', 'n/a')}  |  QQQ weight avg/min/max: "
            f"{float(allocator.get('qqq_weight_mean') or 0.0):.2f}/"
            f"{float(allocator.get('qqq_weight_min') or 0.0):.2f}/"
            f"{float(allocator.get('qqq_weight_max') or 0.0):.2f}"
        ),
        (
            f"Best month: {str(best_month.get('month_end', 'n/a'))[:7]} {as_money(best_month.get('pnl', 0.0))}"
            if best_month
            else "Best month: n/a"
        ),
        (
            f"Worst month: {str(worst_month.get('month_end', 'n/a'))[:7]} {as_money(worst_month.get('pnl', 0.0))}"
            if worst_month
            else "Worst month: n/a"
        ),
        (
            f"Best day: {str(best_day.get('date_utc', 'n/a'))[:10]} {as_money(best_day.get('pnl', 0.0))}"
            if best_day
            else "Best day: n/a"
        ),
        (
            f"Worst day: {str(worst_day.get('date_utc', 'n/a'))[:10]} {as_money(worst_day.get('pnl', 0.0))}"
            if worst_day
            else "Worst day: n/a"
        ),
        stress_125_line,
        stress_150_line,
        boot_line,
    ]
    plot_explainer(
        out_path=final_chart_path,
        title=f"Step 3 Real ML Backtest ({display_symbols[0]} + {display_symbols[1]})",
        subtitle="Walk-forward training/testing with risk-mode adaptation",
        monthly=monthly,
        start_capital=start_capital,
        stats_lines=stats_lines,
    )
    return {
        "avg_monthly_pnl": avg_pnl,
        "median_monthly_pnl": med_pnl,
        "avg_monthly_trades": avg_tpm,
        "positive_month_rate": pos_rate,
        "end_equity": end_eq,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": (float(calmar) if calmar is not None else None),
        "total_trades": total_trades,
        "stress_1_25_avg_monthly_pnl": (
            float(stress_125.get("avg_monthly_pnl")) if (stress_125 and stress_125.get("avg_monthly_pnl") is not None) else None
        ),
        "stress_1_50_avg_monthly_pnl": (
            float(stress_150.get("avg_monthly_pnl")) if (stress_150 and stress_150.get("avg_monthly_pnl") is not None) else None
        ),
        "bootstrap_p10_avg_monthly_pnl": (
            float((boot.get("avg_monthly_pnl") or {}).get("p10")) if bool(boot.get("enabled")) else None
        ),
    }


def write_final_readme(path: Path) -> None:
    lines = [
        "Final Output Layout",
        "",
        "charts/: human-friendly summary charts for Step 2 and Step 3",
        "step2/: Step 2 raw artifacts (summary JSON + raw strategy chart)",
        "step3/: Step 3 raw artifacts from optimizer export",
        "reports/: consolidated machine-readable final summary",
        "",
        "Use reports/final_output_summary.json for automation and comparisons.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    ap.add_argument(
        "--same-bar-policy",
        choices=["worst", "best", "close_direction"],
        default="worst",
        help="Same-bar TP/SL ambiguity policy for Step 2 and Step 3 dataset builds.",
    )
    ap.add_argument("--pair-tag", default="spy_qqq", help="Slug used in exported chart filenames.")
    ap.add_argument("--display-symbol-1", default="SPY", help="Display symbol used for chart/report labels.")
    ap.add_argument("--display-symbol-2", default="QQQ", help="Display symbol used for chart/report labels.")
    ap.add_argument("--run-pattern-experiment", action="store_true", help="Run Step 3 pattern-aid dual-ML comparison.")
    ap.add_argument(
        "--pattern-reuse-existing-runs",
        action="store_true",
        help="When running pattern experiment, reuse existing step3_out/pattern_aid_runs if available.",
    )
    ap.add_argument("--skip-step2", action="store_true", help="Skip running Step 2 and reuse existing step2_out.")
    ap.add_argument("--skip-step3", action="store_true", help="Skip running Step 3 and reuse existing step3_out.")
    ap.add_argument("--no-clean-final-dir", action="store_true", help="Do not wipe final-dir before export.")
    ap.add_argument("--no-prune-step3-temp", action="store_true", help="Do not remove known temporary step3_out folders.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    step2_out = (repo_root / args.step2_out_dir).resolve()
    step3_out = (repo_root / args.step3_out_dir).resolve()
    final_dir = (repo_root / args.final_dir).resolve()
    pair_tag = safe_tag(args.pair_tag, fallback="pair")
    display_symbols = (str(args.display_symbol_1).upper(), str(args.display_symbol_2).upper())

    if not args.no_clean_final_dir:
        reset_dir(final_dir)
    else:
        ensure_dir(str(final_dir))

    charts_dir = final_dir / "charts"
    reports_dir = final_dir / "reports"
    step2_final_dir = final_dir / "step2"
    step3_final_dir = final_dir / "step3"
    dual_ml_dir = final_dir / "dual ml"
    ensure_dir(str(charts_dir))
    ensure_dir(str(reports_dir))
    ensure_dir(str(step2_final_dir))
    ensure_dir(str(step3_final_dir))

    print(f"[FINAL-RUN] script_version={SCRIPT_VERSION}")
    print(f"[FINAL-RUN] generated_utc={datetime.now(timezone.utc).isoformat()}")

    if not args.skip_step2:
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
            "--same-bar-policy",
            args.same_bar_policy,
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
    else:
        print("[FINAL-RUN] Skipping Step 2 run. Reusing existing step2_out.")

    if not args.skip_step3:
        run_cmd(
            [
                py,
                str(repo_root / "step3_build_training_dataset.py"),
                "--data-dir",
                str((repo_root / args.data_dir).resolve()),
                "--out-dir",
                str(step3_out),
                "--same-bar-policy",
                args.same_bar_policy,
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
                "--final-dir",
                str(step3_final_dir),
            ],
            cwd=str(repo_root),
        )
    else:
        print("[FINAL-RUN] Skipping Step 3 run. Reusing existing step3_out.")

    if not args.no_prune_step3_temp:
        removed = remove_known_temp_dirs(step3_out)
        if removed:
            print(f"[FINAL-RUN] Pruned temporary step3 folders: {len(removed)}")

    pattern_json = dual_ml_dir / "01_pattern_ml_dual_comparison.json"
    if args.run_pattern_experiment:
        pattern_cmd = [
            py,
            str(repo_root / "step3_pattern_aid_experiment.py"),
            "--dataset-dir",
            str(step3_out / "dataset"),
            "--step3-out-dir",
            str(step3_out),
            "--best-config-path",
            str(step3_out / "optimization" / "step3_best_config.json"),
            "--final-dir",
            str(dual_ml_dir),
            "--start-capital",
            str(args.start_capital),
        ]
        if args.pattern_reuse_existing_runs:
            pattern_cmd.append("--reuse-existing-runs")
        run_cmd(pattern_cmd, cwd=str(repo_root))

    step2_img_src = step2_out / "selection" / "dual_portfolio_ml" / "dual_symbol_portfolio_curve.png"
    step2_json_src = step2_out / "selection" / "dual_portfolio_ml" / "dual_symbol_portfolio_summary.json"
    step3_img_src = step3_out / "backtest" / "step3_dual_portfolio_curve.png"
    step3_json_src = step3_out / "backtest" / "step3_summary.json"
    step3_monthly_src = step3_out / "backtest" / "step3_monthly_table.parquet"
    step3_opt_src = step3_out / "optimization" / "step3_optimization_report.json"
    step3_best_cfg_src = step3_out / "optimization" / "step3_best_config.json"

    required = [step2_img_src, step2_json_src, step3_img_src, step3_json_src, step3_monthly_src]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    shutil.copy2(step2_img_src, step2_final_dir / "step2_dual_symbol_portfolio_curve_raw.png")
    shutil.copy2(step2_json_src, step2_final_dir / "step2_dual_symbol_portfolio_summary.json")
    shutil.copy2(step3_img_src, step3_final_dir / "step3_dual_portfolio_curve_raw.png")
    shutil.copy2(step3_json_src, step3_final_dir / "step3_summary_raw.json")
    shutil.copy2(step3_monthly_src, step3_final_dir / "step3_monthly_table.parquet")
    if step3_opt_src.exists():
        shutil.copy2(step3_opt_src, step3_final_dir / "step3_optimization_report.json")
    if step3_best_cfg_src.exists():
        shutil.copy2(step3_best_cfg_src, step3_final_dir / "step3_best_config.json")

    s2 = load_json(step2_json_src)
    s3 = load_json(step3_json_src)
    opt = load_json(step3_opt_src) if step3_opt_src.exists() else {}
    pattern = load_json(pattern_json) if pattern_json.exists() else {}

    step2_chart = charts_dir / f"01_step2_ml_simulation_{pair_tag}.png"
    step3_chart = charts_dir / f"02_step3_real_ml_{pair_tag}.png"
    step2_snapshot = make_step2_chart(
        step2_chart,
        s2_summary=s2,
        start_capital=args.start_capital,
        display_symbols=display_symbols,
    )
    step3_snapshot = make_step3_chart(
        step3_chart,
        s3_summary=s3,
        monthly_path=step3_monthly_src,
        start_capital=args.start_capital,
        display_symbols=display_symbols,
    )

    report = {
        "meta": {
            "script": "run_step2_step3_final.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "start_capital": args.start_capital,
            "pair_tag": pair_tag,
            "display_symbols": [display_symbols[0], display_symbols[1]],
            "clean_final_dir": (not args.no_clean_final_dir),
            "prune_step3_temp": (not args.no_prune_step3_temp),
        },
        "paths": {
            "final_root": str(final_dir),
            "step2_dir": str(step2_final_dir),
            "step3_dir": str(step3_final_dir),
            "dual_ml_dir": str(dual_ml_dir),
            "charts_dir": str(charts_dir),
            "reports_dir": str(reports_dir),
            "step2_chart": str(step2_chart),
            "step3_chart": str(step3_chart),
        },
        "snapshots": {
            "step2_ml_simulation": step2_snapshot,
            "step3_real_ml": step3_snapshot,
            "step3_active_meta": s3.get("meta"),
            "step3_optimization_best": opt.get("best"),
            "pattern_ml_decision": pattern.get("decision"),
        },
    }
    report = round_obj(report, 6)
    report_path = reports_dir / "final_output_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, separators=(",", ":"), ensure_ascii=True)

    write_final_readme(final_dir / "README.txt")

    print(f"[FINAL-RUN] Wrote: {step2_chart}")
    print(f"[FINAL-RUN] Wrote: {step3_chart}")
    print(f"[FINAL-RUN] Wrote: {report_path}")


if __name__ == "__main__":
    main()
