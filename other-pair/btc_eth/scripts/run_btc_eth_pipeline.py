#!/usr/bin/env python3
"""
Full BTC/ETH pipeline in isolated workspace:
1) Download history from IBKR (optional)
2) Prepare/clean data + alias mapping (QQQ->BTC, SPY->ETH)
3) Step 2 full pipeline
4) Step 3 full pipeline (optimize + BTC/ETH tuning + hybrid overlay)
5) Compact final charts/reports under this pair folder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APP = "BTC-ETH-PIPE"


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def run_cmd(cmd: list[str], cwd: Path, heartbeat_seconds: float = 30.0) -> None:
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
            log(f"heartbeat running elapsed={fmt_duration(now - start)}")
            next_hb = now + max(1.0, heartbeat_seconds)
        time.sleep(1.0)


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        value = part.strip()
        if value:
            out.append(int(value))
    return out


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        value = part.strip()
        if value:
            out.append(float(value))
    return out


def parse_trade_constraints(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(text).split(","):
        value = part.strip()
        if not value or ":" not in value:
            continue
        left, right = value.split(":", 1)
        out.append((int(left.strip()), int(right.strip())))
    return out


def run_step2_compare_with_fallback(
    py: str,
    repo_root: Path,
    data_dir: Path,
    selection_dir: Path,
    n_trials: int,
    seed: int,
    trade_constraints_fallback: list[tuple[int, int]],
    friction_profile: str,
    market_hours: str,
    heartbeat_seconds: float,
) -> bool:
    for mt_train, mt_test in trade_constraints_fallback:
        cmd = [
            py,
            str(repo_root / "step2_compare_and_select.py"),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(selection_dir),
            "--n-trials",
            str(n_trials),
            "--seed",
            str(seed),
            "--min-trades-train",
            str(mt_train),
            "--min-trades-test",
            str(mt_test),
            "--friction-profile",
            str(friction_profile),
            "--market-hours",
            str(market_hours),
        ]
        try:
            run_cmd(cmd, cwd=repo_root, heartbeat_seconds=heartbeat_seconds)
            return True
        except subprocess.CalledProcessError:
            log(
                "step2_compare failed with "
                f"min-train={mt_train} min-test={mt_test}; retrying..."
            )
    return False


def run_dual_with_fallback(
    py: str,
    repo_root: Path,
    data_dir: Path,
    selection_dir: Path,
    out_dir: Path,
    mode: str,
    start_capital: float,
    objectives: list[str],
    train_ratio: float,
    weight_step: float,
    fallback_min_test_trades: list[int],
    candidate_min_net_bps_fallback: list[float],
    candidate_min_cagr_fallback: list[float],
    heartbeat_seconds: float,
) -> bool:
    for objective in objectives:
        for min_test_trades in fallback_min_test_trades:
            for min_net_bps in candidate_min_net_bps_fallback:
                for min_cagr in candidate_min_cagr_fallback:
                    cmd = [
                        py,
                        str(repo_root / "step2_dual_symbol_portfolio_test.py"),
                        "--data-dir",
                        str(data_dir),
                        "--selection-dir",
                        str(selection_dir),
                        "--out-dir",
                        str(out_dir),
                        "--mode",
                        mode,
                        "--start-capital",
                        str(start_capital),
                        "--objective",
                        objective,
                        "--train-ratio",
                        str(train_ratio),
                        "--weight-step",
                        str(weight_step),
                        "--min-test-trades",
                        str(min_test_trades),
                        "--candidate-min-net-bps",
                        str(min_net_bps),
                        "--candidate-min-cagr",
                        str(min_cagr),
                    ]
                    try:
                        run_cmd(cmd, cwd=repo_root, heartbeat_seconds=heartbeat_seconds)
                        return True
                    except subprocess.CalledProcessError:
                        log(
                            "dual mode failed: "
                            f"mode={mode} objective={objective} min-test-trades={min_test_trades} "
                            f"candidate-min-net-bps={min_net_bps} candidate-min-cagr={min_cagr}; retrying..."
                        )
    return False


def write_ml_dual_fallback(selection_dir: Path) -> None:
    src = selection_dir / "dual_portfolio"
    dst = selection_dir / "dual_portfolio_ml"
    if not src.exists():
        raise RuntimeError(f"Cannot create ML fallback: source missing: {src}")
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)

    summary_path = dst / "dual_symbol_portfolio_summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    summary["ml_dual_fallback_used"] = True
    summary["ml_dual_fallback_reason"] = (
        "No eligible ML candidates met minimum-trade constraints under current search budget."
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def remove_legacy_named_outputs(final_dir: Path) -> None:
    legacy_files = [
        final_dir / "charts" / "01_step2_ml_simulation_spy_qqq.png",
        final_dir / "charts" / "02_step3_real_ml_spy_qqq.png",
    ]
    for path in legacy_files:
        if path.exists():
            path.unlink()
            log(f"removed legacy chart: {path}")


def prune_verbose_final_outputs(final_dir: Path) -> None:
    for dirname in ("step2", "step3", "dual ml"):
        path = final_dir / dirname
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            log(f"removed verbose final dir: {path}")


def write_compact_final_report(final_dir: Path, step3_out: Path, start_capital: float) -> None:
    reports_dir = final_dir / "reports"
    charts_dir = final_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "final_output_summary.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing expected report to compact: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        old = json.load(f)

    hybrid_report_path = step3_out / "optimization" / "btc_eth_hybrid_overlay_report.json"
    hybrid_report = {}
    if hybrid_report_path.exists():
        try:
            with hybrid_report_path.open("r", encoding="utf-8") as f:
                hybrid_report = json.load(f)
        except Exception:
            hybrid_report = {}

    promoted = bool(hybrid_report.get("promoted"))
    best_row = hybrid_report.get("best_candidate") or {}
    best_promotable_row = hybrid_report.get("best_promotable_candidate") or {}
    selected_row = best_promotable_row if (promoted and best_promotable_row) else best_row
    selected_name = hybrid_report.get("promoted_from") if promoted else "active_baseline"
    if not selected_name and selected_row:
        selected_name = selected_row.get("name")

    compact = {
        "meta": {
            "script": "run_btc_eth_pipeline.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "pair_tag": "btc_eth",
            "display_symbols": ["BTC", "ETH"],
            "start_capital": float(start_capital),
            "note": "Compact side-pair summary for BTC+ETH.",
            "hybrid_overlay_promoted": promoted,
        },
        "paths": {
            "final_root": str(final_dir),
            "charts_dir": str(charts_dir),
            "reports_dir": str(reports_dir),
            "step2_chart": str(charts_dir / "01_step2_ml_simulation_btc_eth.png"),
            "step3_chart": str(charts_dir / "02_step3_real_ml_btc_eth.png"),
            "hybrid_overlay_report": str(hybrid_report_path) if hybrid_report else None,
        },
        "snapshots": {
            "step2_ml_simulation": (old.get("snapshots") or {}).get("step2_ml_simulation"),
            "step3_real_ml": (old.get("snapshots") or {}).get("step3_real_ml"),
            "step3_hybrid_overlay": {
                "promoted": promoted,
                "promoted_from": hybrid_report.get("promoted_from"),
                "best_candidate": best_row.get("name"),
                "best_metrics": best_row.get("metrics"),
                "selected_candidate": selected_name,
                "selected_metrics": (selected_row or {}).get("metrics"),
            }
            if hybrid_report
            else None,
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, separators=(",", ":"), ensure_ascii=True)
    log(f"wrote compact report: {report_path}")


def as_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(float(x)):,.0f}"


def normalize_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    x = monthly.copy()
    x["month_end"] = pd.to_datetime(x["month_end"], utc=True, errors="coerce")
    x = x.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    for c in ("equity", "pnl", "ret", "trades"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        else:
            x[c] = 0.0
    return x


def rewrite_btc_eth_step3_chart(final_dir: Path, step3_out: Path, start_capital: float) -> None:
    summary_path = step3_out / "backtest" / "step3_summary.json"
    monthly_path = step3_out / "backtest" / "step3_monthly_table.parquet"
    if (not summary_path.exists()) or (not monthly_path.exists()):
        log("skip btc/eth step3 chart rewrite (missing summary/monthly source).")
        return

    with summary_path.open("r", encoding="utf-8") as f:
        s3 = json.load(f)
    monthly = normalize_monthly(pd.read_parquet(monthly_path))
    if monthly.empty:
        log("skip btc/eth step3 chart rewrite (empty monthly table).")
        return

    p = s3.get("portfolio") or {}
    perf = p.get("dual_perf") or {}
    allocator = p.get("allocator") or {}
    stress = p.get("cost_stress_tests") or []
    boot = p.get("bootstrap") or {}
    s125 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.25) <= 0.03), {})
    s150 = next((r for r in stress if abs(float(r.get("cost_multiplier") or 0.0) - 1.50) <= 0.03), {})
    bavg = (boot.get("avg_monthly_pnl") or {}) if bool(boot.get("enabled")) else {}

    avg_pnl = float(p.get("avg_monthly_pnl") or 0.0)
    med_pnl = float(p.get("median_monthly_pnl") or 0.0)
    avg_tpm = float(p.get("avg_monthly_trades") or 0.0)
    pos_rate = float(p.get("monthly_positive_rate") or 0.0)
    end_eq = float(perf.get("end_equity") or start_capital)
    cagr = float(perf.get("cagr") or 0.0)
    mdd = float(perf.get("max_drawdown") or 0.0)
    calmar = float(perf.get("calmar") or 0.0)
    best_month = p.get("best_month") or {}
    worst_month = p.get("worst_month") or {}
    mode = str(allocator.get("selected") or "n/a")

    stats_lines = [
        f"End equity: {as_money(end_eq)}",
        f"CAGR: {100.0 * cagr:.2f}%  |  Max drawdown: {100.0 * mdd:.2f}%",
        f"Calmar: {calmar:.2f}",
        f"Average monthly PnL: {as_money(avg_pnl)}  |  Median: {as_money(med_pnl)}",
        f"Positive months: {100.0 * pos_rate:.1f}%  |  Avg trades/month: {avg_tpm:.1f}",
        f"Allocator mode: {mode}",
    ]
    if mode == "btc_eth_hybrid_core_satellite":
        stats_lines.append(
            "Hybrid core/active: "
            f"core={float(allocator.get('core_fraction') or 0.0):.2f}, "
            f"active={float(allocator.get('active_sleeve_share') or 0.0):.2f}"
        )
        stats_lines.append(
            "Core BTC weight avg/min/max: "
            f"{float(allocator.get('core_btc_weight_mean') or 0.0):.2f}/"
            f"{float(allocator.get('core_btc_weight_min') or 0.0):.2f}/"
            f"{float(allocator.get('core_btc_weight_max') or 0.0):.2f}"
        )
    stats_lines += [
        (
            f"Best month: {str(best_month.get('month_end', 'n/a'))[:7]} "
            f"{as_money(best_month.get('pnl', 0.0))}"
        ),
        (
            f"Worst month: {str(worst_month.get('month_end', 'n/a'))[:7]} "
            f"{as_money(worst_month.get('pnl', 0.0))}"
        ),
        f"Cost stress x1.25 avg monthly: {as_money(s125.get('avg_monthly_pnl', 0.0))}",
        f"Cost stress x1.50 avg monthly: {as_money(s150.get('avg_monthly_pnl', 0.0))}",
        (
            "Bootstrap P10/P50/P90 monthly PnL: "
            f"{as_money(bavg.get('p10', 0.0))}/{as_money(bavg.get('p50', 0.0))}/{as_money(bavg.get('p90', 0.0))}"
            if bavg
            else "Bootstrap P10/P50/P90 monthly PnL: n/a"
        ),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [3.2, 1.4, 1.1]},
    )
    ax1.plot(monthly["month_end"], monthly["equity"], color="#1d3557", linewidth=2.2, label="Portfolio equity")
    ax1.axhline(start_capital, color="black", linewidth=0.8, linestyle="--", alpha=0.7, label="Start capital")
    ax1.set_title("Step 3 Real ML Backtest (BTC + ETH)\nWalk-forward ML with BTC/ETH risk-aware overlay")
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

    pnl = monthly["pnl"].to_numpy(dtype=float)
    colors = np.where(pnl >= 0.0, "#2a9d8f", "#d62828")
    ax2.bar(monthly["month_end"], pnl, width=20, color=colors, alpha=0.86)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_title("Monthly Profit / Loss")
    ax2.set_ylabel("Monthly PnL ($)")
    ax2.grid(alpha=0.2)

    trades = monthly["trades"].to_numpy(dtype=float)
    ax3.bar(monthly["month_end"], trades, width=20, color="#264653", alpha=0.74, label="Trades/month")
    ax3.plot(
        monthly["month_end"],
        pd.Series(trades).rolling(3, min_periods=1).mean().to_numpy(dtype=float),
        color="#e76f51",
        linewidth=1.5,
        label="3-month avg trades",
    )
    ax3.set_title("Trading Activity")
    ax3.set_ylabel("# Trades")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    step3_chart = final_dir / "charts" / "02_step3_real_ml_btc_eth.png"
    step3_chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(step3_chart, dpi=160)
    backtest_chart = step3_out / "backtest" / "step3_dual_portfolio_curve.png"
    backtest_chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(backtest_chart, dpi=160)
    plt.close(fig)
    log(f"rewrote step3 btc/eth chart without alias labels: {step3_chart}")
    log(f"rewrote step3 backtest chart without alias labels: {backtest_chart}")


def write_final_readme(final_dir: Path) -> None:
    readme_path = final_dir / "README.txt"
    lines = [
        "BTC + ETH Side-Pair Output",
        "",
        "This folder is a compact side experiment snapshot.",
        "",
        "- charts/01_step2_ml_simulation_btc_eth.png",
        "- charts/02_step3_real_ml_btc_eth.png",
        "- reports/final_output_summary.json",
        "",
        "Notes:",
        "- This pair uses the same core pipeline architecture as SPY+QQQ.",
        "- Main production benchmark remains SPY+QQQ at repository root final output.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote final readme: {readme_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-step2", action="store_true")
    ap.add_argument("--skip-step3", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--skip-pattern", action="store_true")
    ap.add_argument("--skip-step3-btceth-tune", action="store_true")
    ap.add_argument("--skip-step3-hybrid", action="store_true")
    ap.add_argument(
        "--enable-pattern",
        action="store_true",
        help="Opt-in: run BTC/ETH pattern-aid experiment (disabled by default).",
    )
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--step2-n-trials", type=int, default=48)
    ap.add_argument("--step2-seed", type=int, default=42)
    ap.add_argument(
        "--step2-trade-constraint-fallback",
        default="120:35,90:25,70:18,50:12,25:6,10:3",
        help="Comma-separated train:test fallback list for Step2 compare.",
    )
    ap.add_argument(
        "--dual-min-test-trades-fallback",
        default="30,20,12,5,0",
        help="Comma-separated fallback list for dual --min-test-trades.",
    )
    ap.add_argument(
        "--dual-objective-fallback",
        default="calmar,end_equity",
        help="Comma-separated fallback objectives for dual portfolio selection.",
    )
    ap.add_argument(
        "--dual-candidate-min-net-bps-fallback",
        default="0,-25,-1000000000",
        help="Comma-separated fallback list for --candidate-min-net-bps.",
    )
    ap.add_argument(
        "--dual-candidate-min-cagr-fallback",
        default="-0.05,-0.25,-1",
        help="Comma-separated fallback list for --candidate-min-cagr.",
    )
    ap.add_argument("--step3-max-candidates", type=int, default=8)
    ap.add_argument("--ib-host", default="127.0.0.1")
    ap.add_argument("--ib-port", type=int, default=4001)
    ap.add_argument("--ib-client-id", type=int, default=191)
    ap.add_argument("--download-years-back", type=int, default=6)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    ap.add_argument("--pattern-reuse-existing-runs", action="store_true")
    ap.add_argument("--hybrid-core-fractions", default="0.30,0.50,0.70,0.85")
    ap.add_argument("--hybrid-core-btc-shares", default="0.30,0.50,0.70,0.85")
    ap.add_argument("--hybrid-active-scales", default="1.00,0.80,0.60")
    ap.add_argument("--hybrid-core-modes", default="fixed,vol_parity_6m")
    ap.add_argument("--hybrid-min-stress125-avg-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--hybrid-min-stress150-avg-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--hybrid-min-bootstrap-p10-monthly-pnl", type=float, default=0.0)
    ap.add_argument("--hybrid-max-drawdown-cap", type=float, default=0.35)
    ap.add_argument("--hybrid-min-active-sleeve-share", type=float, default=0.05)
    ap.add_argument("--hybrid-promote-avg-monthly-pnl-margin", type=float, default=8.0)
    ap.add_argument("--hybrid-promote-calmar-tolerance", type=float, default=0.05)
    ap.add_argument("--hybrid-promote-drawdown-tolerance", type=float, default=0.03)
    ap.add_argument("--hybrid-no-promote", action="store_true")
    args = ap.parse_args()

    trade_constraints_fallback = parse_trade_constraints(args.step2_trade_constraint_fallback)
    if not trade_constraints_fallback:
        trade_constraints_fallback = [(120, 35), (90, 25), (70, 18), (50, 12), (25, 6), (10, 3)]
    dual_mt_fallback = parse_int_list(args.dual_min_test_trades_fallback) or [30, 20, 12, 5, 0]
    dual_objectives = [x.strip() for x in str(args.dual_objective_fallback).split(",") if x.strip()]
    if not dual_objectives:
        dual_objectives = ["calmar", "end_equity"]
    dual_min_bps_fallback = parse_float_list(args.dual_candidate_min_net_bps_fallback) or [0.0, -25.0, -1_000_000_000.0]
    dual_min_cagr_fallback = parse_float_list(args.dual_candidate_min_cagr_fallback) or [-0.05, -0.25, -1.0]

    script_path = Path(__file__).resolve()
    pair_root = script_path.parents[1]  # .../other-pair/btc_eth
    repo_root = script_path.parents[3]  # .../adaptive-systematic-trading
    py = sys.executable

    raw_dir = pair_root / "data" / "raw"
    clean_alias_dir = pair_root / "data_clean_alias"
    step2_out = pair_root / "step2_out"
    step3_out = pair_root / "step3_out"
    final_dir = pair_root / "final output"
    pattern_ready = True
    pattern_enabled = bool(args.enable_pattern and (not args.skip_pattern))

    log("start")
    log(
        f"workspace={pair_root} start_capital={args.start_capital} "
        f"download_years_back={args.download_years_back}"
    )
    if not pattern_enabled:
        log("pattern experiment is disabled by default for BTC/ETH (use --enable-pattern to opt in).")

    if not args.skip_download:
        run_cmd(
            [
                py,
                str(pair_root / "scripts" / "download_history_btc_eth.py"),
                "--host",
                args.ib_host,
                "--port",
                str(args.ib_port),
                "--client-id",
                str(args.ib_client_id),
                "--years-back",
                str(args.download_years_back),
                "--symbols",
                "BTC",
                "ETH",
                "--out-dir",
                str(raw_dir),
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    if not args.skip_prepare:
        run_cmd(
            [
                py,
                str(pair_root / "scripts" / "prepare_btc_eth_data.py"),
                "--raw-dir",
                str(raw_dir),
                "--clean-dir",
                str(pair_root / "data_clean"),
                "--alias-dir",
                str(clean_alias_dir),
                "--qqq-alias-source",
                "BTC",
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    if not args.skip_step2:
        step25_dir = step2_out / "step2_5"
        selection_dir = step2_out / "selection"

        run_cmd(
            [
                py,
                str(repo_root / "step2_build_events_dataset.py"),
                "--data-dir",
                str(clean_alias_dir),
                "--out-dir",
                str(step2_out),
                "--symbols",
                "SPY",
                "QQQ",
                "--trade-symbol",
                "QQQ",
                "--friction-profile",
                "crypto",
                "--market-hours",
                "24_7",
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        run_cmd(
            [
                py,
                str(repo_root / "step2_5_analyze_events.py"),
                "--events-path",
                str(step2_out / "events" / "qqq_events.parquet"),
                "--bar-features-path",
                str(step2_out / "bar_features" / "qqq_bar_features.parquet"),
                "--out-dir",
                str(step25_dir),
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        ok_compare = run_step2_compare_with_fallback(
            py=py,
            repo_root=repo_root,
            data_dir=clean_alias_dir,
            selection_dir=selection_dir,
            n_trials=args.step2_n_trials,
            seed=args.step2_seed,
            trade_constraints_fallback=trade_constraints_fallback,
            friction_profile="crypto",
            market_hours="24_7",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if not ok_compare:
            raise RuntimeError("Step2 compare/select failed for all fallback constraints.")

        run_cmd(
            [
                py,
                str(repo_root / "step2_capital_curve.py"),
                "--data-dir",
                str(clean_alias_dir),
                "--out-dir",
                str(selection_dir / "visual"),
                "--candidate-non-ml",
                str(selection_dir / "best_candidate_non_ml.json"),
                "--candidate-ml",
                str(selection_dir / "best_candidate_ml.json"),
                "--start-capital",
                str(args.start_capital),
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )

        ok_no_ml = run_dual_with_fallback(
            py=py,
            repo_root=repo_root,
            data_dir=clean_alias_dir,
            selection_dir=selection_dir,
            out_dir=selection_dir / "dual_portfolio",
            mode="no_ml",
            start_capital=args.start_capital,
            objectives=dual_objectives,
            train_ratio=0.6,
            weight_step=0.05,
            fallback_min_test_trades=dual_mt_fallback,
            candidate_min_net_bps_fallback=dual_min_bps_fallback,
            candidate_min_cagr_fallback=dual_min_cagr_fallback,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if not ok_no_ml:
            raise RuntimeError("Step2 dual no-ML failed for all fallback values.")

        ok_ml = run_dual_with_fallback(
            py=py,
            repo_root=repo_root,
            data_dir=clean_alias_dir,
            selection_dir=selection_dir,
            out_dir=selection_dir / "dual_portfolio_ml",
            mode="ml_sim",
            start_capital=args.start_capital,
            objectives=dual_objectives,
            train_ratio=0.6,
            weight_step=0.05,
            fallback_min_test_trades=dual_mt_fallback,
            candidate_min_net_bps_fallback=dual_min_bps_fallback,
            candidate_min_cagr_fallback=dual_min_cagr_fallback,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if not ok_ml:
            log("ML dual portfolio failed; writing fallback from no-ML dual portfolio.")
            write_ml_dual_fallback(selection_dir)

    if not args.skip_step3:
        run_cmd(
            [
                py,
                str(repo_root / "step3_build_training_dataset.py"),
                "--data-dir",
                str(clean_alias_dir),
                "--out-dir",
                str(step3_out),
                "--friction-profile",
                "crypto",
                "--market-hours",
                "24_7",
            ],
            cwd=repo_root,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        try:
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
                    "--max-candidates",
                    str(args.step3_max_candidates),
                    "--crypto-mode",
                    "--final-dir",
                    str(final_dir / "step3"),
                ],
                cwd=repo_root,
                heartbeat_seconds=args.heartbeat_seconds,
            )
        except subprocess.CalledProcessError:
            log("step3_optimize_model failed; using short-history fallback profile.")
            pattern_ready = False
            opt_dir = step3_out / "optimization"
            if opt_dir.exists():
                shutil.rmtree(opt_dir, ignore_errors=True)
            run_cmd(
                [
                    py,
                    str(repo_root / "step3_train_and_backtest.py"),
                    "--dataset-dir",
                    str(step3_out / "dataset"),
                    "--out-dir",
                    str(step3_out),
                    "--start-capital",
                    str(args.start_capital),
                    "--train-lookback-days",
                    "365",
                    "--embargo-days",
                    "2",
                    "--min-train-events",
                    "60",
                    "--min-val-events",
                    "12",
                    "--min-test-events",
                    "3",
                    "--retune-every-folds",
                    "1",
                    "--policy-profile",
                    "growth",
                    "--max-aggressive-size",
                    "1.2",
                    "--portfolio-allocator",
                    "equal_split",
                    "--portfolio-objective",
                    "end_equity",
                    "--portfolio-train-ratio",
                    "0.5",
                    "--portfolio-lookback-days",
                    "180",
                    "--portfolio-min-train-days",
                    "90",
                    "--portfolio-no-spy-guard",
                    "--exclude-leaky-features",
                    "--display-symbol-1",
                    "BTC",
                    "--display-symbol-2",
                    "ETH",
                ],
                cwd=repo_root,
                heartbeat_seconds=args.heartbeat_seconds,
            )

        if not args.skip_step3_btceth_tune:
            run_cmd(
                [
                    py,
                    str(pair_root / "scripts" / "optimize_step3_btc_eth.py"),
                    "--dataset-dir",
                    str(step3_out / "dataset"),
                    "--step3-out-dir",
                    str(step3_out),
                    "--start-capital",
                    str(args.start_capital),
                    "--heartbeat-seconds",
                    str(args.heartbeat_seconds),
                ],
                cwd=repo_root,
                heartbeat_seconds=args.heartbeat_seconds,
            )
        if not args.skip_step3_hybrid:
            run_cmd(
                [
                    py,
                    str(pair_root / "scripts" / "optimize_step3_hybrid_btc_eth.py"),
                    "--step3-out-dir",
                    str(step3_out),
                    "--alias-dir",
                    str(clean_alias_dir),
                    "--start-capital",
                    str(args.start_capital),
                    "--core-fractions",
                    str(args.hybrid_core_fractions),
                    "--core-btc-shares",
                    str(args.hybrid_core_btc_shares),
                    "--active-scales",
                    str(args.hybrid_active_scales),
                    "--core-modes",
                    str(args.hybrid_core_modes),
                    "--min-stress125-avg-monthly-pnl",
                    str(args.hybrid_min_stress125_avg_monthly_pnl),
                    "--min-stress150-avg-monthly-pnl",
                    str(args.hybrid_min_stress150_avg_monthly_pnl),
                    "--min-bootstrap-p10-monthly-pnl",
                    str(args.hybrid_min_bootstrap_p10_monthly_pnl),
                    "--max-drawdown-cap",
                    str(args.hybrid_max_drawdown_cap),
                    "--min-active-sleeve-share",
                    str(args.hybrid_min_active_sleeve_share),
                    "--promote-avg-monthly-pnl-margin",
                    str(args.hybrid_promote_avg_monthly_pnl_margin),
                    "--promote-calmar-tolerance",
                    str(args.hybrid_promote_calmar_tolerance),
                    "--promote-drawdown-tolerance",
                    str(args.hybrid_promote_drawdown_tolerance),
                ]
                + (["--no-promote"] if args.hybrid_no_promote else []),
                cwd=repo_root,
                heartbeat_seconds=args.heartbeat_seconds,
            )

    if (
        pattern_enabled
        and pattern_ready
        and (step3_out / "optimization" / "step3_best_config.json").exists()
    ):
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
            str(final_dir / "dual ml"),
            "--start-capital",
            str(args.start_capital),
        ]
        if args.pattern_reuse_existing_runs:
            pattern_cmd.append("--reuse-existing-runs")
        run_cmd(pattern_cmd, cwd=repo_root, heartbeat_seconds=args.heartbeat_seconds)
    elif pattern_enabled:
        log("skipping pattern experiment (best config unavailable).")

    if not args.skip_final:
        final_cmd = [
            py,
            str(repo_root / "run_step2_step3_final.py"),
            "--step2-out-dir",
            os.path.relpath(step2_out, repo_root),
            "--step3-out-dir",
            os.path.relpath(step3_out, repo_root),
            "--final-dir",
            os.path.relpath(final_dir, repo_root),
            "--start-capital",
            str(args.start_capital),
            "--pair-tag",
            "btc_eth",
            "--display-symbol-1",
            "BTC",
            "--display-symbol-2",
            "ETH",
            "--skip-step2",
            "--skip-step3",
        ]
        if (
            pattern_enabled
            and pattern_ready
            and (step3_out / "optimization" / "step3_best_config.json").exists()
        ):
            final_cmd += [
                "--run-pattern-experiment",
                "--pattern-reuse-existing-runs",
            ]
        run_cmd(final_cmd, cwd=repo_root, heartbeat_seconds=args.heartbeat_seconds)
        rewrite_btc_eth_step3_chart(final_dir=final_dir, step3_out=step3_out, start_capital=args.start_capital)
        remove_legacy_named_outputs(final_dir)
        prune_verbose_final_outputs(final_dir)
        write_compact_final_report(final_dir, step3_out=step3_out, start_capital=args.start_capital)
        write_final_readme(final_dir)

    log("complete")
    log(f"workspace: {pair_root}")
    log(f"final output: {final_dir}")


if __name__ == "__main__":
    main()
