#!/usr/bin/env python3
"""
Full BTC/ETH pipeline in isolated workspace:
1) Download history from IBKR (optional)
2) Prepare/clean data + alias mapping (QQQ->BTC, SPY->ETH)
3) Step 2 full pipeline
4) Step 3 full pipeline (optimize + optional pattern experiment)
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


def write_compact_final_report(final_dir: Path, start_capital: float) -> None:
    reports_dir = final_dir / "reports"
    charts_dir = final_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "final_output_summary.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing expected report to compact: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        old = json.load(f)

    compact = {
        "meta": {
            "script": "run_btc_eth_pipeline.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "pair_tag": "btc_eth",
            "display_symbols": ["BTC", "ETH"],
            "start_capital": float(start_capital),
            "note": "Compact side-pair summary for BTC+ETH.",
        },
        "paths": {
            "final_root": str(final_dir),
            "charts_dir": str(charts_dir),
            "reports_dir": str(reports_dir),
            "step2_chart": str(charts_dir / "01_step2_ml_simulation_btc_eth.png"),
            "step3_chart": str(charts_dir / "02_step3_real_ml_btc_eth.png"),
        },
        "snapshots": {
            "step2_ml_simulation": (old.get("snapshots") or {}).get("step2_ml_simulation"),
            "step3_real_ml": (old.get("snapshots") or {}).get("step3_real_ml"),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, separators=(",", ":"), ensure_ascii=True)
    log(f"wrote compact report: {report_path}")


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

    log("start")
    log(
        f"workspace={pair_root} start_capital={args.start_capital} "
        f"download_years_back={args.download_years_back}"
    )

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

    if (
        (not args.skip_pattern)
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
    elif not args.skip_pattern:
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
            (not args.skip_pattern)
            and pattern_ready
            and (step3_out / "optimization" / "step3_best_config.json").exists()
        ):
            final_cmd += [
                "--run-pattern-experiment",
                "--pattern-reuse-existing-runs",
            ]
        run_cmd(final_cmd, cwd=repo_root, heartbeat_seconds=args.heartbeat_seconds)
        remove_legacy_named_outputs(final_dir)
        prune_verbose_final_outputs(final_dir)
        write_compact_final_report(final_dir, start_capital=args.start_capital)
        write_final_readme(final_dir)

    log("complete")
    log(f"workspace: {pair_root}")
    log(f"final output: {final_dir}")


if __name__ == "__main__":
    main()
