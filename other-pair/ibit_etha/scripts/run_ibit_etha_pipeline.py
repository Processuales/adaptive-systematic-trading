#!/usr/bin/env python3
"""
Full IBIT/ETHA pipeline in isolated workspace:
1) Download history from IBKR (optional)
2) Prepare/clean data + alias mapping (QQQ->IBIT, SPY->ETHA)
3) Step 2 full pipeline
4) Step 3 full pipeline (optimize + pattern ML experiment)
5) Final charts/reports under this pair folder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[IBIT-ETHA-PIPE] $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def remove_legacy_named_outputs(final_dir: Path) -> None:
    legacy_files = [
        final_dir / "charts" / "01_step2_ml_simulation_spy_qqq.png",
        final_dir / "charts" / "02_step3_real_ml_spy_qqq.png",
    ]
    for p in legacy_files:
        if p.exists():
            p.unlink()
            print(f"[IBIT-ETHA-PIPE] removed legacy chart name: {p}")


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def parse_trade_constraints(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        if ":" not in t:
            continue
        a, b = t.split(":", 1)
        out.append((int(a.strip()), int(b.strip())))
    return out


def run_step2_compare_with_fallback(
    py: str,
    repo_root: Path,
    data_dir: Path,
    selection_dir: Path,
    n_trials: int,
    seed: int,
    trade_constraints_fallback: list[tuple[int, int]],
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
            run_cmd(cmd, cwd=repo_root)
            return True
        except subprocess.CalledProcessError:
            print(
                "[IBIT-ETHA-PIPE] step2_compare failed with "
                f"min-trades-train={mt_train} min-trades-test={mt_test}; retrying..."
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
) -> bool:
    last_err: Exception | None = None
    for obj in objectives:
        for mt in fallback_min_test_trades:
            for min_bps in candidate_min_net_bps_fallback:
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
                        obj,
                        "--train-ratio",
                        str(train_ratio),
                        "--weight-step",
                        str(weight_step),
                        "--min-test-trades",
                        str(mt),
                        "--candidate-min-net-bps",
                        str(min_bps),
                        "--candidate-min-cagr",
                        str(min_cagr),
                    ]
                    try:
                        run_cmd(cmd, cwd=repo_root)
                        return True
                    except subprocess.CalledProcessError as e:
                        last_err = e
                        print(
                            "[IBIT-ETHA-PIPE] dual mode="
                            f"{mode} failed with objective={obj} min-test-trades={mt} "
                            f"candidate-min-net-bps={min_bps} candidate-min-cagr={min_cagr}; retrying..."
                        )
    if last_err is not None:
        return False
    return False


def write_ml_dual_fallback(selection_dir: Path) -> None:
    src = selection_dir / "dual_portfolio"
    dst = selection_dir / "dual_portfolio_ml"
    if not src.exists():
        raise RuntimeError(f"Cannot create ML fallback: source does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    summary_path = dst / "dual_symbol_portfolio_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["ml_dual_fallback_used"] = True
    summary["ml_dual_fallback_reason"] = (
        "No eligible ML candidates met minimum-trade constraints under current search budget."
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-step2", action="store_true")
    ap.add_argument("--skip-step3", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--skip-pattern", action="store_true")
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--step2-n-trials", type=int, default=36)
    ap.add_argument("--step2-seed", type=int, default=42)
    ap.add_argument(
        "--step2-trade-constraint-fallback",
        default="80:25,60:18,40:12,25:8,15:5",
        help="Comma-separated train:test fallback list for Step2 compare (e.g. 80:25,40:12,15:5).",
    )
    ap.add_argument(
        "--dual-min-test-trades-fallback",
        default="15,5,0",
        help="Comma-separated fallback list for step2_dual_symbol_portfolio_test --min-test-trades.",
    )
    ap.add_argument(
        "--dual-objective-fallback",
        default="calmar,end_equity",
        help="Comma-separated fallback objective list for dual portfolio selection.",
    )
    ap.add_argument(
        "--dual-candidate-min-net-bps-fallback",
        default="0,-1000000000",
        help="Comma-separated fallback list for --candidate-min-net-bps in dual selector.",
    )
    ap.add_argument(
        "--dual-candidate-min-cagr-fallback",
        default="-0.1,-1",
        help="Comma-separated fallback list for --candidate-min-cagr in dual selector.",
    )
    ap.add_argument("--step3-max-candidates", type=int, default=4)
    ap.add_argument("--ib-host", default="127.0.0.1")
    ap.add_argument("--ib-port", type=int, default=4001)
    ap.add_argument("--ib-client-id", type=int, default=71)
    ap.add_argument("--download-years-back", type=int, default=4)
    ap.add_argument("--pattern-reuse-existing-runs", action="store_true")
    args = ap.parse_args()
    trade_constraints_fallback = parse_trade_constraints(args.step2_trade_constraint_fallback)
    if not trade_constraints_fallback:
        trade_constraints_fallback = [(80, 25), (60, 18), (40, 12), (25, 8), (15, 5)]
    dual_mt_fallback = parse_int_list(args.dual_min_test_trades_fallback)
    if not dual_mt_fallback:
        dual_mt_fallback = [15, 5, 0]
    dual_objectives = [x.strip() for x in str(args.dual_objective_fallback).split(",") if x.strip()]
    if not dual_objectives:
        dual_objectives = ["calmar", "end_equity"]
    dual_min_bps_fallback = parse_float_list(args.dual_candidate_min_net_bps_fallback)
    if not dual_min_bps_fallback:
        dual_min_bps_fallback = [0.0, -1_000_000_000.0]
    dual_min_cagr_fallback = parse_float_list(args.dual_candidate_min_cagr_fallback)
    if not dual_min_cagr_fallback:
        dual_min_cagr_fallback = [-0.1, -1.0]

    script_path = Path(__file__).resolve()
    pair_root = script_path.parents[1]  # .../other-pair/ibit_etha
    repo_root = script_path.parents[3]  # .../adaptive-systematic-trading
    py = sys.executable

    raw_dir = pair_root / "data" / "raw"
    clean_alias_dir = pair_root / "data_clean_alias"
    step2_out = pair_root / "step2_out"
    step3_out = pair_root / "step3_out"
    final_dir = pair_root / "final output"
    pattern_ready = True

    if not args.skip_download:
        run_cmd(
            [
                py,
                str(pair_root / "scripts" / "download_history_ibit_etha.py"),
                "--host",
                args.ib_host,
                "--port",
                str(args.ib_port),
                "--client-id",
                str(args.ib_client_id),
                "--years-back",
                str(args.download_years_back),
                "--out-dir",
                str(raw_dir),
            ],
            cwd=repo_root,
        )

    if not args.skip_prepare:
        run_cmd(
            [
                py,
                str(pair_root / "scripts" / "prepare_ibit_etha_data.py"),
                "--raw-dir",
                str(raw_dir),
                "--clean-dir",
                str(pair_root / "data_clean"),
                "--alias-dir",
                str(clean_alias_dir),
                "--qqq-alias-source",
                "IBIT",
            ],
            cwd=repo_root,
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
        )
        ok_compare = run_step2_compare_with_fallback(
            py=py,
            repo_root=repo_root,
            data_dir=clean_alias_dir,
            selection_dir=selection_dir,
            n_trials=args.step2_n_trials,
            seed=args.step2_seed,
            trade_constraints_fallback=trade_constraints_fallback,
        )
        if not ok_compare:
            raise RuntimeError(
                "Step2 compare/select failed for all configured trade-constraint fallbacks."
            )
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
        )
        if not ok_no_ml:
            raise RuntimeError(
                "Step2 dual no-ML portfolio failed for all min-test-trades fallback values."
            )
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
        )
        if not ok_ml:
            print(
                "[IBIT-ETHA-PIPE] ML dual portfolio failed; using no-ML dual portfolio as ML fallback."
            )
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
            )
        except subprocess.CalledProcessError:
            print(
                "[IBIT-ETHA-PIPE] step3_optimize_model failed; using short-history fallback "
                "step3_train_and_backtest profile."
            )
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
        run_cmd(pattern_cmd, cwd=repo_root)
    elif not args.skip_pattern:
        print("[IBIT-ETHA-PIPE] skipping pattern experiment (best config unavailable).")

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
            "ibit_etha",
            "--display-symbol-1",
            "IBIT",
            "--display-symbol-2",
            "ETHA",
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
        run_cmd(final_cmd, cwd=repo_root)
        remove_legacy_named_outputs(final_dir)

    print("[IBIT-ETHA-PIPE] complete")
    print("[IBIT-ETHA-PIPE] workspace:", pair_root)
    print("[IBIT-ETHA-PIPE] final output:", final_dir)


if __name__ == "__main__":
    main()
