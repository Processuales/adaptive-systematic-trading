#!/usr/bin/env python3
"""
Run multi-scenario Step 2b comparisons, then save best candidates.

Outputs:
  - <out-dir>/comparison_report.json
  - <out-dir>/best_candidate_non_ml.json
  - <out-dir>/best_candidate_ml.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


SCRIPT_VERSION = "1.0.0"


@dataclass
class Scenario:
    name: str
    symbols: List[str]
    trade_symbol: str
    cross_symbol: str
    include_cross: bool


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def round_obj(obj, ndigits: int = 6):
    if isinstance(obj, dict):
        return {k: round_obj(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_obj(v, ndigits) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return round(obj, ndigits)
    return obj


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="qqq_cross_off",
            symbols=["SPY", "QQQ"],
            trade_symbol="QQQ",
            cross_symbol="SPY",
            include_cross=False,
        ),
        Scenario(
            name="qqq_cross_on",
            symbols=["SPY", "QQQ"],
            trade_symbol="QQQ",
            cross_symbol="SPY",
            include_cross=True,
        ),
        Scenario(
            name="spy_cross_off",
            symbols=["SPY", "QQQ"],
            trade_symbol="SPY",
            cross_symbol="QQQ",
            include_cross=False,
        ),
        Scenario(
            name="spy_cross_on",
            symbols=["SPY", "QQQ"],
            trade_symbol="SPY",
            cross_symbol="QQQ",
            include_cross=True,
        ),
    ]


def run_step2b_for_scenario(
    repo_root: str,
    data_dir: str,
    out_dir: str,
    sc: Scenario,
    n_trials: int,
    seed: int,
    min_trades_train: int,
    min_trades_test: int,
    filters: str,
    friction_profile: str,
    market_hours: str,
) -> None:
    step2b_path = os.path.join(repo_root, "step2b_knob_sweep_backtest.py")
    sc_out = os.path.join(out_dir, sc.name)
    ensure_dir(sc_out)

    cmd = [
        sys.executable,
        step2b_path,
        "--data-dir",
        data_dir,
        "--out-dir",
        sc_out,
        "--symbols",
        *sc.symbols,
        "--trade-symbol",
        sc.trade_symbol,
        "--cross-symbol",
        sc.cross_symbol,
        "--n-trials",
        str(n_trials),
        "--seed",
        str(seed),
        "--min-trades-train",
        str(min_trades_train),
        "--min-trades-test",
        str(min_trades_test),
        "--friction-profile",
        friction_profile,
        "--market-hours",
        market_hours,
    ]
    if not sc.include_cross:
        cmd.append("--no-cross")
    if filters.strip():
        cmd.extend(["--filters", filters])

    print(f"[COMPARE] Running scenario={sc.name} ...")
    subprocess.run(cmd, check=True, cwd=repo_root)


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_candidate(
    scenario: Scenario,
    summary_path: str,
    trial: Dict,
    mode: str,
    rank_metric: str,
) -> Dict:
    test = trial[mode]["test"]
    cagr = test.get("cagr")
    dd = test.get("max_drawdown")
    calmar = None
    if cagr is not None and dd not in (None, 0):
        calmar = cagr / dd

    return {
        "mode": mode,
        "selected_by": rank_metric,
        "scenario": {
            "name": scenario.name,
            "symbols": scenario.symbols,
            "trade_symbol": scenario.trade_symbol,
            "cross_symbol": scenario.cross_symbol,
            "include_cross": scenario.include_cross,
        },
        "summary_path": summary_path,
        "trial": trial,
        "test_snapshot": {
            "n": test.get("n"),
            "end_equity": test.get("end_equity"),
            "cagr": cagr,
            "max_drawdown": dd,
            "calmar": calmar,
            "net_bps_mean": test.get("net_bps_mean"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to cleaned bars directory")
    ap.add_argument("--out-dir", required=True, help="Directory for comparison outputs")
    ap.add_argument("--n-trials", type=int, default=24, help="Step2b trials per scenario")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min-trades-train", type=int, default=80)
    ap.add_argument("--min-trades-test", type=int, default=25)
    ap.add_argument(
        "--filters",
        type=str,
        default="",
        help="Optional comma-separated filters for Step2b; default uses built-in filter set",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running Step2b and only aggregate existing scenario summaries",
    )
    ap.add_argument(
        "--friction-profile",
        choices=["equity", "crypto"],
        default="equity",
        help="Friction profile for Step2b sweeps.",
    )
    ap.add_argument(
        "--market-hours",
        choices=["rth", "24_7"],
        default="rth",
        help="Market-hours mode for Step2b sweeps.",
    )
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    scenarios = default_scenarios()
    scenario_summaries: List[Dict] = []
    no_ml_candidates: List[Dict] = []
    ml_candidates: List[Dict] = []

    for sc in scenarios:
        sc_out = os.path.join(out_dir, sc.name)
        ensure_dir(sc_out)
        summary_path = os.path.join(sc_out, "step2b_summary.json")

        if not args.skip_run:
            run_step2b_for_scenario(
                repo_root=repo_root,
                data_dir=data_dir,
                out_dir=out_dir,
                sc=sc,
                n_trials=args.n_trials,
                seed=args.seed,
                min_trades_train=args.min_trades_train,
                min_trades_test=args.min_trades_test,
                filters=args.filters,
                friction_profile=args.friction_profile,
                market_hours=args.market_hours,
            )

        if not os.path.exists(summary_path):
            print(f"[COMPARE] Missing summary for scenario={sc.name}, skipping.")
            continue

        s = load_summary(summary_path)
        top_no = s.get("top_trials_no_ml", [])
        top_ml = s.get("top_trials_ml", [])
        if not top_no:
            print(f"[COMPARE] No non-ML trials in scenario={sc.name}, skipping.")
            continue

        best_no = top_no[0]
        no_ml_candidates.append(
            make_candidate(
                scenario=sc,
                summary_path=summary_path,
                trial=best_no,
                mode="no_ml",
                rank_metric="end_equity",
            )
        )

        if top_ml:
            best_ml = top_ml[0]
            ml_candidates.append(
                make_candidate(
                    scenario=sc,
                    summary_path=summary_path,
                    trial=best_ml,
                    mode="ml_sim",
                    rank_metric="end_equity",
                )
            )

        scenario_summaries.append(
            {
                "scenario": sc.name,
                "include_cross": sc.include_cross,
                "trade_symbol": sc.trade_symbol,
                "cross_symbol": sc.cross_symbol,
                "meta": s.get("meta", {}),
                "best_no_ml_test": best_no["no_ml"]["test"],
                "best_no_ml_filter": best_no.get("filter_name"),
                "best_ml_test": (top_ml[0]["ml_sim"]["test"] if top_ml else None),
                "best_ml_filter": (top_ml[0].get("filter_name") if top_ml else None),
            }
        )

    if not no_ml_candidates:
        raise RuntimeError("No valid non-ML candidates found across scenarios.")

    best_non_ml = max(
        no_ml_candidates,
        key=lambda x: float(x["test_snapshot"]["end_equity"]),
    )

    best_ml: Optional[Dict] = None
    if ml_candidates:
        best_ml = max(
            ml_candidates,
            key=lambda x: float(x["test_snapshot"]["end_equity"]),
        )

    # Calmar-oriented picks for risk/return users.
    best_non_ml_calmar = max(
        no_ml_candidates,
        key=lambda x: float(x["test_snapshot"]["calmar"] or -1e18),
    )
    best_ml_calmar = (
        max(ml_candidates, key=lambda x: float(x["test_snapshot"]["calmar"] or -1e18))
        if ml_candidates
        else None
    )

    report = {
        "meta": {
            "script": "step2_compare_and_select.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_scenarios": len(scenario_summaries),
            "n_trials_per_scenario": args.n_trials,
            "seed": args.seed,
            "filters_override": args.filters if args.filters.strip() else None,
            "friction_profile": args.friction_profile,
            "market_hours": args.market_hours,
        },
        "scenario_summaries": scenario_summaries,
        "best_candidates": {
            "non_ml_by_end_equity": best_non_ml,
            "non_ml_by_calmar": best_non_ml_calmar,
            "ml_by_end_equity": best_ml,
            "ml_by_calmar": best_ml_calmar,
        },
    }
    report = round_obj(report, ndigits=6)

    report_path = os.path.join(out_dir, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, separators=(",", ":"), ensure_ascii=True)

    best_no_path = os.path.join(out_dir, "best_candidate_non_ml.json")
    with open(best_no_path, "w", encoding="utf-8") as f:
        json.dump(round_obj(best_non_ml, 6), f, separators=(",", ":"), ensure_ascii=True)

    if best_ml is not None:
        best_ml_path = os.path.join(out_dir, "best_candidate_ml.json")
        with open(best_ml_path, "w", encoding="utf-8") as f:
            json.dump(round_obj(best_ml, 6), f, separators=(",", ":"), ensure_ascii=True)
    else:
        best_ml_path = None

    print(f"[COMPARE] Wrote: {report_path}")
    print(f"[COMPARE] Wrote: {best_no_path}")
    if best_ml_path:
        print(f"[COMPARE] Wrote: {best_ml_path}")
    else:
        print("[COMPARE] No valid ML candidate found.")


if __name__ == "__main__":
    main()
