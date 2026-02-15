#!/usr/bin/env python3
"""
Step 2.5: analysis report + compact JSON summary for Step 2 events.

Outputs:
  - <out-dir>/figures/*.png
  - <out-dir>/step2_5_summary.json   (compact, AI-friendly)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "1.1.0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


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


def load_events(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("events path must be .parquet or .csv")

    required = {
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "gross_logret",
        "cost_rt",
        "net_logret",
        "y",
        "family",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Events file missing required columns: {missing}")

    for c in ["decision_time_utc", "entry_time_utc", "exit_time_utc"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.dropna(subset=["decision_time_utc", "entry_time_utc", "exit_time_utc"]).copy()
    df = df.sort_values("decision_time_utc").reset_index(drop=True)

    # Stable helper columns
    df["year"] = df["decision_time_utc"].dt.year
    df["net_bps"] = df["net_logret"] * 10000.0
    df["gross_bps"] = df["gross_logret"] * 10000.0
    df["cost_bps"] = df["cost_rt"] * 10000.0
    return df


def summarize_slice(df: pd.DataFrame, name: str, notes: str = "") -> Dict:
    if len(df) == 0:
        return {
            "scenario": name,
            "notes": notes,
            "n": 0,
            "y_rate": None,
            "net_bps_mean": None,
            "net_bps_med": None,
            "cost_bps_mean": None,
            "net_cum_logret": None,
        }
    return {
        "scenario": name,
        "notes": notes,
        "n": int(len(df)),
        "y_rate": float(df["y"].mean()),
        "net_bps_mean": float(df["net_bps"].mean()),
        "net_bps_med": float(df["net_bps"].median()),
        "cost_bps_mean": float(df["cost_bps"].mean()),
        "net_cum_logret": float(df["net_logret"].sum()),
    }


def add_trend_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trend_decile"] = np.nan
    valid = out["trend_score"].notna()
    if int(valid.sum()) < 100:
        return out
    try:
        out.loc[valid, "trend_decile"] = pd.qcut(
            out.loc[valid, "trend_score"],
            10,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return out
    return out


def filter_scenario_table(df: pd.DataFrame) -> List[Dict]:
    x = add_trend_deciles(df)

    scenarios: List[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("baseline_all", "No filter", lambda d: pd.Series(True, index=d.index)),
        ("no_overnight", "entry_overnight == 0", lambda d: d.get("entry_overnight", 1) == 0),
        (
            "no_overnight_sigma_50_85",
            "entry_overnight == 0 and 0.50 <= sigma_prank <= 0.85",
            lambda d: (d.get("entry_overnight", 1) == 0) & d["sigma_prank"].between(0.50, 0.85),
        ),
        (
            "no_overnight_sigma_45_80",
            "entry_overnight == 0 and 0.45 <= sigma_prank <= 0.80",
            lambda d: (d.get("entry_overnight", 1) == 0) & d["sigma_prank"].between(0.45, 0.80),
        ),
        (
            "no_overnight_sigma_55_90",
            "entry_overnight == 0 and 0.55 <= sigma_prank <= 0.90",
            lambda d: (d.get("entry_overnight", 1) == 0) & d["sigma_prank"].between(0.55, 0.90),
        ),
        (
            "no_overnight_sigma_50_85_trend_decile_4_6",
            "entry_overnight == 0 and 0.50 <= sigma_prank <= 0.85 and trend decile in [4,6]",
            lambda d: (d.get("entry_overnight", 1) == 0)
            & d["sigma_prank"].between(0.50, 0.85)
            & d["trend_decile"].between(4, 6),
        ),
        (
            "no_overnight_sigma_50_85_trend_0p3_1p0",
            "entry_overnight == 0 and 0.50 <= sigma_prank <= 0.85 and 0.3 <= trend_score <= 1.0",
            lambda d: (d.get("entry_overnight", 1) == 0)
            & d["sigma_prank"].between(0.50, 0.85)
            & d["trend_score"].between(0.3, 1.0),
        ),
        (
            "no_overnight_sigma_50_85_trend_decile_4_6_tp8",
            "entry_overnight == 0 and 0.50 <= sigma_prank <= 0.85 and trend decile in [4,6] and tp_to_cost >= 8",
            lambda d: (d.get("entry_overnight", 1) == 0)
            & d["sigma_prank"].between(0.50, 0.85)
            & d["trend_decile"].between(4, 6)
            & (d["tp_to_cost"] >= 8.0),
        ),
        (
            "no_overnight_sigma_50_85_trend_decile_4_6_tp10",
            "entry_overnight == 0 and 0.50 <= sigma_prank <= 0.85 and trend decile in [4,6] and tp_to_cost >= 10",
            lambda d: (d.get("entry_overnight", 1) == 0)
            & d["sigma_prank"].between(0.50, 0.85)
            & d["trend_decile"].between(4, 6)
            & (d["tp_to_cost"] >= 10.0),
        ),
    ]

    rows: List[Dict] = []
    baseline_n = len(x)
    for name, notes, fn in scenarios:
        mask = fn(x).fillna(False)
        s = x[mask].copy()
        row = summarize_slice(s, name=name, notes=notes)
        row["keep_rate"] = float(len(s) / baseline_n) if baseline_n > 0 else None
        rows.append(row)

    # score: prefer positive net with enough sample size
    for row in rows:
        if row["n"] and row["n"] > 0 and row["net_bps_mean"] is not None:
            row["score"] = float(row["net_bps_mean"] * np.sqrt(row["n"]))
        else:
            row["score"] = None
    rows = sorted(
        rows,
        key=lambda r: (
            -1e18 if r["score"] is None else r["score"],
            -1e18 if r["net_bps_mean"] is None else r["net_bps_mean"],
        ),
        reverse=True,
    )
    return rows


def decile_frame(df: pd.DataFrame, col: str, min_rows: int = 30) -> List[Dict]:
    s = df[col]
    valid = df[s.notna()].copy()
    if len(valid) < min_rows:
        return []

    try:
        valid["decile"] = pd.qcut(valid[col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    if valid["decile"].nunique() < 5:
        return []

    g = valid.groupby("decile", as_index=False).agg(
        n=("y", "size"),
        y_rate=("y", "mean"),
        net_bps_mean=("net_bps", "mean"),
        net_bps_med=("net_bps", "median"),
    )
    g["decile"] = g["decile"].astype(int)
    return g.to_dict(orient="records")


def top_correlations(df: pd.DataFrame, target: str, limit: int, exclude: set[str]) -> List[Dict]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    cols = [c for c in numeric.columns if c not in exclude and c != target]
    out: List[Dict] = []
    for c in cols:
        pair = numeric[[target, c]].dropna()
        if len(pair) < 200:
            continue
        if pair[target].std() == 0 or pair[c].std() == 0:
            continue
        corr = pair[target].corr(pair[c])
        if pd.notna(corr):
            out.append({"feature": c, "corr": float(corr), "abs_corr": abs(float(corr))})
    out = sorted(out, key=lambda x: x["abs_corr"], reverse=True)[:limit]
    for x in out:
        x.pop("abs_corr", None)
    return out


def plot_equity(df: pd.DataFrame, out_path: str) -> None:
    x = df["decision_time_utc"]
    gross = df["gross_logret"].cumsum()
    net = df["net_logret"].cumsum()

    plt.figure(figsize=(11, 5))
    plt.plot(x, gross, label="Gross Cumulative Log Return", linewidth=1.5)
    plt.plot(x, net, label="Net Cumulative Log Return", linewidth=1.5)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Step 2 Cumulative Performance")
    plt.xlabel("Decision Time (UTC)")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_return_hist(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(11, 5))
    plt.hist(df["gross_bps"], bins=80, alpha=0.5, label="Gross (bps)")
    plt.hist(df["net_bps"], bins=80, alpha=0.5, label="Net (bps)")
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("Event Return Distribution")
    plt.xlabel("Return (bps)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_yearly(df: pd.DataFrame, out_path: str) -> None:
    y = (
        df.groupby("year", as_index=False)
        .agg(n=("y", "size"), y_rate=("y", "mean"), net_bps_mean=("net_bps", "mean"))
    )

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.bar(y["year"].astype(str), y["net_bps_mean"], alpha=0.7, color="#2A9D8F", label="Mean Net bps")
    ax2.plot(y["year"].astype(str), y["y_rate"], color="#E76F51", marker="o", label="Win Rate")
    ax1.set_ylabel("Mean Net Return (bps)")
    ax2.set_ylabel("Win Rate")
    ax1.set_title("Yearly Net Return + Win Rate")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cost_sensitivity(df: pd.DataFrame, out_path: str) -> List[Dict]:
    mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rows = []
    for m in mults:
        net_m = df["gross_logret"] - (m * df["cost_rt"])
        rows.append(
            {
                "cost_mult": m,
                "net_bps_mean": float(net_m.mean() * 10000.0),
                "win_rate": float((net_m > 0).mean()),
                "cum_logret": float(net_m.sum()),
            }
        )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    xs = [r["cost_mult"] for r in rows]
    ax1.plot(xs, [r["net_bps_mean"] for r in rows], marker="o", color="#264653", label="Mean Net bps")
    ax2.plot(xs, [r["win_rate"] for r in rows], marker="o", color="#D62828", label="Win Rate")
    ax1.set_xlabel("Cost Multiplier")
    ax1.set_ylabel("Mean Net Return (bps)")
    ax2.set_ylabel("Win Rate")
    ax1.set_title("Cost Sensitivity")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return rows


def plot_deciles(df: pd.DataFrame, out_path: str, col_a: str, col_b: str) -> Dict[str, List[Dict]]:
    d_a = decile_frame(df, col_a)
    d_b = decile_frame(df, col_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, title in [
        (axes[0], d_a, f"{col_a} deciles"),
        (axes[1], d_b, f"{col_b} deciles"),
    ]:
        if rows:
            xs = [r["decile"] for r in rows]
            ys = [r["net_bps_mean"] for r in rows]
            ax.plot(xs, ys, marker="o")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Decile")
        ax.set_ylabel("Mean Net bps")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {col_a: d_a, col_b: d_b}


def build_summary(df: pd.DataFrame, bar_features_path: str | None, corr_limit: int) -> Dict:
    weekly_counts = df.set_index("decision_time_utc").resample("W").size()
    family_tab = (
        df.groupby("family", as_index=False)
        .agg(n=("y", "size"), y_rate=("y", "mean"), net_bps_mean=("net_bps", "mean"))
        .sort_values("n", ascending=False)
    )
    overnight_tab = (
        df.groupby("entry_overnight", as_index=False)
        .agg(n=("y", "size"), y_rate=("y", "mean"), net_bps_mean=("net_bps", "mean"))
        if "entry_overnight" in df.columns
        else pd.DataFrame()
    )
    exit_tab = (
        df.groupby("exit_reason", as_index=False)
        .agg(n=("y", "size"), y_rate=("y", "mean"), net_bps_mean=("net_bps", "mean"))
        if "exit_reason" in df.columns
        else pd.DataFrame()
    )
    yearly_tab = (
        df.groupby("year", as_index=False)
        .agg(n=("y", "size"), y_rate=("y", "mean"), net_bps_mean=("net_bps", "mean"), net_bps_sum=("net_bps", "sum"))
    )

    cross_cols = [c for c in df.columns if c.startswith("spy_") or c in {"rs_log", "ret_spread", "beta_proxy", "regime_agree"}]
    cross_nan = [{"feature": c, "nan_rate": float(df[c].isna().mean())} for c in cross_cols]

    nan_rates = (
        df.isna().mean().sort_values(ascending=False).head(25).reset_index().rename(columns={"index": "feature", 0: "nan_rate"})
    )

    exclude = {
        "t_idx",
        "label_end_idx",
        "y",
        "gross_logret",
        "cost_rt",
        "net_logret",
        "net_bps",
        "gross_bps",
        "cost_bps",
    }
    corr_net = top_correlations(df, target="net_logret", limit=corr_limit, exclude=exclude)
    corr_y = top_correlations(df, target="y", limit=corr_limit, exclude=exclude)
    filter_rows = filter_scenario_table(df)
    recommended = None
    for row in filter_rows:
        if row.get("n", 0) >= 250 and row.get("net_bps_mean") is not None:
            recommended = row
            break

    sanity_flags = {
        "cost_dominates_gross_mean": bool(df["cost_bps"].mean() > df["gross_bps"].mean()),
        "net_mean_negative": bool(df["net_bps"].mean() < 0),
        "family_concentration_gt_90pct": bool((family_tab["n"].max() / len(df)) > 0.90) if len(family_tab) > 0 else False,
        "same_bar_ambiguous_gt_1pct": bool(df["same_bar_ambiguous"].mean() > 0.01) if "same_bar_ambiguous" in df.columns else False,
        "weekly_events_gt_40": bool(weekly_counts.mean() > 40.0),
    }

    bar_context = {}
    if bar_features_path:
        try:
            bf = pd.read_parquet(bar_features_path) if bar_features_path.endswith(".parquet") else pd.read_csv(bar_features_path)
            bar_context = {
                "bar_rows": int(len(bf)),
                "events_per_100_bars": float(100.0 * len(df) / max(1, len(bf))),
            }
        except Exception as e:
            bar_context = {"error": str(e)}

    return {
        "meta": {
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_events": int(len(df)),
        },
        "headline": {
            "y_rate": float(df["y"].mean()),
            "mean_gross_bps": float(df["gross_bps"].mean()),
            "mean_cost_bps": float(df["cost_bps"].mean()),
            "mean_net_bps": float(df["net_bps"].mean()),
            "median_tp_to_cost": float(df["tp_to_cost"].median()) if "tp_to_cost" in df.columns else None,
            "weekly_events_mean": float(weekly_counts.mean()),
            "weekly_events_p90": float(weekly_counts.quantile(0.90)),
            "same_bar_ambiguous_rate": float(df["same_bar_ambiguous"].mean()) if "same_bar_ambiguous" in df.columns else None,
            "entry_overnight_rate": float(df["entry_overnight"].mean()) if "entry_overnight" in df.columns else None,
        },
        "slices": {
            "family": family_tab.to_dict(orient="records"),
            "overnight": overnight_tab.to_dict(orient="records"),
            "exit_reason": exit_tab.to_dict(orient="records"),
            "yearly": yearly_tab.to_dict(orient="records"),
        },
        "feature_signal": {
            "top_corr_net_logret": corr_net,
            "top_corr_y": corr_y,
            "trend_score_deciles": decile_frame(df, "trend_score"),
            "tp_to_cost_deciles": decile_frame(df, "tp_to_cost"),
            "sigma_prank_deciles": decile_frame(df, "sigma_prank"),
            "u_atr_prank_deciles": decile_frame(df, "u_atr_prank"),
        },
        "filter_scenarios": filter_rows,
        "recommended_filter": recommended,
        "data_quality": {
            "top_nan_rates": nan_rates.to_dict(orient="records"),
            "cross_nan_rates": cross_nan,
            "sanity_flags": sanity_flags,
            "bar_context": bar_context,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-path", required=True, help="Step 2 events parquet/csv")
    ap.add_argument("--out-dir", required=True, help="Step 2.5 output directory")
    ap.add_argument("--bar-features-path", default=None, help="Optional trade symbol bar_features parquet/csv")
    ap.add_argument("--corr-limit", type=int, default=12, help="Number of top feature correlations to keep")
    ap.add_argument("--pretty-json", action="store_true", help="Write indented JSON (default compact/minified)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(fig_dir)

    df = load_events(args.events_path)
    if len(df) == 0:
        raise ValueError("Events dataset is empty.")

    # Visuals
    paths = {
        "equity": os.path.join(fig_dir, "equity_gross_vs_net.png"),
        "return_hist": os.path.join(fig_dir, "return_hist_bps.png"),
        "yearly": os.path.join(fig_dir, "yearly_net_and_winrate.png"),
        "cost_sens": os.path.join(fig_dir, "cost_sensitivity.png"),
        "deciles": os.path.join(fig_dir, "decile_profiles.png"),
    }
    plot_equity(df, paths["equity"])
    plot_return_hist(df, paths["return_hist"])
    plot_yearly(df, paths["yearly"])
    cost_rows = plot_cost_sensitivity(df, paths["cost_sens"])
    decile_rows = plot_deciles(df, paths["deciles"], "trend_score", "tp_to_cost")

    summary = build_summary(df, args.bar_features_path, corr_limit=max(1, args.corr_limit))
    summary["cost_sensitivity"] = cost_rows
    summary["decile_profiles"] = decile_rows
    summary["figure_paths"] = paths

    summary = round_obj(summary, ndigits=6)
    out_json = os.path.join(args.out_dir, "step2_5_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        if args.pretty_json:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        else:
            json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[STEP2.5] Wrote summary JSON: {out_json}")
    print(f"[STEP2.5] Wrote figures in: {fig_dir}")


if __name__ == "__main__":
    main()
