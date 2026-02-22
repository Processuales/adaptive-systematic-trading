#!/usr/bin/env python3
"""
Build a combined portfolio chart for:
- SPY+QQQ Step 3 sleeve
- BTC+ETH Step 3 sleeve

The allocator dynamically re-weights sleeves using trailing monthly stats.
Outputs are written under final output/combined by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month_end", "equity", "pnl", "ret", "trades"])
    x = df.copy()
    x["month_end"] = pd.to_datetime(x["month_end"], utc=True, errors="coerce")
    x = x.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    for c in ("equity", "pnl", "ret", "trades"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        else:
            x[c] = 0.0
    return x


def max_drawdown_from_returns(ret: pd.Series, start_capital: float = 1.0) -> float:
    if ret.empty:
        return 0.0
    eq = float(start_capital) * (1.0 + ret.fillna(0.0).astype(float)).cumprod()
    roll = eq.cummax()
    dd = eq / np.maximum(roll, 1e-12) - 1.0
    return abs(float(dd.min())) if len(dd) else 0.0


def perf_from_returns(ret: pd.Series, start_capital: float) -> Dict[str, float]:
    r = ret.fillna(0.0).astype(float)
    if r.empty:
        return {
            "end_equity": float(start_capital),
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "avg_monthly_pnl": 0.0,
            "median_monthly_pnl": 0.0,
            "positive_month_rate": 0.0,
            "n_months": 0,
        }
    eq = float(start_capital) * (1.0 + r).cumprod()
    pnl = eq.diff().fillna(eq.iloc[0] - float(start_capital))
    n = len(r)
    years = max(n / 12.0, 1.0 / 12.0)
    end_equity = float(eq.iloc[-1])
    ratio = max(end_equity / max(float(start_capital), 1e-9), 1e-9)
    cagr = float(ratio ** (1.0 / years) - 1.0)
    mdd = max_drawdown_from_returns(r, start_capital=1.0)
    calmar = float(cagr / mdd) if mdd > 1e-9 else 0.0
    return {
        "end_equity": end_equity,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": calmar,
        "avg_monthly_pnl": float(pnl.mean()),
        "median_monthly_pnl": float(pnl.median()),
        "positive_month_rate": float((pnl > 0.0).mean()),
        "n_months": int(n),
    }


@dataclass(frozen=True)
class AllocConfig:
    lookback_months: int
    dd_penalty: float
    temp: float
    smoothing: float
    min_weight_spyqqq: float
    max_weight_spyqqq: float


def _rolling_sharpe_like(window_ret: np.ndarray) -> float:
    if window_ret.size < 2:
        return 0.0
    mu = float(np.mean(window_ret))
    sd = float(np.std(window_ret))
    return mu / max(sd, 1e-9)


def build_dynamic_weights(
    ret_spyqqq: pd.Series,
    ret_btceth: pd.Series,
    cfg: AllocConfig,
) -> pd.Series:
    idx = ret_spyqqq.index
    w = np.zeros(len(idx), dtype=float)
    prev = 0.5
    for i in range(len(idx)):
        lo = max(0, i - int(cfg.lookback_months))
        a = ret_spyqqq.iloc[lo:i].to_numpy(dtype=float)
        b = ret_btceth.iloc[lo:i].to_numpy(dtype=float)
        if i < cfg.lookback_months or len(a) < 3 or len(b) < 3:
            raw = 0.5
        else:
            score_a = _rolling_sharpe_like(a) - cfg.dd_penalty * max_drawdown_from_returns(pd.Series(a))
            score_b = _rolling_sharpe_like(b) - cfg.dd_penalty * max_drawdown_from_returns(pd.Series(b))
            z_a = score_a / max(cfg.temp, 1e-6)
            z_b = score_b / max(cfg.temp, 1e-6)
            z_m = max(z_a, z_b)
            e_a = np.exp(z_a - z_m)
            e_b = np.exp(z_b - z_m)
            raw = float(e_a / max(e_a + e_b, 1e-12))

        clipped = float(np.clip(raw, cfg.min_weight_spyqqq, cfg.max_weight_spyqqq))
        cur = float(cfg.smoothing * prev + (1.0 - cfg.smoothing) * clipped)
        cur = float(np.clip(cur, cfg.min_weight_spyqqq, cfg.max_weight_spyqqq))
        w[i] = cur
        prev = cur
    return pd.Series(w, index=idx, dtype=float)


def score_config(
    ret_spyqqq: pd.Series,
    ret_btceth: pd.Series,
    cfg: AllocConfig,
    train_end_idx: int,
    start_capital: float,
    objective: str,
    max_dd_cap: float,
) -> float:
    w = build_dynamic_weights(ret_spyqqq, ret_btceth, cfg)
    r = w * ret_spyqqq + (1.0 - w) * ret_btceth
    r_train = r.iloc[:train_end_idx]
    m = perf_from_returns(r_train, start_capital=start_capital)

    dd = float(m["max_drawdown"])
    dd_over = max(0.0, dd - float(max_dd_cap))
    if objective == "return":
        # Return-first with soft DD cap.
        return float(1.00 * m["end_equity"] + 35.0 * m["avg_monthly_pnl"] - 18000.0 * dd_over)
    if objective == "calmar":
        return float(90.0 * m["calmar"] + 0.03 * m["avg_monthly_pnl"] - 12.0 * dd)
    # balanced
    return float(55.0 * m["calmar"] + 0.06 * m["avg_monthly_pnl"] + 0.12 * m["end_equity"] - 16.0 * dd)


def grid_configs() -> Iterable[AllocConfig]:
    for lookback in (3, 6, 9, 12, 18):
        for dd_penalty in (0.0, 0.4, 0.8, 1.2, 1.6, 2.4):
            for temp in (0.35, 0.5, 0.8, 1.2):
                for smoothing in (0.10, 0.20, 0.30, 0.50, 0.70):
                    for min_w in (0.0, 0.1, 0.15, 0.25):
                        for max_w in (0.75, 0.85, 0.95, 1.0):
                            if min_w >= max_w:
                                continue
                            yield AllocConfig(
                                lookback_months=int(lookback),
                                dd_penalty=float(dd_penalty),
                                temp=float(temp),
                                smoothing=float(smoothing),
                                min_weight_spyqqq=float(min_w),
                                max_weight_spyqqq=float(max_w),
                            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spyqqq-monthly",
        default="combo_workspace/step3_spyqqq/backtest/step3_monthly_table.parquet",
    )
    ap.add_argument(
        "--btceth-monthly",
        default="other-pair/btc_eth/step3_out/backtest/step3_monthly_table.parquet",
    )
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument(
        "--objective",
        choices=["balanced", "return", "calmar"],
        default="balanced",
        help="Allocator tuning objective.",
    )
    ap.add_argument(
        "--max-dd-cap",
        type=float,
        default=0.20,
        help="Soft max drawdown cap used during allocator tuning.",
    )
    ap.add_argument("--out-png", default="final output/combined/charts/SPYQQQ_BTCETH_combined_chart.png")
    ap.add_argument("--out-json", default="final output/combined/reports/SPYQQQ_BTCETH_combined_summary.json")
    ap.add_argument("--out-monthly", default="final output/combined/data/SPYQQQ_BTCETH_combined_monthly.parquet")
    args = ap.parse_args()

    spy_path = Path(args.spyqqq_monthly).resolve()
    btc_path = Path(args.btceth_monthly).resolve()
    if not spy_path.exists():
        raise FileNotFoundError(f"Missing SPY+QQQ monthly table: {spy_path}")
    if not btc_path.exists():
        raise FileNotFoundError(f"Missing BTC+ETH monthly table: {btc_path}")

    spy_m = normalize_monthly(pd.read_parquet(spy_path))
    btc_m = normalize_monthly(pd.read_parquet(btc_path))
    if spy_m.empty or btc_m.empty:
        raise RuntimeError("One input monthly table is empty.")

    spy_ret = pd.Series(spy_m["ret"].to_numpy(dtype=float), index=spy_m["month_end"], dtype=float)
    btc_ret = pd.Series(btc_m["ret"].to_numpy(dtype=float), index=btc_m["month_end"], dtype=float)
    idx = pd.DatetimeIndex(sorted(spy_ret.index.intersection(btc_ret.index)))
    if len(idx) < 36:
        raise RuntimeError(f"Insufficient overlap months: {len(idx)}")

    spy_ret = spy_ret.reindex(idx).fillna(0.0)
    btc_ret = btc_ret.reindex(idx).fillna(0.0)
    split_i = int(max(24, min(len(idx) - 12, round(float(args.train_ratio) * len(idx)))))

    best_cfg = None
    best_score = -1e18
    top_rows = []
    for cfg in grid_configs():
        s = score_config(
            ret_spyqqq=spy_ret,
            ret_btceth=btc_ret,
            cfg=cfg,
            train_end_idx=split_i,
            start_capital=float(args.start_capital),
            objective=str(args.objective),
            max_dd_cap=float(args.max_dd_cap),
        )
        top_rows.append((float(s), cfg))
        if s > best_score:
            best_score = s
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError("Allocator tuning failed to select a config.")
    top_rows = sorted(top_rows, key=lambda x: x[0], reverse=True)[:8]

    w_spy = build_dynamic_weights(spy_ret, btc_ret, best_cfg)
    w_btc = 1.0 - w_spy
    ret_dyn = w_spy * spy_ret + w_btc * btc_ret
    ret_static = 0.5 * spy_ret + 0.5 * btc_ret

    eq_spy = float(args.start_capital) * (1.0 + spy_ret).cumprod()
    eq_btc = float(args.start_capital) * (1.0 + btc_ret).cumprod()
    eq_dyn = float(args.start_capital) * (1.0 + ret_dyn).cumprod()
    eq_static = float(args.start_capital) * (1.0 + ret_static).cumprod()

    m_spy = perf_from_returns(spy_ret, start_capital=float(args.start_capital))
    m_btc = perf_from_returns(btc_ret, start_capital=float(args.start_capital))
    m_dyn = perf_from_returns(ret_dyn, start_capital=float(args.start_capital))
    m_static = perf_from_returns(ret_static, start_capital=float(args.start_capital))

    monthly_out = pd.DataFrame(
        {
            "month_end": idx,
            "spyqqq_ret": spy_ret.to_numpy(dtype=float),
            "btceth_ret": btc_ret.to_numpy(dtype=float),
            "combined_ret_dynamic": ret_dyn.to_numpy(dtype=float),
            "combined_ret_static_50_50": ret_static.to_numpy(dtype=float),
            "spyqqq_weight_dynamic": w_spy.to_numpy(dtype=float),
            "btceth_weight_dynamic": w_btc.to_numpy(dtype=float),
            "equity_dynamic": eq_dyn.to_numpy(dtype=float),
            "equity_static_50_50": eq_static.to_numpy(dtype=float),
        }
    )

    out_monthly = Path(args.out_monthly).resolve()
    out_png = Path(args.out_png).resolve()
    out_json = Path(args.out_json).resolve()
    out_monthly.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    monthly_out.to_parquet(out_monthly, index=False)

    corr = float(spy_ret.corr(btc_ret))
    stats_lines = [
        f"Dynamic: End ${m_dyn['end_equity']:,.0f} | CAGR {100*m_dyn['cagr']:.2f}% | DD {100*m_dyn['max_drawdown']:.2f}% | Calmar {m_dyn['calmar']:.2f}",
        f"Static 50/50: End ${m_static['end_equity']:,.0f} | CAGR {100*m_static['cagr']:.2f}% | DD {100*m_static['max_drawdown']:.2f}% | Calmar {m_static['calmar']:.2f}",
        f"Sleeves: SPY+QQQ End ${m_spy['end_equity']:,.0f} | BTC+ETH End ${m_btc['end_equity']:,.0f} | Corr {corr:.3f}",
        (
            f"Objective={args.objective} (soft DD cap {100*float(args.max_dd_cap):.1f}%) | Best cfg: "
            f"lookback={best_cfg.lookback_months}, dd_penalty={best_cfg.dd_penalty}, "
            f"temp={best_cfg.temp}, smoothing={best_cfg.smoothing}, "
            f"w_spyqqq=[{best_cfg.min_weight_spyqqq:.2f},{best_cfg.max_weight_spyqqq:.2f}]"
        ),
        f"Train months: {split_i} / {len(idx)} | Train ratio: {args.train_ratio:.2f}",
    ]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 3.0, 1.3, 1.2])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    ax0.axis("off")
    ax0.text(
        0.01,
        0.98,
        "\n".join(stats_lines),
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.90, "edgecolor": "#999999"},
    )

    ax1.plot(idx, eq_dyn.values, color="#1d3557", linewidth=2.3, label="Combined dynamic allocator")
    ax1.plot(idx, eq_static.values, color="#457b9d", linewidth=1.4, alpha=0.85, label="Combined static 50/50")
    ax1.plot(idx, eq_spy.values, color="#2a9d8f", linewidth=1.1, alpha=0.75, label="SPY+QQQ sleeve only")
    ax1.plot(idx, eq_btc.values, color="#e76f51", linewidth=1.1, alpha=0.75, label="BTC+ETH sleeve only")
    ax1.axhline(float(args.start_capital), color="black", linewidth=0.8, linestyle="--", alpha=0.65)
    ax1.set_title("Combined Portfolio: SPY+QQQ + BTC+ETH (Dynamic Sleeve Allocation)")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper left", ncol=2, framealpha=0.92)

    ax2.plot(idx, w_spy.values, color="#264653", linewidth=1.7, label="SPY+QQQ weight")
    ax2.plot(idx, w_btc.values, color="#f4a261", linewidth=1.7, label="BTC+ETH weight")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Weight Share")
    ax2.set_title("Dynamic Sleeve Weights")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", framealpha=0.92)

    pnl_dyn = eq_dyn.diff().fillna(eq_dyn.iloc[0] - float(args.start_capital)).to_numpy(dtype=float)
    colors = np.where(pnl_dyn >= 0.0, "#2a9d8f", "#d62828")
    ax3.bar(idx, pnl_dyn, width=20, color=colors, alpha=0.86)
    ax3.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax3.set_title("Dynamic Combined Monthly PnL")
    ax3.set_ylabel("PnL ($)")
    ax3.set_xlabel("Time (UTC)")
    ax3.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    top_cfg_rows = [{"train_objective_score": s, "config": asdict(c)} for (s, c) in top_rows]
    out = {
        "meta": {
            "script": "build_spyqqq_btceth_combined_chart.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "start_capital": float(args.start_capital),
            "objective": str(args.objective),
            "max_dd_cap": float(args.max_dd_cap),
            "spyqqq_monthly_source": str(spy_path),
            "btceth_monthly_source": str(btc_path),
            "overlap_start_utc": str(idx.min()),
            "overlap_end_utc": str(idx.max()),
            "n_overlap_months": int(len(idx)),
            "train_end_idx": int(split_i),
            "train_ratio": float(args.train_ratio),
        },
        "best_allocator_config": asdict(best_cfg),
        "top_allocator_candidates_train": top_cfg_rows,
        "cross_pair_monthly_correlation": corr,
        "metrics": {
            "combined_dynamic": m_dyn,
            "combined_static_50_50": m_static,
            "spyqqq_sleeve": m_spy,
            "btceth_sleeve": m_btc,
        },
        "outputs": {
            "combined_chart_png": str(out_png),
            "combined_summary_json": str(out_json),
            "combined_monthly_parquet": str(out_monthly),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[COMBINED] wrote chart: {out_png}", flush=True)
    print(f"[COMBINED] wrote summary: {out_json}", flush=True)
    print(f"[COMBINED] wrote monthly: {out_monthly}", flush=True)


if __name__ == "__main__":
    main()
