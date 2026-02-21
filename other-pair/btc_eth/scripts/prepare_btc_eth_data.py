#!/usr/bin/env python3
"""
Clean raw BTC/ETH bars and create alias files for the existing SPY/QQQ pipeline.

Outputs:
  other-pair/btc_eth/data_clean/{btc,eth}_1h_all_clean.parquet
  other-pair/btc_eth/data_clean_alias/{qqq,spy}_1h_rth_clean.parquet
  other-pair/btc_eth/data_clean_alias/pair_alias_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_raw(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return df


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
    x = x.dropna(subset=["date"]).copy()

    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close", "volume"]).copy()

    x = x[(x["open"] > 0) & (x["high"] > 0) & (x["low"] > 0) & (x["close"] > 0)].copy()
    x = x[x["high"] >= x["low"]].copy()
    x = x[x["volume"] >= 0].copy()

    x = x.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return x


def summarize(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"rows": 0, "start_utc": None, "end_utc": None}
    return {
        "rows": int(len(df)),
        "start_utc": str(df["date"].min()),
        "end_utc": str(df["date"].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str((Path(__file__).resolve().parents[1] / "data" / "raw")))
    ap.add_argument("--clean-dir", default=str((Path(__file__).resolve().parents[1] / "data_clean")))
    ap.add_argument("--alias-dir", default=str((Path(__file__).resolve().parents[1] / "data_clean_alias")))
    ap.add_argument("--btc-file", default="btc_1h_all.parquet")
    ap.add_argument("--eth-file", default="eth_1h_all.parquet")
    ap.add_argument("--qqq-alias-source", default="BTC", choices=["BTC", "ETH"])
    args = ap.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    clean_dir = os.path.abspath(args.clean_dir)
    alias_dir = os.path.abspath(args.alias_dir)
    ensure_dir(clean_dir)
    if os.path.exists(alias_dir):
        shutil.rmtree(alias_dir, ignore_errors=True)
    ensure_dir(alias_dir)

    btc_path = os.path.join(raw_dir, args.btc_file)
    eth_path = os.path.join(raw_dir, args.eth_file)
    if not os.path.exists(btc_path):
        raise FileNotFoundError(f"Missing raw file: {btc_path}")
    if not os.path.exists(eth_path):
        raise FileNotFoundError(f"Missing raw file: {eth_path}")

    btc_raw = read_raw(btc_path)
    eth_raw = read_raw(eth_path)
    btc = clean_bars(btc_raw)
    eth = clean_bars(eth_raw)
    if btc.empty or eth.empty:
        raise RuntimeError("One of the cleaned datasets is empty.")

    btc_clean_path = os.path.join(clean_dir, "btc_1h_all_clean.parquet")
    eth_clean_path = os.path.join(clean_dir, "eth_1h_all_clean.parquet")
    btc.to_parquet(btc_clean_path, index=False)
    eth.to_parquet(eth_clean_path, index=False)

    if args.qqq_alias_source.upper() == "BTC":
        qqq_df = btc
        spy_df = eth
        qqq_src = "BTC"
        spy_src = "ETH"
    else:
        qqq_df = eth
        spy_df = btc
        qqq_src = "ETH"
        spy_src = "BTC"

    qqq_alias_path = os.path.join(alias_dir, "qqq_1h_rth_clean.parquet")
    spy_alias_path = os.path.join(alias_dir, "spy_1h_rth_clean.parquet")
    qqq_df.to_parquet(qqq_alias_path, index=False)
    spy_df.to_parquet(spy_alias_path, index=False)

    summary = {
        "meta": {
            "script": "prepare_btc_eth_data.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "QQQ/SPY alias files run the existing pipeline without touching core strategy code.",
        },
        "raw_files": {"btc": btc_path, "eth": eth_path},
        "clean_files": {"btc": btc_clean_path, "eth": eth_clean_path},
        "alias_files": {"qqq": qqq_alias_path, "spy": spy_alias_path},
        "alias_map": {
            "QQQ": qqq_src,
            "SPY": spy_src,
            "pair_actual": ["BTC", "ETH"],
        },
        "stats": {
            "BTC": summarize(btc),
            "ETH": summarize(eth),
        },
    }
    summary_path = os.path.join(alias_dir, "pair_alias_map.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print("[BTC-ETH-PREP] clean saved:", btc_clean_path)
    print("[BTC-ETH-PREP] clean saved:", eth_clean_path)
    print("[BTC-ETH-PREP] alias saved:", qqq_alias_path)
    print("[BTC-ETH-PREP] alias saved:", spy_alias_path)
    print("[BTC-ETH-PREP] map saved:", summary_path)
    print("[BTC-ETH-PREP] alias_map: QQQ->", qqq_src, " SPY->", spy_src)


if __name__ == "__main__":
    main()
