#!/usr/bin/env python3
"""
Clean raw IBIT/ETHA bars and create alias files for the existing SPY/QQQ pipeline.

Outputs:
  other-pair/ibit_etha/data_clean/{ibit,etha}_1h_rth_clean.parquet
  other-pair/ibit_etha/data_clean_alias/{qqq,spy}_1h_rth_clean.parquet
  other-pair/ibit_etha/data_clean_alias/pair_alias_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
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
    ap.add_argument("--ibit-file", default="ibit_1h_rth.parquet")
    ap.add_argument("--etha-file", default="etha_1h_rth.parquet")
    ap.add_argument("--qqq-alias-source", default="IBIT", choices=["IBIT", "ETHA"])
    args = ap.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    clean_dir = os.path.abspath(args.clean_dir)
    alias_dir = os.path.abspath(args.alias_dir)
    ensure_dir(clean_dir)
    if os.path.exists(alias_dir):
        shutil.rmtree(alias_dir, ignore_errors=True)
    ensure_dir(alias_dir)

    ibit_path = os.path.join(raw_dir, args.ibit_file)
    etha_path = os.path.join(raw_dir, args.etha_file)
    if not os.path.exists(ibit_path):
        raise FileNotFoundError(f"Missing raw file: {ibit_path}")
    if not os.path.exists(etha_path):
        raise FileNotFoundError(f"Missing raw file: {etha_path}")

    ibit_raw = read_raw(ibit_path)
    etha_raw = read_raw(etha_path)
    ibit = clean_bars(ibit_raw)
    etha = clean_bars(etha_raw)
    if ibit.empty or etha.empty:
        raise RuntimeError("One of the cleaned datasets is empty.")

    ibit_clean_path = os.path.join(clean_dir, "ibit_1h_rth_clean.parquet")
    etha_clean_path = os.path.join(clean_dir, "etha_1h_rth_clean.parquet")
    ibit.to_parquet(ibit_clean_path, index=False)
    etha.to_parquet(etha_clean_path, index=False)

    if args.qqq_alias_source.upper() == "IBIT":
        qqq_df = ibit
        spy_df = etha
        qqq_src = "IBIT"
        spy_src = "ETHA"
    else:
        qqq_df = etha
        spy_df = ibit
        qqq_src = "ETHA"
        spy_src = "IBIT"

    qqq_alias_path = os.path.join(alias_dir, "qqq_1h_rth_clean.parquet")
    spy_alias_path = os.path.join(alias_dir, "spy_1h_rth_clean.parquet")
    qqq_df.to_parquet(qqq_alias_path, index=False)
    spy_df.to_parquet(spy_alias_path, index=False)

    summary = {
        "meta": {
            "script": "prepare_ibit_etha_data.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "QQQ/SPY alias files are used to run existing pipeline without modifying core strategy code.",
        },
        "raw_files": {"ibit": ibit_path, "etha": etha_path},
        "clean_files": {"ibit": ibit_clean_path, "etha": etha_clean_path},
        "alias_files": {"qqq": qqq_alias_path, "spy": spy_alias_path},
        "alias_map": {
            "QQQ": qqq_src,
            "SPY": spy_src,
            "pair_actual": ["IBIT", "ETHA"],
        },
        "stats": {
            "IBIT": summarize(ibit),
            "ETHA": summarize(etha),
        },
    }
    summary_path = os.path.join(alias_dir, "pair_alias_map.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"), ensure_ascii=True)

    print("[IBIT-ETHA-PREP] clean saved:", ibit_clean_path)
    print("[IBIT-ETHA-PREP] clean saved:", etha_clean_path)
    print("[IBIT-ETHA-PREP] alias saved:", qqq_alias_path)
    print("[IBIT-ETHA-PREP] alias saved:", spy_alias_path)
    print("[IBIT-ETHA-PREP] map saved:", summary_path)
    print("[IBIT-ETHA-PREP] alias_map: QQQ->", qqq_src, " SPY->", spy_src)


if __name__ == "__main__":
    main()

