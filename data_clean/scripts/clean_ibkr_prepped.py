#!/usr/bin/env python3
"""
Clean IBKR prepped 1h RTH bars for research.

Goal:
- Keep your original IBKR data unchanged.
- Produce a research-clean dataset by DROPPING a tiny number of session-days
  where the calendar + IBKR 1h alignment expects bars that are missing.

Inputs:
- qa_reports/prepped/{symbol}_1h_rth_prepped.parquet

Outputs:
- <out-dir>/{symbol}_1h_rth_clean.parquet
- <out-dir>/{symbol}_bad_days.csv
- <out-dir>/clean_summary.csv

This is intentionally strict and simple.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from datetime import datetime, timedelta, time

import pandas as pd

NY_TZ = "America/New_York"
UTC_TZ = "UTC"


def _as_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize(UTC_TZ)
    return t


def next_hour_boundary(dt: datetime) -> datetime:
    """Return the next full-hour boundary strictly after dt, in the same tz."""
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    if dt == dt0:
        return dt
    return dt0 + timedelta(hours=1)


def ibkr_expected_1h_starts(open_local: datetime, close_local: datetime) -> List[datetime]:
    """
    IBKR 1h RTH alignment rule we want:
    - first bar starts exactly at session open (09:30)
    - subsequent bars start on the hour (10:00, 11:00, ...)
    - no bar starts at/after close
    """
    out: List[datetime] = []
    if open_local >= close_local:
        return out

    # first bar at open
    out.append(open_local)

    # then at the next full hour boundary after open
    t = next_hour_boundary(open_local)
    while t < close_local:
        out.append(t)
        t = t + timedelta(hours=1)

    return out


def load_prepped(prepped_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(prepped_path)
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {prepped_path}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Basic columns check (not enforcing extras)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column in {prepped_path}")

    return df


def get_calendar_schedule(cal_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Uses pandas_market_calendars if installed.
    Returns schedule indexed by session date with market_open/market_close tz-aware in NY time.
    """
    try:
        import pandas_market_calendars as mcal
    except Exception as e:
        raise RuntimeError(
            "pandas_market_calendars is required for this cleaner.\n"
            "Install: pip install pandas_market_calendars"
        ) from e

    cal = mcal.get_calendar(cal_name)

    # Some versions support tz=...
    try:
        sched = cal.schedule(start_date=start_date, end_date=end_date, tz=NY_TZ)
    except TypeError:
        sched = cal.schedule(start_date=start_date, end_date=end_date)
        # Force to NY
        for col in ["market_open", "market_close"]:
            sched[col] = pd.to_datetime(sched[col])
            if sched[col].dt.tz is None:
                sched[col] = sched[col].dt.tz_localize(UTC_TZ).dt.tz_convert(NY_TZ)
            else:
                sched[col] = sched[col].dt.tz_convert(NY_TZ)

    if sched.empty:
        raise RuntimeError("Calendar schedule is empty. Check date range and calendar name.")
    return sched


@dataclass
class DayDiff:
    session_date: str
    actual_bars: int
    expected_bars: int
    missing_bars: int
    extra_bars: int
    first_actual_utc: Optional[str]
    last_actual_utc: Optional[str]


def per_day_diff(df: pd.DataFrame, sched: pd.DataFrame) -> Tuple[List[DayDiff], List[str]]:
    """
    Compare actual bar-start timestamps to expected IBKR 1h starts for each session day.
    Returns per-day diffs and list of "bad days" (missing>0 or extra>0).
    """
    # Add NY session date
    ny = df["date"].dt.tz_convert(NY_TZ)
    df = df.copy()
    df["session_date"] = ny.dt.date.astype(str)

    by_day = df.groupby("session_date")["date"].apply(list).to_dict()

    diffs: List[DayDiff] = []
    bad_days: List[str] = []

    for idx, row in sched.iterrows():
        # idx is session date as Timestamp (usually)
        d = str(pd.Timestamp(idx).date())
        open_ny = row["market_open"].to_pydatetime()
        close_ny = row["market_close"].to_pydatetime()

        exp_local = ibkr_expected_1h_starts(open_ny, close_ny)
        exp_utc = [pd.Timestamp(t).tz_convert(UTC_TZ) for t in exp_local]
        exp_set = set(exp_utc)

        actual_utc = [pd.Timestamp(t) for t in by_day.get(d, [])]
        actual_set = set(actual_utc)

        missing = exp_set - actual_set
        extra = actual_set - exp_set

        actual_sorted = sorted(actual_utc)
        first_actual = actual_sorted[0].isoformat() if actual_sorted else None
        last_actual = actual_sorted[-1].isoformat() if actual_sorted else None

        dd = DayDiff(
            session_date=d,
            actual_bars=len(actual_utc),
            expected_bars=len(exp_utc),
            missing_bars=len(missing),
            extra_bars=len(extra),
            first_actual_utc=first_actual,
            last_actual_utc=last_actual,
        )
        diffs.append(dd)

        if dd.missing_bars > 0 or dd.extra_bars > 0:
            bad_days.append(d)

    return diffs, bad_days


def detect_suspicious_overnight_gaps(df: pd.DataFrame, sched: pd.DataFrame, threshold_minutes: int = 90) -> List[str]:
    """
    Flags a session_date (the *later* day) if the actual gap between
    previous day's last bar-start and today's first bar-start differs
    from expected by more than threshold_minutes.
    """
    # Prepare actual first/last per day
    ny = df["date"].dt.tz_convert(NY_TZ)
    tmp = df.copy()
    tmp["session_date"] = ny.dt.date.astype(str)

    g = tmp.groupby("session_date")["date"]
    first_map = g.min().to_dict()
    last_map = g.max().to_dict()

    # Build expected last bar start from schedule (close minus 1h, but using our expected generator)
    sched_days = []
    for idx, row in sched.iterrows():
        d = str(pd.Timestamp(idx).date())
        open_ny = row["market_open"].to_pydatetime()
        close_ny = row["market_close"].to_pydatetime()
        exp = ibkr_expected_1h_starts(open_ny, close_ny)
        if not exp:
            continue
        last_start_ny = exp[-1]
        sched_days.append((d, pd.Timestamp(last_start_ny).tz_convert(UTC_TZ), row["market_open"].tz_convert(UTC_TZ)))

    # Compare adjacent schedule days
    suspicious: List[str] = []
    for i in range(1, len(sched_days)):
        prev_d, prev_last_start_exp_utc, _prev_open_utc = sched_days[i - 1]
        cur_d, _cur_last_start_exp_utc, cur_open_utc = sched_days[i]

        if prev_d not in last_map or cur_d not in first_map:
            continue

        prev_last_actual = pd.Timestamp(last_map[prev_d])
        cur_first_actual = pd.Timestamp(first_map[cur_d])

        expected_gap = cur_open_utc - prev_last_start_exp_utc
        actual_gap = cur_first_actual - prev_last_actual

        diff = abs(actual_gap - expected_gap)
        if diff > pd.Timedelta(minutes=threshold_minutes):
            suspicious.append(cur_d)

    return sorted(set(suspicious))


def clean_symbol(root: Path, out_dir: Path, symbol: str, calendar: str) -> Dict[str, object]:
    prepped = root / "prepped" / f"{symbol.lower()}_1h_rth_prepped.parquet"
    if not prepped.exists():
        raise FileNotFoundError(f"Missing prepped parquet: {prepped}")

    df = load_prepped(prepped)

    start_date = str(df["date"].dt.tz_convert(NY_TZ).dt.date.min())
    end_date = str(df["date"].dt.tz_convert(NY_TZ).dt.date.max())

    sched = get_calendar_schedule(calendar, start_date, end_date)

    diffs, bad_days_sched = per_day_diff(df, sched)
    suspicious_gap_days = detect_suspicious_overnight_gaps(df, sched, threshold_minutes=90)

    bad_days = sorted(set(bad_days_sched) | set(suspicious_gap_days))

    # Apply drop
    ny = df["date"].dt.tz_convert(NY_TZ)
    df = df.copy()
    df["session_date"] = ny.dt.date.astype(str)

    before_rows = len(df)
    if bad_days:
        df_clean = df[~df["session_date"].isin(bad_days)].copy()
    else:
        df_clean = df.copy()

    df_clean = df_clean.drop(columns=["session_date"]).reset_index(drop=True)
    after_rows = len(df_clean)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{symbol.lower()}_1h_rth_clean.parquet"
    df_clean.to_parquet(out_parquet, index=False)

    bad_days_csv = out_dir / f"{symbol.lower()}_bad_days.csv"
    pd.DataFrame({"session_date": bad_days}).to_csv(bad_days_csv, index=False)

    # Also save the per-day diff (small, useful)
    per_day_csv = out_dir / f"{symbol.lower()}_per_day_diff.csv"
    pd.DataFrame([d.__dict__ for d in diffs]).to_csv(per_day_csv, index=False)

    return {
        "symbol": symbol,
        "input_rows": before_rows,
        "output_rows": after_rows,
        "dropped_rows": before_rows - after_rows,
        "bad_days_count": len(bad_days),
        "bad_days_sched_count": len(bad_days_sched),
        "bad_days_gap_count": len(suspicious_gap_days),
        "output_parquet": str(out_parquet),
        "bad_days_csv": str(bad_days_csv),
        "per_day_csv": str(per_day_csv),
        "start_utc": str(df["date"].min()),
        "end_utc": str(df["date"].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="qa_reports directory")
    ap.add_argument("--out-dir", type=str, default="data_clean", help="output dir for clean data")
    ap.add_argument("--symbols", nargs="+", required=True, help="e.g. SPY QQQ")
    ap.add_argument("--calendar", type=str, default="NYSE", help="e.g. NYSE")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    rows = []
    for sym in args.symbols:
        res = clean_symbol(root, out_dir, sym.upper(), args.calendar)
        rows.append(res)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "clean_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("CLEAN COMPLETE")
    print(summary.to_string(index=False))
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
