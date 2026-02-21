#!/usr/bin/env python3
"""
Robust IBKR historical downloader for BTC + ETH (1h bars).

Key behavior:
- Uses `reqHeadTimeStamp` to bound requests to real available history.
- Prints heartbeat/progress/ETA during sleeps and retries.
- Saves incrementally after each successful chunk.
- Supports extending backward from existing parquet/csv files.

Outputs (default):
  other-pair/btc_eth/data/raw/{symbol}_1h_all.parquet
  other-pair/btc_eth/data/raw/{symbol}_1h_all.csv
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import Crypto, IB, util

try:
    from ib_insync import RequestError
except Exception:  # pragma: no cover
    try:
        from ib_insync.wrapper import RequestError
    except Exception:  # pragma: no cover
        RequestError = Exception

UTC = ZoneInfo("UTC")
APP = "BTC-ETH-DL"
MONTH_SEC = 30 * 24 * 60 * 60


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parquet_path(out_dir: str, symbol: str, use_rth: bool) -> str:
    tag = "rth" if use_rth else "all"
    return os.path.join(out_dir, f"{symbol.lower()}_1h_{tag}.parquet")


def csv_path(out_dir: str, symbol: str, use_rth: bool) -> str:
    tag = "rth" if use_rth else "all"
    return os.path.join(out_dir, f"{symbol.lower()}_1h_{tag}.csv")


def load_existing(out_dir: str, symbol: str, use_rth: bool) -> pd.DataFrame:
    p = parquet_path(out_dir, symbol, use_rth)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def save_data(out_dir: str, symbol: str, use_rth: bool, df: pd.DataFrame) -> None:
    x = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    x.to_parquet(parquet_path(out_dir, symbol, use_rth), index=False)
    x.to_csv(csv_path(out_dir, symbol, use_rth), index=False)


def earliest_timestamp(df: pd.DataFrame) -> datetime | None:
    if df is None or df.empty:
        return None
    ts = pd.to_datetime(df["date"].min(), utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def latest_timestamp(df: pd.DataFrame) -> datetime | None:
    if df is None or df.empty:
        return None
    ts = pd.to_datetime(df["date"].max(), utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def is_pacing_error(err: Exception) -> bool:
    msg = str(err).lower()
    if "pacing" in msg:
        return True
    code = getattr(err, "code", None)
    return code in {162, 366} and "pacing" in msg


def parse_head_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        dt = None
        for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d-%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
            if pd.isna(dt):
                raise ValueError(f"Could not parse head timestamp: {value!r}")
            dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def estimate_chunks(start_utc: datetime, end_utc: datetime, duration_str: str) -> int:
    span = max(0.0, (end_utc - start_utc).total_seconds())
    dur = duration_str.strip().upper()
    if dur.endswith("M"):
        step = MONTH_SEC * float(dur[:-1].strip() or "1")
    elif dur.endswith("W"):
        step = 7 * 24 * 3600 * float(dur[:-1].strip() or "1")
    elif dur.endswith("D"):
        step = 24 * 3600 * float(dur[:-1].strip() or "1")
    else:
        step = MONTH_SEC
    return max(1, int(math.ceil(span / max(1.0, step))))


def sleep_with_heartbeat(
    ib: IB,
    seconds: float,
    heartbeat_seconds: float,
    reason: str,
    stats: Dict[str, Any],
) -> None:
    if seconds <= 0:
        return
    end_t = time.monotonic() + float(seconds)
    next_hb = time.monotonic()
    while True:
        now = time.monotonic()
        rem = end_t - now
        if rem <= 0:
            break
        step = min(1.0, rem)
        ib.sleep(step)
        if now >= next_hb:
            elapsed = now - stats["started_monotonic"]
            done = stats["requests_ok"] + stats["requests_empty"]
            avg_req = elapsed / max(1, stats["requests_total"])
            eta = max(0.0, stats["requests_remaining_est"]) * avg_req
            log(
                f"heartbeat reason={reason} wait_remaining={fmt_duration(rem)} "
                f"elapsed={fmt_duration(elapsed)} req_total={stats['requests_total']} "
                f"req_ok={stats['requests_ok']} req_empty={stats['requests_empty']} "
                f"req_fail={stats['requests_fail']} progress_done={done} "
                f"progress_est_remaining={stats['requests_remaining_est']} "
                f"eta~{fmt_duration(eta)}"
            )
            next_hb = now + max(1.0, heartbeat_seconds)


def fetch_chunk(
    ib: IB,
    contract: Any,
    end_dt_utc: datetime,
    duration_str: str,
    bar_size: str,
    what_to_show: str,
    use_rth: bool,
    timeout_s: int,
) -> pd.DataFrame:
    if end_dt_utc.tzinfo is None:
        end_dt_utc = end_dt_utc.replace(tzinfo=UTC)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt_utc,
        durationStr=duration_str,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
        timeout=timeout_s,
    )
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def connect_with_fallback(
    ib: IB,
    host: str,
    primary_port: int,
    fallback_ports: List[int],
    client_id: int,
    client_id_tries: int,
    timeout_s: int,
    readonly: bool,
) -> tuple[int, int]:
    ports = [int(primary_port)] + [int(p) for p in fallback_ports if int(p) != int(primary_port)]
    last_err: Exception | None = None
    for p in ports:
        for i in range(max(1, int(client_id_tries))):
            cid = int(client_id) + i
            try:
                if ib.isConnected():
                    ib.disconnect()
                log(f"connect try host={host} port={p} client_id={cid} timeout={timeout_s}s")
                ib.connect(host, p, clientId=cid, timeout=timeout_s, readonly=readonly)
                if not ib.isConnected():
                    raise RuntimeError("connect() returned but isConnected() is false")
                return p, cid
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                log(f"connect failed port={p} client_id={cid}: {e}")
                if ("already in use" in msg) or ("client id is already in use" in msg):
                    continue
                if ("timed out" in msg) or isinstance(e, TimeoutError):
                    continue
                break
    if last_err is not None:
        raise last_err
    raise RuntimeError("unable to connect to IBKR")


def ib_await(ib: IB, awaitable: Any, timeout_s: float, label: str) -> Any:
    try:
        return ib.run(asyncio.wait_for(awaitable, timeout=float(timeout_s)))
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"{label} timed out after {timeout_s}s") from e


def parse_exchange_list(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        value = part.strip().upper()
        if value:
            out.append(value)
    return out


def request_head_timestamp_with_fallback(
    ib: IB,
    symbol: str,
    contract: Any,
    use_rth: bool,
    preferred_what_to_show: str,
    startup_timeout_s: float,
) -> tuple[datetime, str]:
    tried: List[str] = []
    candidates = [str(preferred_what_to_show).strip().upper() or "AGGTRADES"]
    if "AGGTRADES" not in candidates:
        candidates.append("AGGTRADES")
    if "TRADES" not in candidates:
        candidates.append("TRADES")

    last_err: Exception | None = None
    for what in candidates:
        tried.append(what)
        try:
            log(f"{symbol} requesting head timestamp what_to_show={what}...")
            head_raw = ib_await(
                ib,
                ib.reqHeadTimeStampAsync(contract, whatToShow=what, useRTH=use_rth, formatDate=2),
                timeout_s=startup_timeout_s,
                label=f"{symbol} reqHeadTimeStamp[{what}]",
            )
            if head_raw in (None, "", [], ()):
                raise ValueError(f"empty head timestamp response for what_to_show={what}")
            return parse_head_timestamp(head_raw), what
        except Exception as e:
            last_err = e
            log(f"{symbol} head timestamp failed what_to_show={what}: {e}")

    if last_err is not None:
        raise RuntimeError(
            f"{symbol} could not fetch head timestamp after trying what_to_show={tried}"
        ) from last_err
    raise RuntimeError(f"{symbol} could not fetch head timestamp")


def qualify_crypto_contract(
    ib: IB,
    symbol: str,
    exchanges: List[str],
    startup_timeout_s: float,
) -> Any:
    last_err: Exception | None = None
    for exch in exchanges:
        c = Crypto(symbol, exch, "USD")
        try:
            log(f"{symbol} qualifying crypto contract exchange={exch}...")
            q = ib_await(
                ib,
                ib.qualifyContractsAsync(c),
                timeout_s=startup_timeout_s,
                label=f"{symbol} qualifyContracts[{exch}]",
            )
            if q:
                qc = q[0]
                log(
                    f"{symbol} qualified conId={getattr(qc, 'conId', None)} "
                    f"exchange={getattr(qc, 'exchange', None)} primaryExch={getattr(qc, 'primaryExchange', None)}"
                )
                return qc
        except Exception as e:
            last_err = e
            log(f"{symbol} qualify failed exchange={exch}: {e}")
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"{symbol} could not be qualified on exchanges={exchanges}")


def build_symbol_state(
    ib: IB,
    symbol: str,
    contract: Any,
    existing: pd.DataFrame,
    now_utc: datetime,
    target_start_utc: datetime,
    use_rth: bool,
    preferred_what_to_show: str,
    duration_str: str,
    startup_timeout_s: float,
) -> Dict[str, Any]:
    head_utc, resolved_what_to_show = request_head_timestamp_with_fallback(
        ib=ib,
        symbol=symbol,
        contract=contract,
        use_rth=use_rth,
        preferred_what_to_show=preferred_what_to_show,
        startup_timeout_s=startup_timeout_s,
    )
    bounded_start_utc = max(target_start_utc, head_utc)
    existing_earliest = earliest_timestamp(existing)
    end_ptr = existing_earliest if existing_earliest is not None else now_utc
    est_chunks = estimate_chunks(bounded_start_utc, end_ptr, duration_str=duration_str)
    done = end_ptr <= bounded_start_utc
    return {
        "symbol": symbol,
        "contract": contract,
        "what_to_show": resolved_what_to_show,
        "data": existing,
        "head_utc": head_utc,
        "target_start_utc": bounded_start_utc,
        "end_ptr": end_ptr,
        "done": done,
        "consecutive_empty": 0,
        "chunks_done_est": 0,
        "chunks_est_total": est_chunks,
        "existing_earliest": existing_earliest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4001, help="4001 live, 4002 paper")
    ap.add_argument("--fallback-ports", nargs="*", type=int, default=[4002])
    ap.add_argument("--client-id", type=int, default=171)
    ap.add_argument("--client-id-tries", type=int, default=10)
    ap.add_argument("--connect-timeout-seconds", type=int, default=20)
    ap.add_argument("--readonly", action="store_true", default=True)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--years-back", type=int, default=6)
    ap.add_argument("--exchanges", default="PAXOS,ZEROHASH,SMART")
    ap.add_argument("--bar-size", default="1 hour")
    ap.add_argument("--what-to-show", default="AGGTRADES")
    ap.add_argument("--use-rth", action="store_true", default=False, help="Use regular-trading-hours only.")
    ap.add_argument("--duration-str", default="1 M")
    ap.add_argument("--sleep-seconds", type=float, default=8.0)
    ap.add_argument("--backoff-on-error", type=float, default=30.0)
    ap.add_argument("--backoff-on-pacing", type=float, default=120.0)
    ap.add_argument("--timeout-seconds", type=int, default=60)
    ap.add_argument("--heartbeat-seconds", type=float, default=20.0)
    ap.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    ap.add_argument("--max-retries-per-chunk", type=int, default=4)
    ap.add_argument("--max-consecutive-empty-chunks", type=int, default=2)
    ap.add_argument("--out-dir", default=str((Path(__file__).resolve().parents[1] / "data" / "raw")))
    args = ap.parse_args()

    symbols = [s.upper() for s in args.symbols]
    exchanges = parse_exchange_list(args.exchanges)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    now_utc = datetime.now(tz=UTC)
    target_start_utc = now_utc - timedelta(days=365 * args.years_back)

    stats: Dict[str, Any] = {
        "started_monotonic": time.monotonic(),
        "requests_total": 0,
        "requests_ok": 0,
        "requests_empty": 0,
        "requests_fail": 0,
        "requests_remaining_est": 0,
    }

    log("start")
    log(
        "config "
        f"symbols={symbols} years_back={args.years_back} bar_size={args.bar_size} "
        f"duration={args.duration_str} use_rth={args.use_rth} timeout_s={args.timeout_seconds} "
        f"exchanges={exchanges}"
    )
    log(f"target_start_utc={target_start_utc.isoformat()} out_dir={out_dir}")

    ib = IB()
    used_port, used_client_id = connect_with_fallback(
        ib=ib,
        host=args.host,
        primary_port=args.port,
        fallback_ports=args.fallback_ports,
        client_id=args.client_id,
        client_id_tries=args.client_id_tries,
        timeout_s=args.connect_timeout_seconds,
        readonly=args.readonly,
    )
    log(f"connected={ib.isConnected()} port={used_port} client_id={used_client_id}")
    try:
        log("requesting server time...")
        server_time = ib_await(
            ib,
            ib.reqCurrentTimeAsync(),
            timeout_s=args.startup_timeout_seconds,
            label="reqCurrentTime",
        )
        log(f"server_time={server_time}")
    except Exception as e:
        log(f"warning: reqCurrentTime failed ({e}); continuing")

    contracts: Dict[str, Any] = {}
    for sym in symbols:
        contracts[sym] = qualify_crypto_contract(
            ib=ib,
            symbol=sym,
            exchanges=exchanges,
            startup_timeout_s=args.startup_timeout_seconds,
        )

    states: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        existing = load_existing(out_dir, sym, args.use_rth)
        st = build_symbol_state(
            ib=ib,
            symbol=sym,
            contract=contracts[sym],
            existing=existing,
            now_utc=now_utc,
            target_start_utc=target_start_utc,
            use_rth=args.use_rth,
            preferred_what_to_show=args.what_to_show,
            duration_str=args.duration_str,
            startup_timeout_s=args.startup_timeout_seconds,
        )
        states[sym] = st
        log(
            f"{sym} head_utc={st['head_utc'].isoformat()} "
            f"effective_start_utc={st['target_start_utc'].isoformat()} "
            f"existing_rows={len(existing)} existing_range={earliest_timestamp(existing)} -> {latest_timestamp(existing)} "
            f"est_chunks={st['chunks_est_total']} already_done={st['done']}"
        )

    stats["requests_remaining_est"] = int(sum(st["chunks_est_total"] for st in states.values() if not st["done"]))

    try:
        while True:
            active = [s for s in symbols if not states[s]["done"]]
            if not active:
                break

            for sym in active:
                st = states[sym]
                this_end = st["end_ptr"]
                target = st["target_start_utc"]
                if this_end <= target:
                    st["done"] = True
                    log(f"{sym} done: reached effective_start_utc")
                    continue

                elapsed = time.monotonic() - stats["started_monotonic"]
                avg_req = elapsed / max(1, stats["requests_total"])
                eta = avg_req * max(0, stats["requests_remaining_est"])
                progress = 100.0 * st["chunks_done_est"] / max(1, st["chunks_est_total"])
                log(
                    f"{sym} request end={this_end.isoformat()} progress={progress:.1f}% "
                    f"symbol_chunks={st['chunks_done_est']}/{st['chunks_est_total']} "
                    f"global_req_total={stats['requests_total']} eta~{fmt_duration(eta)}"
                )

                success = False
                for attempt in range(1, args.max_retries_per_chunk + 1):
                    stats["requests_total"] += 1
                    t0 = time.monotonic()
                    try:
                        chunk = fetch_chunk(
                            ib=ib,
                            contract=st["contract"],
                            end_dt_utc=this_end,
                            duration_str=args.duration_str,
                            bar_size=args.bar_size,
                            what_to_show=st["what_to_show"],
                            use_rth=args.use_rth,
                            timeout_s=args.timeout_seconds,
                        )
                        dt = time.monotonic() - t0
                        if chunk.empty:
                            stats["requests_empty"] += 1
                            st["consecutive_empty"] += 1
                            st["chunks_done_est"] += 1
                            stats["requests_remaining_est"] = max(0, stats["requests_remaining_est"] - 1)
                            log(
                                f"{sym} empty chunk in {dt:.1f}s attempt={attempt}/{args.max_retries_per_chunk} "
                                f"consecutive_empty={st['consecutive_empty']}"
                            )
                            if st["consecutive_empty"] >= args.max_consecutive_empty_chunks:
                                st["done"] = True
                                success = True
                                log(f"{sym} done: max consecutive empty chunks reached")
                                break
                            sleep_with_heartbeat(
                                ib=ib,
                                seconds=args.backoff_on_error,
                                heartbeat_seconds=args.heartbeat_seconds,
                                reason=f"{sym}_empty_backoff",
                                stats=stats,
                            )
                            continue

                        st["consecutive_empty"] = 0
                        combined = pd.concat([st["data"], chunk], ignore_index=True)
                        combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
                        st["data"] = combined
                        save_data(out_dir, sym, args.use_rth, combined)

                        new_earliest = earliest_timestamp(combined)
                        if new_earliest is None:
                            raise RuntimeError("unexpected empty timestamp after merge")
                        if new_earliest >= this_end - timedelta(minutes=1):
                            st["end_ptr"] = this_end - timedelta(days=28)
                        else:
                            st["end_ptr"] = new_earliest - timedelta(seconds=1)

                        stats["requests_ok"] += 1
                        st["chunks_done_est"] += 1
                        stats["requests_remaining_est"] = max(0, stats["requests_remaining_est"] - 1)
                        log(
                            f"{sym} ok in {dt:.1f}s rows={len(combined)} "
                            f"range={earliest_timestamp(combined)} -> {latest_timestamp(combined)}"
                        )
                        success = True
                        break

                    except RequestError as e:
                        stats["requests_fail"] += 1
                        wait_s = args.backoff_on_pacing if is_pacing_error(e) else args.backoff_on_error
                        code = getattr(e, "code", None)
                        msg = getattr(e, "message", str(e))
                        log(
                            f"{sym} RequestError code={code} attempt={attempt}/{args.max_retries_per_chunk} "
                            f"wait={wait_s}s msg={msg}"
                        )
                        sleep_with_heartbeat(
                            ib=ib,
                            seconds=wait_s,
                            heartbeat_seconds=args.heartbeat_seconds,
                            reason=f"{sym}_request_error_backoff",
                            stats=stats,
                        )
                    except Exception as e:
                        stats["requests_fail"] += 1
                        wait_s = args.backoff_on_pacing if is_pacing_error(e) else args.backoff_on_error
                        log(
                            f"{sym} error attempt={attempt}/{args.max_retries_per_chunk} "
                            f"wait={wait_s}s error={e}"
                        )
                        sleep_with_heartbeat(
                            ib=ib,
                            seconds=wait_s,
                            heartbeat_seconds=args.heartbeat_seconds,
                            reason=f"{sym}_error_backoff",
                            stats=stats,
                        )

                if not success and not st["done"]:
                    log(f"{sym} failed chunk after retries, moving on")

                sleep_with_heartbeat(
                    ib=ib,
                    seconds=args.sleep_seconds,
                    heartbeat_seconds=args.heartbeat_seconds,
                    reason=f"{sym}_pace_sleep",
                    stats=stats,
                )
    finally:
        if ib.isConnected():
            ib.disconnect()

    elapsed = time.monotonic() - stats["started_monotonic"]
    log(
        "complete "
        f"elapsed={fmt_duration(elapsed)} req_total={stats['requests_total']} "
        f"req_ok={stats['requests_ok']} req_empty={stats['requests_empty']} req_fail={stats['requests_fail']}"
    )
    for sym in symbols:
        df = states[sym]["data"]
        log(f"{sym} rows={len(df)} range={earliest_timestamp(df)} -> {latest_timestamp(df)}")
        log(f"{sym} parquet={parquet_path(out_dir, sym, args.use_rth)}")
        log(f"{sym} csv={csv_path(out_dir, sym, args.use_rth)}")


if __name__ == "__main__":
    main()
