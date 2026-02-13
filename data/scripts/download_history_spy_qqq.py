import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Stock, util

# Optional import for richer error handling across ib_insync versions
try:
    from ib_insync import RequestError  # newer/exports in some versions
except Exception:  # pragma: no cover
    try:
        from ib_insync.wrapper import RequestError  # fallback
    except Exception:
        RequestError = Exception  # last resort fallback


# -----------------------
# Configuration
# -----------------------
HOST = "127.0.0.1"
PORT = 4001          # 4001 live, 4002 paper (match your IB Gateway)
CLIENT_ID = 7        # choose any integer not used by other IB clients

SYMBOLS = ["SPY", "QQQ"]

# Download settings
BAR_SIZE = "1 hour"
WHAT_TO_SHOW = "TRADES"
USE_RTH = True

# How far back you want to go
YEARS_BACK = 10

# Chunk size per request
DURATION_STR = "1 M"

# Pacing
SLEEP_SECONDS_AFTER_EACH_REQUEST = 12

# Backoff
BACKOFF_SECONDS_ON_ERROR = 60
BACKOFF_SECONDS_ON_PACING = 300

# Request timeout
HIST_TIMEOUT_SECONDS = 120

# Retries
MAX_RETRIES_PER_CHUNK = 6
MAX_CONSECUTIVE_EMPTY_CHUNKS = 3

DATA_DIR = "data"

UTC = ZoneInfo("UTC")


# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parquet_path(symbol: str) -> str:
    tag = "rth" if USE_RTH else "all"
    return os.path.join(DATA_DIR, f"{symbol.lower()}_1h_{tag}.parquet")

def csv_path(symbol: str) -> str:
    tag = "rth" if USE_RTH else "all"
    return os.path.join(DATA_DIR, f"{symbol.lower()}_1h_{tag}.csv")

def load_existing(symbol: str) -> pd.DataFrame:
    p = parquet_path(symbol)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return pd.DataFrame()

def save_data(symbol: str, df: pd.DataFrame) -> None:
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df.to_parquet(parquet_path(symbol), index=False)
    df.to_csv(csv_path(symbol), index=False)

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
    # IB often reports historical pacing issues under these codes, but check message too.
    if code in {162, 366} and "pacing" in msg:
        return True
    return False

def fetch_chunk(ib: IB, contract, end_dt_utc: datetime) -> pd.DataFrame:
    """
    Request one chunk of historical bars ending at end_dt_utc (tz-aware).
    Returns a dataframe with UTC timestamps.
    """
    if end_dt_utc.tzinfo is None:
        end_dt_utc = end_dt_utc.replace(tzinfo=UTC)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt_utc,          # tz-aware datetime, no string formatting
        durationStr=DURATION_STR,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        formatDate=2,                    # intraday bars returned as UTC tz-aware
        keepUpToDate=False,
        timeout=HIST_TIMEOUT_SECONDS,
    )

    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    return df


# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(DATA_DIR)

    now_utc = datetime.now(tz=UTC)
    start_utc = now_utc - timedelta(days=365 * YEARS_BACK)

    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
    print("Connected:", ib.isConnected())
    print("Server time:", ib.reqCurrentTime())
    print()

    # Prepare contracts
    contracts = {}
    for sym in SYMBOLS:
        c = Stock(sym, "SMART", "USD")
        ib.qualifyContracts(c)
        contracts[sym] = c

    # Load existing data
    data = {sym: load_existing(sym) for sym in SYMBOLS}

    for sym in SYMBOLS:
        df = data[sym]
        print(f"{sym}: existing rows =", 0 if df.empty else len(df))
        if not df.empty:
            print(f"{sym}: range =", earliest_timestamp(df), "to", latest_timestamp(df))
        print()

    # Per-symbol end pointer:
    # If existing data, extend backwards from earliest known timestamp.
    # If none, start from now.
    end_ptr = {}
    for sym in SYMBOLS:
        e = earliest_timestamp(data[sym])
        end_ptr[sym] = e if e is not None else now_utc

    print("Downloading 1h bars in 1-month chunks.")
    print("Sleeping after each request to stay under historical pacing limits.")
    print("Target start:", start_utc)
    print()

    done = {sym: False for sym in SYMBOLS}
    consecutive_empty = {sym: 0 for sym in SYMBOLS}

    try:
        while True:
            all_done = True

            for sym in SYMBOLS:
                if done[sym]:
                    continue

                all_done = False
                contract = contracts[sym]
                this_end = end_ptr[sym]

                if this_end <= start_utc:
                    done[sym] = True
                    print(f"{sym}: reached target start date. Done.")
                    continue

                print(f"{sym}: requesting chunk ending {this_end.isoformat()}")

                # Retry loop for this chunk
                success = False
                for attempt in range(1, MAX_RETRIES_PER_CHUNK + 1):
                    try:
                        chunk = fetch_chunk(ib, contract, this_end)

                        if chunk.empty:
                            consecutive_empty[sym] += 1
                            print(
                                f"{sym}: empty chunk (attempt {attempt}/{MAX_RETRIES_PER_CHUNK}). "
                                f"consecutive_empty={consecutive_empty[sym]}"
                            )

                            if consecutive_empty[sym] >= MAX_CONSECUTIVE_EMPTY_CHUNKS:
                                # Usually means we are past available history or wrong settings.
                                print(f"{sym}: too many empty chunks. Marking done.")
                                done[sym] = True
                                success = True
                                break

                            ib.sleep(BACKOFF_SECONDS_ON_ERROR)
                            continue

                        consecutive_empty[sym] = 0

                        combined = pd.concat([data[sym], chunk], ignore_index=True)
                        combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
                        data[sym] = combined
                        save_data(sym, combined)

                        new_earliest = earliest_timestamp(combined)
                        if new_earliest is None:
                            raise RuntimeError("Unexpected missing timestamps after merge")

                        # Step back 1 second to avoid overlapping identical requests
                        end_ptr[sym] = new_earliest - timedelta(seconds=1)

                        print(
                            f"{sym}: saved rows={len(combined)} "
                            f"earliest={new_earliest} latest={latest_timestamp(combined)}"
                        )
                        success = True
                        break

                    except RequestError as e:
                        msg = getattr(e, "message", str(e))
                        code = getattr(e, "code", None)
                        pacing = is_pacing_error(e)
                        wait_s = BACKOFF_SECONDS_ON_PACING if pacing else BACKOFF_SECONDS_ON_ERROR
                        print(
                            f"{sym}: RequestError code={code} msg={msg}. "
                            f"backoff={wait_s}s (attempt {attempt}/{MAX_RETRIES_PER_CHUNK})"
                        )
                        ib.sleep(wait_s)

                    except Exception as e:
                        pacing = is_pacing_error(e)
                        wait_s = BACKOFF_SECONDS_ON_PACING if pacing else BACKOFF_SECONDS_ON_ERROR
                        print(
                            f"{sym}: error: {e}. "
                            f"backoff={wait_s}s (attempt {attempt}/{MAX_RETRIES_PER_CHUNK})"
                        )
                        ib.sleep(wait_s)

                if not success and not done[sym]:
                    print(f"{sym}: failed this chunk after retries. Skipping symbol for now.")
                    ib.sleep(BACKOFF_SECONDS_ON_ERROR)

                # Global pacing sleep after each symbol request cycle
                ib.sleep(SLEEP_SECONDS_AFTER_EACH_REQUEST)

            if all_done:
                break

        print("\nAll symbols complete.")
        for sym in SYMBOLS:
            df = data[sym]
            print(f"{sym}: rows={len(df)} range={earliest_timestamp(df)} to {latest_timestamp(df)}")
            print("Saved:", parquet_path(sym))
            print("Saved:", csv_path(sym))
            print()

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
