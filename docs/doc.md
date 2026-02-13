1) What parts of IBKR “API” you actually use for this plan

You only need four buckets of functionality:

A. Connectivity and session state

Connect to IB Gateway or TWS (socket API).

Detect disconnects and reconnect cleanly.

Pull server time (sanity check for timezone and clock drift).

B. Instrument definition (avoid symbol ambiguity)

Resolve SPY and QQQ to a specific contract (conId).

Pull trading schedule and timezone for session logic (RTH open/close, holidays). reqContractDetails includes TradingHours, LiquidHours, and timeZoneId.

C. Market data for research and signals

You will use two forms:

Historical bars (primary): reqHistoricalData for 1h bars (and later 4h).

Delayed streaming quotes (optional, sanity only): delayed top of book if you set delayed mode.

Important: IBKR’s own docs state historical data via API requires the same market data subscriptions as live top-of-book. In practice you may still receive some historical bars, but you should design assuming that requirement can bite later.

D. Orders and fills (later, when you go live)

Submit simple orders (market or limit), cancel/replace, read fills.

Pull executions and commissions for friction calibration.
You can keep “Read-Only API” enabled while you build the research pipeline. When you later trade, Read-Only must be off.

2) Data entitlement reality for your current “no pay” setup
Delayed streaming data

IBKR provides delayed streaming data for many instruments if you explicitly switch market data type to delayed (type 3) before requesting market data.

Delayed is typically about 15 to 20 minutes.

In paper trading, US stocks are delayed 15 minutes.

In ib_insync:

ib.reqMarketDataType(3)  # 3 = delayed

Snapshot quotes (paid per request)

If you ever want “current-ish” NBBO snapshots without a monthly subscription, IBKR offers snapshot quotes priced per request. US-listed equities and ETFs are $0.01 per request and there is a $1 monthly waiver; IBKR may auto-upgrade you to streaming for that month if snapshot fees reach the streaming cost.

This is useful for occasional sanity checks, not for systematic intraday trading.

Market data subscriptions (when you do pay)

For US stocks/ETFs, IBKR generally expects Level 1 subscriptions and exchange coverage is usually “Network A/B/C” (NYSE, ARCA/AMEX/BATS/IEX, NASDAQ).

3) Rate limits that matter for your plan

There are two pacing layers you care about:

A. Global API message pacing

IBKR’s current guidance: max API requests per second is your Maximum Market Data Lines divided by 2. With the default 100 market data lines, that implies 50 requests/second.

Also, TWS/IBG can be configured to either reject messages above the max rate or apply pacing (queue them).

Impact on your bot:

You will be nowhere near this if you are trading 2 symbols on hourly bars.

It becomes relevant only if you try to backfill many symbols, many timeframes, or do tick-level stuff.

B. Historical data pacing and limits

IBKR explicitly limits historical data request behavior:

Max simultaneous open historical requests: 50

Pacing violation conditions include:

Identical historical requests within 15 seconds

6+ historical requests for the same contract/exchange/tick type within 2 seconds

More than 60 requests within 10 minutes

They also note soft throttling can occur even when “hard limits” are lifted for bar sizes 1 min and greater.

Impact on your bot:

For SPY + QQQ hourly bars, this is trivial if you design correctly.

The only time you hit limits is initial bulk backfill if you chunk poorly.

Practical backfill approach for your use case:

Request 1 hour bars in “chunks” that return a few thousand bars at most.

IBKR’s own “step size” guidance includes examples like “1 M duration supports 30 mins to 1 month bar sizes”, and “1 D supports 1 min to 1 day”, etc.

For 1h bars, grabbing a month at a time is typically fine.

Real-time bars pacing

If you ever use reqRealTimeBars (5-second bars), it is subject to historical pacing limits (60 queries per 600 seconds).
You probably do not need this for an hourly strategy.

4) Bars, sessions, and “overnight-aware Fix B” using IBKR primitives
Timezone control

Historical bars are returned in a timezone controlled by TWS/Gateway settings (login / API configuration).

For your plan, you want everything aligned to US/Eastern (Toronto is same offset in practice for market hours).

Validate by printing a few bar timestamps and checking they align with 09:30 to 16:00 for RTH.

RTH vs ETH

reqHistoricalData has useRTH:

useRTH=1 returns data generated only during regular trading hours.
This is what you want for clean overnight gap modeling.

Getting the trading session schedule

Use reqContractDetails to pull:

TradingHours / LiquidHours and the timeZoneId

This is the cleanest way to implement your overnight indicator g_t without hard-coding hours and without getting wrecked by holidays or half-days.

Bar timestamp semantics you should assume

For intraday bars, IBKR’s bars are delivered with a timestamp string and the timezone is your TWS timezone setting.

For daily bars, IBKR notes that a “daily bar” can correspond to a session that crosses calendar days, and the bar date corresponds to the day on which the bar closes.

For your 1h/4h work, you will treat bar timestamps as “bar end time” in practice. Your current outputs already look like that (11:00, 12:00, etc).

Data type choice for bars

For equities/ETFs, “TRADES” is the normal choice for your features and labeling. IBKR notes TRADES data is adjusted for splits but not dividends.

That is usually acceptable for SPY/QQQ at hourly horizon.

Volume caveat (matters for your liquidity proxies)

IBKR notes their historical feed is filtered for certain trade types that occur away from NBBO (combos, block trades, some derivatives), and historical volume can be lower than “unfiltered” sources.

For your plan:

Use volume as a relative feature (z-score, percentile), not as an absolute “true volume” reference.

5) “No live data” development mode that still supports your plan

Given you are not paying for live quotes right now, the most robust workflow is:

Research pipeline (fully supported)

Use reqHistoricalData to backfill bars for SPY and QQQ.

Update bars on a schedule (hourly) with one small request per symbol.

Build features, labels (triple barrier), and model training entirely on bars.

This uses almost no API capacity and stays far away from pacing issues.

Paper “shadow book” execution simulation

Since your plan already separates:

Shadow book (many candidates)

Live book (strict gate)

You can simulate “execution at next bar open” by using the next bar’s open from historical bars, exactly as your research spec states.

Optional sanity price checks

If you want occasional “current” checks:

delayed streaming with reqMarketDataType(3)

or snapshots (paid per request)

6) Operational constraints you should design around
Market data lines

IBKR allocates a number of concurrent market data lines. Default is typically 100.
You will not hit this with 2 ETFs unless you accidentally leave subscriptions open for lots of symbols.

Market Data API Acknowledgement

If you ever see “not subscribed” errors in situations that should work, confirm the Market Data API Acknowledgement is enabled. IBKR explicitly calls out that without it, users get market data errors.

One login reality

Market data subscriptions are tied to the username and can be impacted if you are logged in elsewhere.
For automation later, treat your Gateway instance as the single source of truth and do not also log into TWS with the same user on another machine.

7) Minimal API surface area for your bot plan

Here is the smallest “API contract” you should build your code around:

Instrument resolution

qualifyContracts (ib_insync helper)

reqContractDetails (store: conId, primaryExchange, timeZoneId, TradingHours/LiquidHours)

Bars

reqHistoricalData for:

backfill (chunked)

incremental updates

useRTH=1

whatToShow="TRADES"

Optional: keepUpToDate=True if you later want streaming updates of the current unfinished bar.

(Later) Orders and fills for friction calibration

placeOrder, cancelOrder

reqExecutions, execDetails streams

commissionReport

reqPositions, reqAccountSummary

No scanners, no L2, no news, no options, no tick-by-tick.

8) How rate limits map to your actual plan

If you keep it to SPY + QQQ:

Hourly update loop (safe)

Every hour, request the most recent 2 to 5 bars for each symbol.

That is 2 requests per hour.

You will never hit 60 per 10 minutes.

Initial backfill (still safe if chunked)

If you pull 1 month per request, then 2 symbols × 24 months = 48 requests total.

That stays under the “50 open requests” and you can schedule them with spacing to avoid “identical within 15 seconds” rules.

What would break it

Trying to fetch many symbols concurrently with tiny windows and no caching.

Requesting the same exact historical window repeatedly while debugging.

Opening lots of streaming subscriptions and forgetting to cancel them.

9) Two concrete recommendations for your bot architecture
Recommendation 1: Cache by conId and by (barSize, useRTH, whatToShow)

If you treat historical data as immutable once stored, you drastically reduce API calls and avoid pacing violations.

Recommendation 2: Make “session detection” a first-class module

Use reqContractDetails schedule fields to determine:

first bar of the day

last bar of the day

holiday closures
This directly supports your overnight indicator and gap volatility block.