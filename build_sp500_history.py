"""Build a point-in-time S&P 500 membership file from FMP's change log.

FMP only publishes today's list plus a log of changes, so membership is rebuilt by
walking the log backward from today. Each membership spell is then checked against
FMP, because a ticker in FMP today may be a different company from the one that was
in the index. fmp_symbol is only filled in for spells that pass, so downstream pulls
cannot fetch the wrong company by accident.

Outputs in data/sp500/:
  sp500_spells.csv           one row per membership spell, with data quality status
  sp500_month_end.csv        members at every month end (date, ticker, fmp_symbol, usable)
  sp500_coverage_by_year.csv how much of the index is usable at each mid-year
  sp500_review_needed.csv    spells that look like the wrong company and need a human decision

Run with --refresh to re-download instead of using the cached raw files.
"""
import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fmp_client import FMPClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sp500"
RAW = OUT / "raw"
CONFIG = ROOT / "config"
WINDOW_START = pd.Timestamp("2005-01-01")
MIN_PRICE_COVERAGE = 0.90
MIN_FUNDAMENTAL_COVERAGE = 0.80
# A statement older than this is too stale to score a stock with.
MAX_STATEMENT_AGE_DAYS = 550
# Trading days as a share of weekdays, after market holidays.
TRADING_DAY_RATIO = 0.96
# Index members trade millions of shares a day. A thin series means FMP's ticker is really
# some other company, which name matching misses when the old name is still attached.
MIN_MEDIAN_VOLUME = 100_000
FETCH_WORKERS = 4

log = logging.getLogger("sp500")
NAME_STOPWORDS = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|the|holdings|"
    r"holding|group|llc|lp|class|new|de|sa|nv|intl|int l|international|trust|common|stock|shares)\b"
)


def name_tokens(name):
    text = str(name).lower().replace("'", "").replace(".", "")
    return NAME_STOPWORDS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", text)).split()


def cached_json(client, filename, endpoint, refresh):
    path = RAW / filename
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = client.get(endpoint)
    path.write_text(json.dumps(data))
    return data


def apply_corrections(changes):
    corrections = pd.read_csv(CONFIG / "sp500_log_corrections.csv", dtype=str)
    changes = changes.copy()
    for c in corrections.itertuples():
        column, security = (
            ("addedTicker", "addedSecurity") if c.side == "added" else ("removedTicker", "removedSecurity")
        )
        hit = (
            (changes["date"] == pd.Timestamp(c.date))
            & (changes[column] == c.log_ticker)
            & changes[security].fillna("").str.contains(c.security_contains, regex=False)
        )
        if hit.sum() != 1:
            raise ValueError(f"Correction matched {hit.sum()} rows: {c.date} {c.side} {c.log_ticker}")
        changes.loc[hit, column] = c.new_ticker
    return changes


def build_spells(current, changes):
    """Walk the log backward from today. A spell covers start <= date < end.

    Log dates are either the effective day or the last trading day before it, so a
    spell can start or end one trading day early.
    """
    open_end = {symbol: pd.NaT for symbol in current["symbol"]}
    names = dict(zip(current["symbol"], current["name"]))
    spells = []
    in_window = changes[changes["date"] >= WINDOW_START].sort_values("date", ascending=False)

    for row in in_window.itertuples():
        added, removed = row.addedTicker, row.removedTicker
        if added:
            if added not in open_end:
                raise ValueError(f"{row.date.date()}: {added} added but not a member afterwards")
            spells.append((added, row.date, open_end.pop(added), names.get(added) or row.addedSecurity))
        if removed:
            if removed in open_end:
                raise ValueError(f"{row.date.date()}: {removed} removed but already a member afterwards")
            open_end[removed] = row.date
            names[removed] = row.removedSecurity

    # What is left was already a member when the window opened.
    earlier = changes[changes["date"] < WINDOW_START].sort_values("date")
    last_added = earlier.groupby("addedTicker")["date"].last()
    for ticker, end in open_end.items():
        spells.append((ticker, last_added.get(ticker, pd.NaT), end, names.get(ticker)))

    spells = pd.DataFrame(spells, columns=["ticker", "start", "end", "name"])
    return spells.sort_values(["ticker", "start"], na_position="first").reset_index(drop=True)


def load_fmp_symbol_facts(client, symbols, refresh):
    """Trading dates, statement dates and name for each symbol, cached to survive reruns."""
    path = RAW / "symbol_facts.json"
    facts = {} if refresh or not path.exists() else json.loads(path.read_text())
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    def fetch(symbol):
        profile = client.get("profile", symbol=symbol)
        statements = client.get("income-statement", symbol=symbol, period="annual", limit=40)
        prices = client.price_history(symbol, "2004-01-01", today)
        return symbol, {
            "name": profile[0].get("companyName") if profile else None,
            "cik": profile[0].get("cik") if profile else None,
            "prices": [row["date"] for row in prices],
            "volumes": [row.get("volume") or 0 for row in prices],
            "statements": sorted(s["date"] for s in statements),
        }

    missing = sorted(set(symbols) - set(facts))
    log.info("Fetching FMP facts for %d symbols (%d cached)", len(missing), len(facts))
    failed = []
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in missing}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                symbol, fact = future.result()
                facts[symbol] = fact
            except RuntimeError as e:
                failed.append(futures[future])
                log.warning("Skipping %s for now: %s", futures[future], e)
            if i % 100 == 0:
                log.info("  %d/%d", i, len(missing))
                path.write_text(json.dumps(facts))
    path.write_text(json.dumps(facts))
    if failed:
        raise SystemExit(f"{len(failed)} symbols failed. Progress is saved, so rerun to fetch only those.")
    return facts


def assess_spell(spell, override, facts, today):
    """Decide whether FMP data can be trusted for this spell, and over which dates."""
    result = {"fmp_symbol": "", "status": "no_data", "fmp_name": None, "cik": None, "price_coverage": 0.0,
              "fundamental_coverage": 0.0, "price_from": pd.NaT, "price_to": pd.NaT,
              "statement_from": pd.NaT, "statement_to": pd.NaT, "median_volume": None, "note": "",
              "suspect": False}
    action = override.get("action", "")
    if isinstance(override.get("reason"), str):
        result["note"] = override["reason"]
    if action == "block":
        result["status"] = "wrong_company"
        return result

    symbol = override.get("fmp_symbol") if action == "map" else spell.ticker
    fact = facts[symbol]
    start = max(spell.start, WINDOW_START) if pd.notna(spell.start) else WINDOW_START
    end = spell.end if pd.notna(spell.end) else today
    if isinstance(override.get("valid_from"), str):
        valid_from = pd.Timestamp(override["valid_from"])
    else:
        valid_from = start

    prices = pd.Series(fact["volumes"], index=pd.to_datetime(fact["prices"]), dtype="float64")
    prices = prices[(prices.index >= max(start, valid_from)) & (prices.index <= end)]
    expected_days = max(len(pd.bdate_range(start, end)) * TRADING_DAY_RATIO, 1)
    result["price_coverage"] = round(min(len(prices) / expected_days, 1.0), 3)
    result["fmp_name"], result["cik"] = fact["name"], fact["cik"]
    if len(prices) == 0:
        return result

    tokens_log, tokens_fmp = name_tokens(spell.name), name_tokens(fact["name"])
    name_differs = bool(
        action not in ("same", "map", "low_volume_ok")
        and tokens_fmp and (not tokens_log or tokens_log[0] != tokens_fmp[0])
    )
    result["median_volume"] = float(prices.median())
    thin = result["median_volume"] < MIN_MEDIAN_VOLUME and action != "low_volume_ok"
    result["suspect"] = name_differs or thin
    result["fmp_symbol"] = symbol
    result["price_from"], result["price_to"] = prices.index.min(), prices.index.max()

    statements = pd.to_datetime(fact["statements"])
    lookback = pd.Timedelta(days=MAX_STATEMENT_AGE_DAYS)
    statements = statements[(statements >= start - lookback) & (statements <= end)]
    if len(statements) == 0:
        result["status"] = "no_fundamentals"
        return result
    result["statement_from"], result["statement_to"] = statements.min(), statements.max()
    years = max((end - start).days / 365.25, 1)
    result["fundamental_coverage"] = round(min(len(statements) / years, 1.0), 3)

    complete = (
        result["price_coverage"] >= MIN_PRICE_COVERAGE
        and result["fundamental_coverage"] >= MIN_FUNDAMENTAL_COVERAGE
    )
    result["status"] = "ok" if complete else "partial"
    return result


def members_on(spells, date):
    date = pd.Timestamp(date)
    started = spells["start"].isna() | (spells["start"] <= date)
    not_ended = spells["end"].isna() | (spells["end"] > date)
    return spells[started & not_ended]


def usable_on(members, date):
    """True where a member has a price and a recent statement at this date."""
    date = pd.Timestamp(date)
    has_status = members["status"].isin(["ok", "partial"])
    has_price = (members["price_from"] <= date) & (members["price_to"] >= date)
    recent = (members["statement_from"] <= date) & (
        date - members["statement_to"] <= pd.Timedelta(days=MAX_STATEMENT_AGE_DAYS)
    )
    return has_status & has_price & recent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download instead of using cached raw files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    RAW.mkdir(parents=True, exist_ok=True)

    client = FMPClient()
    current = pd.DataFrame(cached_json(client, "sp500_current.json", "sp500-constituent", args.refresh))
    changes = pd.DataFrame(cached_json(client, "sp500_changes.json", "historical-sp500-constituent", args.refresh))
    changes["date"] = pd.to_datetime(changes["date"])
    log.info("Loaded %d current members and %d log rows", len(current), len(changes))

    spells = build_spells(current, apply_corrections(changes))
    log.info("Built %d membership spells since %s", len(spells), WINDOW_START.date())

    overrides = pd.read_csv(CONFIG / "sp500_symbol_overrides.csv", dtype=str).set_index("ticker")
    overrides = overrides.to_dict("index")
    symbols = set()
    for ticker in spells["ticker"]:
        override = overrides.get(ticker, {})
        if override.get("action") == "block":
            continue
        symbols.add(override["fmp_symbol"] if override.get("action") == "map" else ticker)
    # Corrected tickers such as T.ATTCORP are internal labels with nothing behind them at FMP.
    facts = load_fmp_symbol_facts(client, symbols, args.refresh)

    today = pd.Timestamp.today().normalize()
    assessed = [assess_spell(s, overrides.get(s.ticker, {}), facts, today) for s in spells.itertuples()]
    spells = pd.concat([spells, pd.DataFrame(assessed)], axis=1)

    review = spells[spells["suspect"]]
    review[["ticker", "start", "end", "name", "fmp_name", "median_volume"]].to_csv(
        OUT / "sp500_review_needed.csv", index=False)
    if len(review):
        log.warning("%d spells look like the wrong company and are excluded until reviewed", len(review))
        spells.loc[spells["suspect"], "status"] = "needs_review"
        spells.loc[spells["suspect"], "fmp_symbol"] = ""
    spells.drop(columns="suspect").to_csv(OUT / "sp500_spells.csv", index=False)

    snapshots, coverage = [], []
    for date in pd.date_range(WINDOW_START, today, freq="ME"):
        members = members_on(spells, date)
        snapshots.append(members.assign(date=date.date(), usable=usable_on(members, date))[
            ["date", "ticker", "fmp_symbol", "name", "status", "usable"]])
    pd.concat(snapshots).to_csv(OUT / "sp500_month_end.csv", index=False)

    for year in range(WINDOW_START.year, today.year + 1):
        date = min(pd.Timestamp(f"{year}-06-30"), today)
        members = members_on(spells, date)
        usable = usable_on(members, date).sum()
        coverage.append({"date": date.date(), "members": len(members), "usable": int(usable),
                         "usable_pct": round(usable / len(members) * 100, 1)})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT / "sp500_coverage_by_year.csv", index=False)
    log.info("Status counts:\n%s", spells["status"].value_counts().to_string())
    log.info("Coverage:\n%s", coverage.to_string(index=False))


if __name__ == "__main__":
    main()
