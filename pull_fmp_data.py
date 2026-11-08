"""Pull prices and financial statements for every usable S&P 500 member, one year at a time.

Each year lands in data/fmp/years/<year>/ and holds everything needed to score that year:
  prices.parquet           adjusted close, dividend-adjusted close, actual traded close, volume,
                           and FMP market cap (which is wrong after spin-offs, see score_stocks.py)
  income_quarter.parquet   statements dated from two years before through this year, plus
  balance_quarter.parquet  the annual versions (rows overlap between years, so dedupe on load)
  income_annual.parquet
  balance_annual.parquet
  _summary.json            counts and failed symbols

A year is skipped on rerun once it finished with no failures, so a crash loses at most the
year in progress. The current year is never treated as finished, because it is still growing.

Usage:
  python pull_fmp_data.py                 pull everything not yet finished
  python pull_fmp_data.py --years 2005    pull one year (or several, or 2005-2008)
  python pull_fmp_data.py --force         pull requested years again
"""
import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fmp_client import FMPClient

ROOT = Path(__file__).resolve().parent
SPELLS_FILE = ROOT / "data" / "sp500" / "sp500_spells.csv"
DEFAULT_OUT = ROOT / "data" / "fmp"
FIRST_YEAR = 2005
FETCH_WORKERS = 8
# Keep prices after a stock leaves the index, so a position held through the exit has returns.
HOLD_DAYS = 370
STATEMENT_LOOKBACK_YEARS = 2
STATEMENT_DATASETS = {
    "income_quarter": ("income-statement", "quarter", 200),
    "balance_quarter": ("balance-sheet-statement", "quarter", 200),
    "income_annual": ("income-statement", "annual", 60),
    "balance_annual": ("balance-sheet-statement", "annual", 60),
}

log = logging.getLogger("pull")


def load_windows():
    """Date ranges where each FMP symbol really is the index member, plus the holding tail.

    A ticker can be reused by another company later, so anything outside these windows is
    dropped even when FMP returns it.
    """
    spells = pd.read_csv(SPELLS_FILE, parse_dates=["end", "price_from"])
    spells = spells[spells["status"].isin(["ok", "partial"])]
    today = pd.Timestamp.today().normalize()
    windows = {}
    for s in spells.itertuples():
        end = s.end if pd.notna(s.end) else today
        windows.setdefault(s.fmp_symbol, []).append((s.price_from, end + pd.Timedelta(days=HOLD_DAYS)))
    return windows


def symbols_for_year(windows, year):
    lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    return sorted(s for s, spans in windows.items() if any(a <= hi and b >= lo for a, b in spans))


def clip_to_windows(frame, spans):
    dates = pd.to_datetime(frame["date"])
    keep = pd.Series(False, index=frame.index)
    for lo, hi in spans:
        keep |= (dates >= lo) & (dates <= hi)
    return frame[keep]


def fetch_prices(client, symbol, year, spans):
    dates = {"from": f"{year}-01-01", "to": f"{year}-12-31"}
    full = pd.DataFrame(client.get("historical-price-eod/full", symbol=symbol, **dates))
    if full.empty:
        return full
    prices = full[["symbol", "date", "close", "volume"]]
    adjusted = pd.DataFrame(client.get("historical-price-eod/dividend-adjusted", symbol=symbol, **dates))
    if not adjusted.empty:
        prices = prices.merge(adjusted[["date", "adjClose"]], on="date", how="left")
    raw = pd.DataFrame(client.get("historical-price-eod/non-split-adjusted", symbol=symbol, **dates))
    if not raw.empty:
        prices = prices.merge(raw[["date", "adjClose"]].rename(columns={"adjClose": "rawClose"}), on="date", how="left")
    caps = pd.DataFrame(client.get("historical-market-capitalization", symbol=symbol, limit=500, **dates))
    if not caps.empty:
        prices = prices.merge(caps[["date", "marketCap"]], on="date", how="left")
    prices = prices.reindex(columns=["symbol", "date", "close", "adjClose", "rawClose", "volume", "marketCap"])
    return clip_to_windows(prices, spans)


def fetch_statements(client, symbol, cache_dir):
    """All statements for a symbol. They come back whole, so fetch once and reuse across years."""
    path = cache_dir / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = {name: client.get(endpoint, symbol=symbol, period=period, limit=limit)
            for name, (endpoint, period, limit) in STATEMENT_DATASETS.items()}
    profile = client.get("profile", symbol=symbol)
    data["profile"] = profile[0] if profile else {}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    temp.replace(path)
    return data


def statements_for_year(rows, year, spans):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    first = f"{year - STATEMENT_LOOKBACK_YEARS}-01-01"
    frame = frame[(frame["date"] >= first) & (frame["date"] <= f"{year}-12-31")]
    # Only the end of the window applies: scoring needs statements from before a spell starts.
    latest = max(hi for _, hi in spans)
    return frame[pd.to_datetime(frame["date"]) <= latest]


def fetch_symbol(client, symbol, year, spans, cache_dir):
    prices = fetch_prices(client, symbol, year, spans)
    statements = fetch_statements(client, symbol, cache_dir)
    return prices, {name: statements_for_year(statements[name], year, spans) for name in STATEMENT_DATASETS}


def pull_year(client, year, windows, out_dir, cache_dir, max_symbols=None):
    symbols = symbols_for_year(windows, year)
    if max_symbols:
        symbols = symbols[:max_symbols]
    log.info("Pulling %d: %d symbols", year, len(symbols))

    prices, tables, failed = [], {name: [] for name in STATEMENT_DATASETS}, {}
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_symbol, client, s, year, windows[s], cache_dir): s for s in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                symbol_prices, symbol_tables = future.result()
            except RuntimeError as e:
                failed[symbol] = str(e)
                log.warning("  %s failed: %s", symbol, e)
                continue
            prices.append(symbol_prices)
            for name, frame in symbol_tables.items():
                tables[name].append(frame)
            if i % 100 == 0:
                log.info("  %d: %d/%d symbols", year, i, len(symbols))

    # Build the year in a temp folder and swap it in, so a crash can't leave half a year behind.
    final, temp = out_dir / str(year), out_dir / f"{year}.tmp"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    all_prices = pd.concat(prices, ignore_index=True) if prices else pd.DataFrame()
    all_prices.to_parquet(temp / "prices.parquet", index=False)
    counts = {"prices": len(all_prices)}
    for name, frames in tables.items():
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        table.to_parquet(temp / f"{name}.parquet", index=False)
        counts[name] = len(table)

    priced = set(all_prices["symbol"]) if len(all_prices) else set()
    summary = {
        "year": year,
        "symbols": len(symbols),
        "symbols_with_prices": len(priced),
        "symbols_without_prices": sorted(set(symbols) - priced - set(failed)),
        "rows": counts,
        "failed": failed,
        "finished": not failed and year < pd.Timestamp.today().year,
    }
    (temp / "_summary.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(temp, final)

    note = f" ({len(failed)} symbols failed, rerun to retry this year)" if failed else ""
    log.info("Got %d data: %d symbols with prices, %s price rows%s",
             year, len(priced), f"{counts['prices']:,}", note)


def is_finished(out_dir, year):
    path = out_dir / str(year) / "_summary.json"
    return path.exists() and json.loads(path.read_text())["finished"]


def write_profiles(cache_dir, out_dir):
    """Sector and industry per symbol, needed to drop financials and utilities when scoring."""
    rows = []
    for path in cache_dir.glob("*.json"):
        profile = json.loads(path.read_text())["profile"]
        rows.append({"symbol": path.stem, "name": profile.get("companyName"), "sector": profile.get("sector"),
                     "industry": profile.get("industry"), "country": profile.get("country")})
    pd.DataFrame(rows).sort_values("symbol").to_parquet(out_dir / "profiles.parquet", index=False)


def write_splits(client, windows, out_dir):
    """Every split FMP lists for each symbol. Spin-offs appear here too, as odd ratios."""
    def fetch(symbol):
        return client.get("splits", symbol=symbol)

    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        rows = [row for result in pool.map(fetch, sorted(windows)) for row in result]
    pd.DataFrame(rows).to_parquet(out_dir / "splits.parquet", index=False)


def parse_years(values):
    years = []
    for value in values:
        lo, _, hi = value.partition("-")
        years += range(int(lo), int(hi or lo) + 1)
    return sorted(set(years))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", nargs="+", help="years to pull, such as 2005 or 2005-2008")
    parser.add_argument("--force", action="store_true", help="pull again even if finished")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--max-symbols", type=int, help="only pull the first N symbols, for testing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "pull.log")],
    )

    years_dir, cache_dir = args.out / "years", args.out / "statements_cache"
    years_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    client = FMPClient()
    windows = load_windows()
    this_year = pd.Timestamp.today().year
    years = parse_years(args.years) if args.years else range(FIRST_YEAR, this_year + 1)

    for year in years:
        if not args.force and is_finished(years_dir, year):
            log.info("Skipping %d, already finished", year)
            continue
        pull_year(client, year, windows, years_dir, cache_dir, args.max_symbols)
    write_profiles(cache_dir, args.out)
    write_splits(client, windows, args.out)
    log.info("Done")


if __name__ == "__main__":
    main()
