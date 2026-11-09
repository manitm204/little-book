"""Pulls the benchmarks the strategy is judged against into data/fmp/benchmarks/.

  prices.parquet    daily dividend-adjusted closes for SPY (the S&P 500), RSP (equal-weight S&P 500),
                    and sector ETFs: XLV and XLY (cap-weighted health care and consumer discretionary)
                    and RSPH and RSPD (their equal-weight versions, which start in November 2006)
  tbills.parquet    daily 3-month Treasury bill yield, in percent, for Sharpe ratios
"""
import logging

import pandas as pd

from fmp_client import FMPClient, ROOT

OUT = ROOT / "data" / "fmp" / "benchmarks"
SYMBOLS = ["SPY", "RSP", "XLV", "XLY", "RSPH", "RSPD"]
START, END = "2004-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
# FMP returns at most about three months of Treasury rates per call.
TREASURY_CHUNK_DAYS = 80

log = logging.getLogger("benchmarks")


def pull_prices(client):
    frames = []
    for symbol in SYMBOLS:
        rows = []
        for year in range(int(START[:4]), int(END[:4]) + 1, 10):
            rows += client.get("historical-price-eod/dividend-adjusted", symbol=symbol,
                               **{"from": max(START, f"{year}-01-01"), "to": min(END, f"{year + 9}-12-31")})
        frame = pd.DataFrame(rows)[["symbol", "date", "adjClose"]].drop_duplicates("date")
        log.info("%s: %d days, %s to %s", symbol, len(frame), frame["date"].min(), frame["date"].max())
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["symbol", "date"])


def pull_tbills(client):
    rows = []
    start = pd.Timestamp(START)
    while start <= pd.Timestamp(END):
        end = start + pd.Timedelta(days=TREASURY_CHUNK_DAYS)
        rows += client.get("treasury-rates", **{"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")})
        start = end + pd.Timedelta(days=1)
    tbills = pd.DataFrame(rows)[["date", "month3"]].drop_duplicates("date").dropna()
    tbills["date"] = pd.to_datetime(tbills["date"])
    log.info("T-bills: %d days, %s to %s", len(tbills), tbills["date"].min().date(), tbills["date"].max().date())
    return tbills.sort_values("date")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    client = FMPClient()
    pull_prices(client).to_parquet(OUT / "prices.parquet", index=False)
    pull_tbills(client).to_parquet(OUT / "tbills.parquet", index=False)


if __name__ == "__main__":
    main()
