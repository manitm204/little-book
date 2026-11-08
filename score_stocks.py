"""Score every S&P 500 member each quarter on the two Little Book factors.

  return on capital = EBIT / (net working capital + net fixed assets)
  earnings yield    = EBIT / enterprise value

Stocks are ranked on each factor at every quarter end, the two ranks are added, and the
lowest total is best. Every member gets a row at every date. Members that can't be scored
keep their row, with excluded_reason saying why.

Only numbers public on the score date are used: EBIT is the trailing four quarters, and a
quarter counts once it has been filed.

FMP's historical market cap can't be used. It multiplies an adjusted price by a share count
that isn't always adjusted the same way, so eBay shows $6.8B in 2009 instead of $16B because
of the PayPal spin-off in 2015. That makes companies look cheap for years before a spin-off.
Market cap is rebuilt here as the price that actually traded times the shares outstanding then.

Output: data/scores/magic_formula_scores.csv and data/scores/score_coverage.csv
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from build_sp500_history import members_on

ROOT = Path(__file__).resolve().parent
YEARS = ROOT / "data" / "fmp" / "years"
OUT = ROOT / "data" / "scores"
# The book leaves these out because their balance sheets don't fit the two ratios.
EXCLUDED_SECTORS = {"Financial Services", "Utilities", "Real Estate"}
# Health insurers sit in Healthcare, but their cash is policyholder float. Counting it as spare
# cash shrinks enterprise value, which put Humana at $350M in 2009 against a $4.4B market cap.
EXCLUDED_INDUSTRIES = {"Medical - Healthcare Plans"}
# FMP filing dates under this many days after quarter end are placeholders, not real filings.
MIN_CREDIBLE_FILING_LAG = 10
# Stand-ins for those placeholders, set at about the 95th percentile of real filing lags.
FALLBACK_LAG_QUARTER = 45
FALLBACK_LAG_YEAR_END = 75
MAX_STATEMENT_AGE_DAYS = 200
MAX_PRICE_AGE_DAYS = 10
# Three quarter gaps of about 92 days each, with some slack. Wider means a quarter is missing.
MAX_TTM_SPAN_DAYS = 300
# No index member is this small, so a smaller FMP value is a data error.
MIN_MARKET_CAP = 100_000_000
# Spin-offs show up in FMP's split list with ratios like 297:125 or 25:4. Real splits are
# 1-to-something or simple like 3:2. Some are written oddly, like Google's 1998:1000, so
# near-whole ratios count too.
MAX_SIMPLE_SPLIT_TERM = 5
WHOLE_RATIO_TOLERANCE = 0.01
# Above this rate, FMP's debt figure is missing most of the debt that interest is being paid on.
MAX_INTEREST_RATE = 0.25
MIN_INTEREST_TO_CHECK = 50_000_000
# As a log change, about 28%. Mergers can move shares by this much, and split ratios move more.
MAX_SHARE_DRIFT = 0.25
SHARE_HISTORY_START = "2000-01-01"
EPOCH = pd.Timestamp("1970-01-01")

log = logging.getLogger("score")


def load_table(name, columns=None):
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(YEARS.glob(f"*/{name}.parquet"))]
    return pd.concat(frames, ignore_index=True)


def available_dates(statements):
    """The date each statement became public."""
    period_end = pd.to_datetime(statements["date"])
    filed = pd.to_datetime(statements["filingDate"])
    fallback_days = np.where(statements["period"] == "Q4", FALLBACK_LAG_YEAR_END, FALLBACK_LAG_QUARTER)
    fallback = period_end + pd.to_timedelta(fallback_days, unit="D")
    credible = (filed - period_end).dt.days >= MIN_CREDIBLE_FILING_LAG
    return filed.where(credible, fallback)


def median_ebit(statements):
    pretax_plus_interest = statements["incomeBeforeTax"] + statements["interestExpense"].fillna(0)
    estimates = pd.concat([statements["operatingIncome"], statements["ebit"], pretax_plus_interest], axis=1)
    # Nothing earns more than its revenue, so an estimate above it is a bad row and gets no vote.
    return estimates.where(estimates.le(statements["revenue"], axis=0)).median(axis=1)


def derive_fourth_quarters(income):
    """Replace Q4 EBIT with the annual figure minus the first three quarters.

    FMP builds Q4 rows itself, so all three estimates can be wrong together. Ryder's show
    about $3B in a year it earned about $560M. The annual statement is reported directly.
    """
    annual = load_table("income_annual").sort_values("filingDate").drop_duplicates(["symbol", "date"], keep="last")
    annual = annual.assign(annual_ebit=median_ebit(annual))[["symbol", "date", "annual_ebit"]]
    income = income.merge(annual, on=["symbol", "date"], how="left")
    first_three = income.groupby("symbol")["quarter_ebit"].transform(lambda s: s.shift(1).rolling(3).sum())
    period_end = pd.to_datetime(income["date"])
    span = (period_end - period_end.groupby(income["symbol"]).shift(3)).dt.days
    derived = income["annual_ebit"] - first_three
    # An annual figure that missed a write-down shows up here as a quarter earning more than its sales.
    plausible = derived.abs() <= income["revenue"].abs()
    usable = (income["period"] == "Q4") & plausible & (span <= MAX_TTM_SPAN_DAYS)
    income.loc[usable, "quarter_ebit"] = derived
    return income


def load_statements():
    """One row per symbol and quarter, with trailing EBIT and that quarter's balance sheet."""
    income = load_table("income_quarter").sort_values("filingDate")
    income = income.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    income["available"] = available_dates(income)
    # FMP's operating income and ebit fields each have bad rows: Rockwell Collins shows $1,951M
    # of operating income in a $220M quarter, and Caterpillar $2,212M of ebit in a $1,116M one.
    # Pre-tax income plus interest is a third estimate, so a bad row loses a vote of three.
    income["quarter_ebit"] = median_ebit(income)
    income = derive_fourth_quarters(income)

    grouped = income.groupby("symbol", sort=False)
    income["ebit_ttm"] = grouped["quarter_ebit"].transform(lambda s: s.rolling(4).sum())
    income["fmp_ebit_ttm"] = grouped["ebit"].transform(lambda s: s.rolling(4).sum())
    income["revenue_ttm"] = grouped["revenue"].transform(lambda s: s.rolling(4).sum())
    income["interest_ttm"] = grouped["interestExpense"].transform(lambda s: s.rolling(4).sum())
    # Rolling max doesn't accept dates, so take it on day numbers.
    available_day = (income["available"] - EPOCH).dt.days
    latest_day = available_day.groupby(income["symbol"]).transform(lambda s: s.rolling(4).max())
    income["ttm_available"] = EPOCH + pd.to_timedelta(latest_day, unit="D")
    period_end = pd.to_datetime(income["date"])
    span = (period_end - period_end.groupby(income["symbol"]).shift(3)).dt.days
    income.loc[~(span <= MAX_TTM_SPAN_DAYS), "ebit_ttm"] = np.nan

    balance = load_table("balance_quarter").sort_values("filingDate")
    balance = balance.drop_duplicates(["symbol", "date"], keep="last")
    balance["balance_available"] = available_dates(balance)
    keep = ["symbol", "date", "balance_available", "totalCurrentAssets", "totalCurrentLiabilities",
            "cashAndShortTermInvestments", "shortTermDebt", "propertyPlantEquipmentNet", "totalDebt",
            "preferredStock", "minorityInterest"]
    merged = income[["symbol", "date", "cik", "reportedCurrency", "ebit_ttm", "fmp_ebit_ttm", "revenue_ttm",
                     "interest_ttm", "ttm_available"]].merge(balance[keep], on=["symbol", "date"])
    merged["statement_available"] = merged[["ttm_available", "balance_available"]].max(axis=1)
    merged["statement_date"] = pd.to_datetime(merged["date"])
    return merged.dropna(subset=["ebit_ttm", "statement_available"]).drop(columns="date")


def load_sectors():
    profiles = pd.read_parquet(ROOT / "data" / "fmp" / "profiles.parquet").set_index("symbol")
    overrides = pd.read_csv(ROOT / "config" / "sector_overrides.csv").set_index("symbol")["sector"]
    return overrides.combine_first(profiles["sector"]).to_dict(), profiles["industry"].to_dict()


def load_splits():
    """All FMP split entries, with is_real False for spin-offs posing as splits."""
    splits = pd.read_parquet(ROOT / "data" / "fmp" / "splits.parquet")
    splits = pd.concat([splits, pd.read_csv(ROOT / "config" / "split_fixes.csv")], ignore_index=True)
    splits["date"] = pd.to_datetime(splits["date"])
    splits["ratio"] = splits["numerator"] / splits["denominator"]
    terms = splits[["numerator", "denominator"]]
    small_terms = (terms.min(axis=1) == 1) | (terms.max(axis=1) <= MAX_SIMPLE_SPLIT_TERM)
    near_whole = ((splits["ratio"] - splits["ratio"].round()).abs() <= WHOLE_RATIO_TOLERANCE) & (splits["ratio"] > 1.5)
    splits["is_real"] = small_terms | near_whole
    return splits[splits["ratio"] > 0].sort_values("date")


def apply_price_fixes(prices):
    for fix in pd.read_csv(ROOT / "config" / "price_fixes.csv").itertuples():
        rows = (prices["symbol"] == fix.symbol) & (prices["date"] < pd.Timestamp(fix.before_date))
        prices.loc[rows, "rawClose"] *= fix.multiplier
    return prices


def true_share_counts(group, symbol_splits):
    """Shares outstanding at each quarter, walking back from the latest one.

    A quarter's FMP share count may or may not be restated for later splits and spin-offs,
    depending on which filing it came from. Share counts barely move between quarters, so
    each quarter takes the reading that lines up with the quarter after it. Restatement
    changes rarely, so the walk keeps its current reading unless that one is clearly wrong.
    Otherwise a real jump from a merger could flip it onto the wrong reading for good.
    """
    dates = group["statement_date"].to_numpy()
    reported = group["shares"].to_numpy(dtype=float)
    true_shares = np.full(len(reported), np.nan)
    following = None
    # Newest splits and spin-offs this filing era has not restated for.
    unapplied_real = unapplied_spin = 0
    for i in range(len(reported) - 1, -1, -1):
        if not reported[i] > 0:
            continue
        later = symbol_splits[symbol_splits["date"] > dates[i]]
        real = np.concatenate([[1.0], later.loc[later["is_real"], "ratio"].cumprod().to_numpy()])
        spin = np.concatenate([[1.0], later.loc[~later["is_real"], "ratio"].cumprod().to_numpy()])
        readings = reported[i] / np.outer(real, spin)
        keep = (max(len(real) - 1 - unapplied_real, 0), max(len(spin) - 1 - unapplied_spin, 0))
        if following is None:
            # Nothing filed before a split can be restated for it, so the latest quarter is as reported.
            keep = (0, 0)
            unapplied_real, unapplied_spin = len(real) - 1, len(spin) - 1
        else:
            next_date, next_shares = following
            crossed = later[later["date"] <= next_date]
            expected = next_shares / crossed.loc[crossed["is_real"], "ratio"].prod()
            gaps = np.abs(np.log(readings / expected))
            # The quarters either side of a split are the clearest evidence, so always re-decide there.
            if len(crossed) or gaps[keep] > MAX_SHARE_DRIFT:
                keep = np.unravel_index(np.argmin(gaps), gaps.shape)
                unapplied_real, unapplied_spin = len(real) - 1 - keep[0], len(spin) - 1 - keep[1]
        true_shares[i] = readings[keep]
        following = (dates[i], true_shares[i])
    return true_shares


def load_share_history():
    """Full quarterly share counts per symbol, including years after a stock left the index.

    The walk has to start from the latest quarter FMP has, because only that one is certain
    not to need restating. The yearly files stop soon after a stock leaves the index.
    """
    frames = []
    for path in (ROOT / "data" / "fmp" / "statements_cache").glob("*.json"):
        rows = json.loads(path.read_text())["income_quarter"]
        if rows:
            frame = pd.DataFrame(rows)[["date", "weightedAverageShsOut"]].assign(symbol=path.stem)
            frames.append(frame[frame["date"] >= SHARE_HISTORY_START])
    history = pd.concat(frames, ignore_index=True).rename(columns={"weightedAverageShsOut": "shares"})
    history["statement_date"] = pd.to_datetime(history["date"])
    return history.drop_duplicates(["symbol", "statement_date"]).sort_values("statement_date")


def add_true_shares(statements, splits):
    frames = []
    for symbol, group in load_share_history().groupby("symbol"):
        counts = true_share_counts(group, splits[splits["symbol"] == symbol])
        frames.append(group.assign(true_shares=counts)[["symbol", "statement_date", "true_shares"]])
    return statements.merge(pd.concat(frames), on=["symbol", "statement_date"], how="left")


def shares_on_date(frame, splits, score_date):
    """Roll statement shares forward through any real split between the statement and today."""
    real = splits[splits["is_real"] & (splits["date"] <= score_date)]
    shares = frame["true_shares"].copy()
    for split in real[real["symbol"].isin(frame["fmp_symbol"])].itertuples():
        rows = (frame["fmp_symbol"] == split.symbol) & (frame["statement_date"] < split.date)
        shares[rows] *= split.ratio
    return shares


def latest_prices(prices, score_date):
    recent = prices[(prices["date"] <= score_date)
                    & (prices["date"] >= score_date - pd.Timedelta(days=MAX_PRICE_AGE_DAYS))]
    return recent.sort_values("date").drop_duplicates("symbol", keep="last").rename(columns={"date": "price_date"})


def latest_statements(statements, score_date):
    public = statements[statements["statement_available"] <= score_date]
    return public.sort_values("statement_date").drop_duplicates("symbol", keep="last")


def add_ratios(frame, splits, score_date):
    frame["net_working_capital"] = (
        (frame["totalCurrentAssets"] - frame["cashAndShortTermInvestments"])
        - (frame["totalCurrentLiabilities"] - frame["shortTermDebt"])
    ).clip(lower=0)  # Below zero would shrink capital toward nothing and make the ratio meaningless.
    frame["net_fixed_assets"] = frame["propertyPlantEquipmentNet"]
    frame["invested_capital"] = frame["net_working_capital"] + frame["net_fixed_assets"]

    frame["market_cap"] = frame["rawClose"] * shares_on_date(frame, splits, score_date)
    frame["fmp_market_cap"] = frame["marketCap"]

    frame["enterprise_value"] = (frame["market_cap"] + frame["totalDebt"] + frame["preferredStock"]
                                 + frame["minorityInterest"] - frame["cashAndShortTermInvestments"])
    frame["return_on_capital"] = frame["ebit_ttm"] / frame["invested_capital"]
    frame["earnings_yield"] = frame["ebit_ttm"] / frame["enterprise_value"]
    return frame


def exclusion_reasons(frame, score_date):
    """The first reason each stock can't be scored, or blank if it can."""
    age = (score_date - frame["statement_date"]).dt.days
    rules = [
        (~frame["status"].isin(["ok", "partial"]), frame["status"]),
        (frame["sector"].isna(), "no_sector"),
        (frame["sector"].isin(EXCLUDED_SECTORS), "excluded_sector"),
        (frame["industry"].isin(EXCLUDED_INDUSTRIES), "excluded_industry"),
        (frame["close"].isna() | (frame["price_from"] > score_date), "no_recent_price"),
        (frame["statement_date"].isna(), "no_public_statements"),
        (age > MAX_STATEMENT_AGE_DAYS, "stale_statements"),
        (frame["reportedCurrency"] != "USD", "non_usd_statements"),
        ((frame["totalCurrentAssets"] <= 0) | (frame["totalCurrentLiabilities"] <= 0), "no_current_items"),
        (frame["cashAndShortTermInvestments"] > frame["totalCurrentAssets"], "bad_balance_sheet"),
        (frame["ebit_ttm"] > frame["revenue_ttm"], "ebit_above_revenue"),
        ((frame["interest_ttm"] > MIN_INTEREST_TO_CHECK)
         & (frame["interest_ttm"] > frame["totalDebt"] * MAX_INTEREST_RATE), "debt_understated"),
        (frame["propertyPlantEquipmentNet"] <= 0, "no_fixed_assets"),
        (~(frame["market_cap"] >= MIN_MARKET_CAP), "bad_market_cap"),
        (frame["enterprise_value"] <= 0, "nonpositive_enterprise_value"),
    ]
    reasons = pd.Series("", index=frame.index, dtype=object)
    for condition, reason in reversed(rules):
        reasons = reasons.where(~condition.fillna(False).astype(bool), reason)
    return reasons


def mark_duplicate_classes(frame):
    """Keep one share class per company, the one with the most trading."""
    company = frame["cik"].where(frame["cik"].fillna("").str.strip("0") != "", frame["fmp_symbol"])
    scored = frame[frame["excluded_reason"] == ""].assign(company=company)
    keep = scored.sort_values(["volume", "fmp_symbol"], ascending=[False, True]).drop_duplicates("company").index
    frame.loc[scored.index.difference(keep), "excluded_reason"] = "duplicate_share_class"
    return frame


def rank(frame):
    scored = frame["excluded_reason"] == ""
    frame.loc[scored, "roc_rank"] = frame.loc[scored, "return_on_capital"].rank(ascending=False, method="min")
    frame.loc[scored, "ey_rank"] = frame.loc[scored, "earnings_yield"].rank(ascending=False, method="min")
    frame["combined_rank"] = frame["roc_rank"] + frame["ey_rank"]
    frame.loc[scored, "final_rank"] = frame.loc[scored, "combined_rank"].rank(method="min")
    return frame


def score_date_frame(spells, statements, prices, sectors, industries, splits, score_date):
    members = members_on(spells, score_date)[["ticker", "fmp_symbol", "name", "status", "price_from"]].copy()
    members["sector"] = members["fmp_symbol"].map(sectors)
    members["industry"] = members["fmp_symbol"].map(industries)
    frame = members.merge(latest_prices(prices, score_date), how="left", left_on="fmp_symbol", right_on="symbol")
    frame = frame.drop(columns="symbol").merge(
        latest_statements(statements, score_date), how="left", left_on="fmp_symbol", right_on="symbol"
    ).drop(columns="symbol")
    frame = add_ratios(frame, splits, score_date)
    frame["excluded_reason"] = exclusion_reasons(frame, score_date)
    frame = rank(mark_duplicate_classes(frame))
    frame.insert(0, "score_date", score_date)
    return frame


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    spells = pd.read_csv(ROOT / "data" / "sp500" / "sp500_spells.csv", parse_dates=["start", "end", "price_from"])
    spells["fmp_symbol"] = spells["fmp_symbol"].fillna("")
    prices = load_table("prices", ["symbol", "date", "close", "rawClose", "volume", "marketCap"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = apply_price_fixes(prices)
    splits = load_splits()
    statements = add_true_shares(load_statements(), splits)
    sectors, industries = load_sectors()

    # Last trading day of each completed quarter.
    days = pd.Series(prices["date"].unique()).sort_values()
    quarter_ends = days.groupby(days.dt.to_period("Q")).max()
    score_dates = [d for period, d in quarter_ends.items() if period.end_time < days.max()]

    frames = []
    for score_date in score_dates:
        frames.append(score_date_frame(spells, statements, prices, sectors, industries, splits, score_date))
    scores = pd.concat(frames, ignore_index=True)

    columns = ["score_date", "ticker", "fmp_symbol", "name", "sector", "final_rank", "combined_rank", "roc_rank",
               "ey_rank", "return_on_capital", "earnings_yield", "ebit_ttm", "fmp_ebit_ttm", "net_working_capital",
               "net_fixed_assets", "invested_capital", "market_cap", "fmp_market_cap", "enterprise_value",
               "rawClose", "true_shares", "price_date", "statement_date", "statement_available", "excluded_reason"]
    scores = scores[columns].sort_values(["score_date", "final_rank", "ticker"])
    scores.to_csv(OUT / "magic_formula_scores.csv", index=False)

    coverage = scores.groupby("score_date").agg(
        members=("ticker", "size"), scored=("final_rank", "count")).reset_index()
    coverage.to_csv(OUT / "score_coverage.csv", index=False)
    log.info("Scored %d dates from %s to %s, median %d stocks per date", len(coverage),
             coverage["score_date"].min().date(), coverage["score_date"].max().date(), coverage["scored"].median())
    log.info("Reasons stocks were not scored:\n%s", scores["excluded_reason"].replace("", np.nan).value_counts().to_string())


if __name__ == "__main__":
    main()
