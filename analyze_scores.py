"""Does a better Magic Formula score lead to higher returns? Builds index.html to show it.

For every quarter end, scored stocks are split into ten equal groups by score, and each
group's average return over the next quarter and the next year is measured. A scatter of
score against 1-year return, with a fitted line, shows the same thing stock by stock.

Returns use dividend-adjusted prices, so dividends are included. A stock that stops trading
during the holding period (bought out or bankrupt) keeps its return up to its last trade.

The confidence interval for the top-minus-bottom gap treats each quarter's gap as one
observation, not each stock:
stocks in the same quarter move together, so they aren't independent. A 1-year return
starts every quarter, so neighboring years share nine months. Newey-West standard errors
correct for that overlap, and a block bootstrap is reported alongside as a check.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "data" / "scores" / "magic_formula_scores.csv"
YEARS = ROOT / "data" / "fmp" / "years"
PAGE = ROOT / "index.html"
TEMPLATE = ROOT / "report_template.html"
HORIZONS = {"quarter": 91, "year": 365}
# Quarters of autocorrelation to correct for. A 1-year return overlaps the next three.
NEWEY_WEST_LAGS = {"quarter": 1, "year": 4}
BOOTSTRAP_BLOCK = {"quarter": 2, "year": 4}
BOOTSTRAP_SAMPLES = 20_000
CONFIDENCE = 0.90
# Levels for the chart of how the interval narrows as the confidence asked for drops.
CURVE_LEVELS = np.round(np.arange(0.50, 0.995, 0.01), 2)
GROUPS = 10
MAX_START_PRICE_AGE_DAYS = 10
# A few FMP price series have glitches, like a $0.00 price, that produce impossible returns.
# Capping the most extreme 1% at each end keeps one bad tick from moving a group's average.
WINSOR_QUANTILE = 0.01
MIN_PRICE = 0.01
SCATTER_SAMPLE = 6000

log = logging.getLogger("analyze")


def load_prices():
    frames = [pd.read_parquet(path, columns=["symbol", "date", "adjClose"]) for path in sorted(YEARS.glob("*/prices.parquet"))]
    prices = pd.concat(frames, ignore_index=True).dropna()
    prices = prices[prices["adjClose"] > MIN_PRICE]
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.rename(columns={"symbol": "fmp_symbol"}).sort_values("date")


def forward_returns(scored, prices, days):
    last_trading_day = prices["date"].max()
    start = pd.merge_asof(
        scored.sort_values("score_date"), prices.rename(columns={"date": "start_date", "adjClose": "start_price"}),
        left_on="score_date", right_on="start_date", by="fmp_symbol", direction="backward",
        tolerance=pd.Timedelta(days=MAX_START_PRICE_AGE_DAYS),
    )
    start["target_date"] = start["score_date"] + pd.Timedelta(days=days)
    start = start[start["target_date"] <= last_trading_day].dropna(subset=["start_price"])
    # The last trade on or before the target date. For a stock that stopped trading, that is its final price.
    end = pd.merge_asof(
        start.sort_values("target_date"), prices.rename(columns={"date": "end_date", "adjClose": "end_price"}),
        left_on="target_date", right_on="end_date", by="fmp_symbol", direction="backward",
    )
    end = end[end["end_date"] > end["score_date"]]
    raw = end["end_price"] / end["start_price"] - 1
    low, high = raw.quantile([WINSOR_QUANTILE, 1 - WINSOR_QUANTILE])
    return end.assign(raw_return=raw, forward_return=raw.clip(low, high))


def add_score_groups(scored):
    """Score as a 0-100 percentile within each date (100 is best), and ten equal groups."""
    by_date = scored.groupby("score_date")["final_rank"]
    order = by_date.rank(method="first")
    count = by_date.transform("count")
    scored["score"] = 100 * (1 - (order - 1) / (count - 1))
    scored["group"] = (np.ceil(order / count * GROUPS)).astype(int)
    return scored


def newey_west_se(series, lags):
    """Standard error of a mean when neighboring observations are correlated."""
    x = np.asarray(series, dtype=float)
    deviations = x - x.mean()
    variance = deviations @ deviations / len(x)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (deviations[lag:] @ deviations[:-lag]) / len(x)
    return np.sqrt(variance / len(x))


def confidence_interval(series, lags, level=CONFIDENCE):
    se = newey_west_se(series, lags)
    margin = stats.t.ppf((1 + level) / 2, len(series) - 1) * se
    return series.mean() - margin, series.mean() + margin, se


def block_bootstrap_interval(series, block):
    """Resample runs of consecutive quarters, which keeps their correlation intact."""
    x = np.asarray(series, dtype=float)
    rng = np.random.default_rng(7)
    blocks = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block + 1, size=(BOOTSTRAP_SAMPLES, blocks))
    index = (starts[:, :, None] + np.arange(block)).reshape(BOOTSTRAP_SAMPLES, -1)[:, :len(x)]
    tail = (1 - CONFIDENCE) / 2 * 100
    return np.percentile(x[index].mean(axis=1), [tail, 100 - tail])


def group_summary(returns, horizon):
    """Each group's average, giving every quarter equal weight."""
    lags = NEWEY_WEST_LAGS[horizon]
    per_date = returns.groupby(["score_date", "group"])["forward_return"].mean().unstack()
    table = pd.DataFrame({
        "group": per_date.columns,
        "label": [f"{(g - 1) * 10}-{g * 10}%" for g in per_date.columns],
        "mean_return": per_date.mean().to_numpy(),
        "median_stock_return": returns.groupby("group")["forward_return"].median().to_numpy(),
        "beat_average_pct": (per_date.gt(per_date.mean(axis=1), axis=0).mean() * 100).to_numpy(),
        "observations": returns.groupby("group").size().to_numpy(),
    })
    spread = per_date[1] - per_date[GROUPS]
    spread_low, spread_high, spread_se = confidence_interval(spread, lags)
    t_stat = spread.mean() / spread_se
    p_value = 2 * stats.t.sf(abs(t_stat), len(spread) - 1)
    curve = [[float(level), *map(float, confidence_interval(spread, lags, level)[:2])] for level in CURVE_LEVELS]
    boot_low, boot_high = block_bootstrap_interval(spread, BOOTSTRAP_BLOCK[horizon])
    correlation = returns.groupby("score_date").apply(
        lambda d: d["score"].corr(d["forward_return"], method="spearman"), include_groups=False)
    summary = {
        "quarters": int(len(per_date)),
        "first_date": str(per_date.index.min().date()),
        "last_date": str(per_date.index.max().date()),
        "top_minus_bottom": float(spread.mean()),
        "spread_ci_low": float(spread_low),
        "spread_ci_high": float(spread_high),
        "spread_p_value": float(p_value),
        "confidence_level": CONFIDENCE,
        # The interval stops including zero once the confidence asked for drops below this.
        "break_even_confidence": float(1 - p_value),
        "one_sided_confidence": float(1 - p_value / 2) if spread.mean() > 0 else float(p_value / 2),
        "interval_curve": curve,
        "spread_bootstrap_low": float(boot_low),
        "spread_bootstrap_high": float(boot_high),
        "spread_naive_ci_halfwidth": float(stats.t.ppf((1 + CONFIDENCE) / 2, len(spread) - 1)
                                           * spread.std(ddof=1) / np.sqrt(len(spread))),
        "spread_by_date": [[str(d.date()), round(float(v), 5)] for d, v in spread.items()],
        "newey_west_lags": lags,
        "top_beats_bottom_pct": float((spread > 0).mean() * 100),
        "rank_correlation": float(correlation.mean()),
        "correlation_positive_pct": float((correlation > 0).mean() * 100),
        "all_stock_mean": float(per_date.mean(axis=1).mean()),
    }
    return table, summary


def scatter_data(returns):
    slope, intercept = np.polyfit(returns["score"], returns["forward_return"], 1)
    sample = returns.sample(min(SCATTER_SAMPLE, len(returns)), random_state=7)
    points = [[round(x, 1), round(y, 4), t, str(d.date())] for x, y, t, d in
              zip(sample["score"], sample["forward_return"], sample["ticker"], sample["score_date"])]
    return {
        "points": points,
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(returns["score"].corr(returns["forward_return"])),
        "count": int(len(returns)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    scores = pd.read_csv(SCORES, parse_dates=["score_date"])
    scored = add_score_groups(scores[scores["final_rank"].notna()].copy())
    prices = load_prices()

    payload = {"horizons": {}}
    for name, days in HORIZONS.items():
        returns = forward_returns(scored, prices, days)
        table, summary = group_summary(returns, name)
        payload["horizons"][name] = {"groups": table.round(5).to_dict("records"), "stats": summary}
        log.info("%s: %d stock-quarters over %d quarters, top minus bottom %+.1f%% (%d%% CI %+.1f%% to %+.1f%%, p=%.2f)",
                 name, len(returns), summary["quarters"], summary["top_minus_bottom"] * 100, CONFIDENCE * 100,
                 summary["spread_ci_low"] * 100, summary["spread_ci_high"] * 100, summary["spread_p_value"])
        if name == "year":
            payload["scatter"] = scatter_data(returns)

    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(payload))
    PAGE.write_text(page)
    log.info("Wrote %s", PAGE)


if __name__ == "__main__":
    main()
