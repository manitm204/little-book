"""Equity curves for the score groups, and how the top 10% compares with the S&P 500.

Money is split into four sleeves, one per quarter of the year. Each sleeve rebalances once a
year, three months after the one before it, so no single start date drives the result. A
sleeve buys the group equally weighted and lets the holdings drift for the year. A stock that
stops trading stays at its last price until the sleeve's next rebalance.

Each simulated path buys a random half of the group at every rebalance. The spread of paths
shows how much rides on the exact stocks held. It does not show the luck of which years
happened, which the rolling windows and calendar years are there to show.

Portfolios are valued at every month end. The S&P 500 is SPY and the equal-weight S&P 500 is
RSP, both with dividends reinvested. The risk-free rate is the 3-month Treasury bill.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT / "data" / "fmp" / "benchmarks"
TOP, BOTTOM = 1, 10
SLEEVES = 4
HOLD_MONTHS = 12
SIM_PATHS = 2000
PICK_FRACTION = 0.5
FAN_PERCENTILES = [5, 25, 50, 75, 95]
# Holding windows, in years, for the "what if you started on a different date" check.
WINDOW_YEARS = [3, 5]
# The same start-price and capping rules as the score-group analysis.
MAX_START_PRICE_AGE_DAYS = 10
WINSOR_QUANTILE = 0.01
# Long-short hedges: trailing window, in months, for the betas that size the short side, and the
# fewest months needed before hedging by beta (until then the short side matches the long side).
BETA_WINDOW = 36
BETA_MIN_MONTHS = 12
MAX_HEDGE = 3.0
# Newey-West lags for the t-statistic of the average monthly excess return.
T_STAT_LAGS = 3
# Where the two halves of the test split, to ask whether the edge has faded.
SPLIT_DATE = pd.Timestamp("2016-01-01")
HISTOGRAM_EDGES = np.arange(-0.08, 0.0801, 0.01)


def month_marks(scored, prices):
    """Every month end from the first score date to the last month end with prices."""
    last = prices["date"].max()
    end = last if last.is_month_end else last - pd.offsets.MonthEnd(1)
    return pd.date_range(scored["score_date"].min() + pd.offsets.MonthEnd(0), end, freq="ME")


def holding_growth(scored, prices, marks):
    """Each stock's value at each of the next 12 month ends, per $1 invested at its score date."""
    start = pd.merge_asof(
        scored.sort_values("score_date"), prices.rename(columns={"date": "start_date", "adjClose": "start_price"}),
        left_on="score_date", right_on="start_date", by="fmp_symbol", direction="backward",
        tolerance=pd.Timedelta(days=MAX_START_PRICE_AGE_DAYS),
    ).dropna(subset=["start_price"]).reset_index(drop=True)
    # Score dates are the last trading day of a quarter, which can fall before the calendar month end.
    start["mark"] = (start["score_date"] + pd.offsets.MonthEnd(0)).map({d: i for i, d in enumerate(marks)})
    start = start.dropna(subset=["mark"]).astype({"mark": int}).reset_index(drop=True)
    growth = np.full((len(start), HOLD_MONTHS), np.nan)
    for k in range(1, HOLD_MONTHS + 1):
        held = start[start["mark"] + k < len(marks)].copy()
        if held.empty:
            continue
        held["target_date"] = marks[held["mark"].to_numpy() + k]
        held["row"] = held.index
        # The last trade on or before the mark date, so a stock that stopped trading stays at its final price.
        end = pd.merge_asof(
            held.sort_values("target_date"), prices.rename(columns={"date": "end_date", "adjClose": "end_price"}),
            left_on="target_date", right_on="end_date", by="fmp_symbol", direction="backward",
        )
        raw = (end["end_price"] / end["start_price"] - 1).where(end["end_date"] > end["score_date"], 0.0)
        low, high = raw.quantile([WINSOR_QUANTILE, 1 - WINSOR_QUANTILE])
        growth[end["row"].to_numpy(), k - 1] = 1 + raw.clip(low, high).to_numpy()
    return {m: (frame["group"].to_numpy(), growth[frame.index]) for m, frame in start.groupby("mark")}


def sleeve_values(growth_by_mark, n_marks, offset, members, rng, paths, fraction):
    """Value of $1 in one sleeve at every month end. It buys on the first date, then rebalances
    every 12 months starting `offset` months in, each time buying `fraction` of the group at
    random, equally weighted."""
    values = np.ones((paths, n_marks))
    rebalances = [0, *range(offset or HOLD_MONTHS, n_marks, HOLD_MONTHS)]
    for begin, end in zip(rebalances, [*rebalances[1:], n_marks - 1]):
        if end <= begin:
            continue
        groups, growth = growth_by_mark[begin]
        pool = growth[members(groups), :end - begin]
        count = len(pool)
        picks = max(1, int(round(count * fraction)))
        chosen = np.argsort(rng.random((paths, count)), axis=1)[:, :picks]
        weights = np.zeros((paths, count))
        np.put_along_axis(weights, chosen, 1 / picks, axis=1)
        values[:, begin + 1:end + 1] = values[:, [begin]] * (weights @ pool)
    return values


def portfolio_values(growth_by_mark, n_marks, members, rng, paths=1, fraction=1.0, sleeves=range(SLEEVES)):
    """Equal money in each sleeve at the start, each left to grow on its own."""
    step = HOLD_MONTHS // SLEEVES
    return np.mean([sleeve_values(growth_by_mark, n_marks, s * step, members, rng, paths, fraction)
                    for s in sleeves], axis=0)


def benchmark_curves(marks):
    prices = pd.read_parquet(BENCHMARKS / "prices.parquet")
    curves = {}
    for symbol, frame in prices.groupby("symbol"):
        series = frame.set_index("date")["adjClose"].sort_index()
        values = series.asof(marks).to_numpy()
        # A fund that launched after the first mark is empty until then and starts at $1 on its first month end.
        values[marks < series.index[0]] = np.nan
        curves[symbol] = values / values[~np.isnan(values)][0]
    tbills = pd.read_parquet(BENCHMARKS / "tbills.parquet").set_index("date")["month3"].sort_index()
    # The yield known at the start of each month, earned over that month.
    risk_free = (1 + tbills.asof(marks[:-1]).to_numpy() / 100) ** (1 / 12) - 1
    return curves, risk_free


def drawdowns(values):
    return values / np.maximum.accumulate(values, axis=-1) - 1


def metrics(values, market, risk_free, years):
    """Performance measures for each path (rows of `values`) against the S&P 500 (`market`)."""
    values = np.atleast_2d(values)
    # The S&P 500 against itself has no tracking error, so its information ratio divides by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = values[:, 1:] / values[:, :-1] - 1
        m = market[1:] / market[:-1] - 1
        excess, market_excess = r - risk_free, m - risk_free
        downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2, axis=1))
        beta = ((r - r.mean(axis=1, keepdims=True)) @ (m - m.mean())) / ((m - m.mean()) @ (m - m.mean()))
        active = r - m
        up, down = m > 0, m < 0
        correlation = beta * (m - m.mean()).std() / (r - r.mean(axis=1, keepdims=True)).std(axis=1)
        return {
            "cagr": values[:, -1] ** (1 / years) - 1,
            "volatility": r.std(axis=1, ddof=1) * np.sqrt(12),
            "excess_return": excess.mean(axis=1) * 12,
            "sharpe": excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(12),
            "sortino": excess.mean(axis=1) * 12 / (downside * np.sqrt(12)),
            "max_drawdown": drawdowns(values).min(axis=1),
            "beta": beta,
            "alpha": (excess.mean(axis=1) - beta * market_excess.mean()) * 12,
            "tracking_error": active.std(axis=1, ddof=1) * np.sqrt(12),
            "information_ratio": active.mean(axis=1) * 12 / (active.std(axis=1, ddof=1) * np.sqrt(12)),
            "up_capture": r[:, up].mean(axis=1) / m[up].mean(),
            "down_capture": r[:, down].mean(axis=1) / m[down].mean(),
            "months_beating": (active > 0).mean(axis=1),
            "correlation": correlation,
            # Share of the monthly ups and downs that the market's moves explain.
            "r_squared": correlation ** 2,
            "t_stat": newey_west_t(excess),
            "months_positive": (r > 0).mean(axis=1),
            "ending_value": values[:, -1],
        }


def newey_west_t(x, lags=T_STAT_LAGS):
    """t-statistic of each row's mean, allowing for correlation between neighboring months."""
    d = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    variance = (d * d).sum(axis=1) / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (d[:, lag:] * d[:, :-lag]).sum(axis=1) / n
    return x.mean(axis=1) / np.sqrt(variance / n)


def trailing_beta(r, m, window=BETA_WINDOW, minimum=BETA_MIN_MONTHS):
    """Beta of each row of `r` to `m` in each month, from the months before it only. NaN until `minimum` months."""
    beta = np.full(r.shape, np.nan)
    for t in range(minimum, r.shape[1]):
        lo = max(0, t - window)
        rr, mm = r[:, lo:t], m[lo:t] - m[lo:t].mean()
        beta[:, t] = (rr - rr.mean(axis=1, keepdims=True)) @ mm / (mm @ mm)
    return beta


def long_short(long_values, short_values, market, risk_free, hedge):
    """$1 long, and short either $1 ("dollar") or enough to cancel the long side's market beta ("beta").
    The cash from the short sale earns the T-bill rate. Borrowing costs are left out.
    Returns the value of $1 at each month end and the hedge ratio used each month."""
    def monthly(values):
        values = np.atleast_2d(values)
        return values[:, 1:] / values[:, :-1] - 1
    long_r, short_r = monthly(long_values), monthly(short_values)
    m = market[1:] / market[:-1] - 1
    if hedge == "beta":
        ratio = trailing_beta(long_r, m) / trailing_beta(np.broadcast_to(short_r, long_r.shape), m)
        ratio = np.clip(np.nan_to_num(ratio, nan=1.0), 0, MAX_HEDGE)
    else:
        ratio = np.ones_like(long_r)
    r = risk_free + (long_r - risk_free) - ratio * (short_r - risk_free)
    values = np.concatenate([np.ones((r.shape[0], 1)), np.cumprod(1 + r, axis=1)], axis=1)
    return values, ratio


def summarize(measures):
    """Median and 5th to 95th percentile of each measure across paths."""
    # Measures against the S&P 500 are undefined for the S&P 500 itself, so those come out empty.
    return {k: [float(x) for x in np.percentile(v, [50, 5, 95])] if np.isfinite(v).all() else None
            for k, v in measures.items()}


def calendar_years(values, marks):
    """Return for each calendar year, from the last mark of one year to the last of the next.
    The first and last years are partial."""
    frame = pd.DataFrame(np.atleast_2d(values).T, index=marks)
    year_ends = frame.groupby(frame.index.year).tail(1)
    starts = pd.concat([frame.iloc[[0]], year_ends.iloc[:-1]])
    returns = year_ends.to_numpy() / starts.to_numpy() - 1
    labels = []
    for first, last in zip(starts.index, year_ends.index):
        covered_from = first + pd.offsets.MonthBegin(1) if first.month != 12 else pd.Timestamp(last.year, 1, 1)
        partial = covered_from.month != 1 or last.month != 12
        labels.append(f"{last.year} ({covered_from:%b}–{last:%b})" if partial else str(last.year))
    return labels, returns.T


def sector_mix(scored):
    """Average share of each sector in the top 10% and in all scored stocks, before and after the split."""
    rows = []
    for period, part in [("early", scored[scored["score_date"] < SPLIT_DATE]), ("late", scored[scored["score_date"] >= SPLIT_DATE])]:
        for name, subset in [("top", part[part["group"] == TOP]), ("all", part)]:
            shares = subset.groupby("score_date")["sector"].value_counts(normalize=True).unstack(fill_value=0).mean()
            rows += [{"period": period, "set": name, "sector": s, "share": float(v)} for s, v in shares.items()]
    table = pd.DataFrame(rows).pivot_table(index="sector", columns=["period", "set"], values="share", fill_value=0)
    table = table.sort_values(("late", "top"), ascending=False)
    return [{"sector": s, **{f"{p}_{k}": round(float(row[(p, k)]), 4) for p, k in row.index}} for s, row in table.iterrows()]


def build(scored, prices):
    marks = month_marks(scored, prices)
    growth_by_mark = holding_growth(scored, prices, marks)
    n = len(marks)
    years = (n - 1) / 12
    rng = np.random.default_rng(7)
    selectors = {"top": lambda g: g == TOP, "bottom": lambda g: g == BOTTOM, "all": lambda g: np.ones(len(g), bool)}
    bench, risk_free = benchmark_curves(marks)
    spy = bench["SPY"]
    late = np.asarray(marks >= SPLIT_DATE)
    split = int(np.argmax(late))

    paths, simulated = {}, {}
    for name in ("top", "bottom"):
        values = portfolio_values(growth_by_mark, n, selectors[name], rng, SIM_PATHS, PICK_FRACTION)
        paths[name] = values
        simulated[name] = {
            "fan": np.percentile(values, FAN_PERCENTILES, axis=0).round(4).tolist(),
            "stats": summarize(metrics(values, spy, risk_free, years)),
        }

    full = {}
    for name, members in selectors.items():
        values = portfolio_values(growth_by_mark, n, members, rng)[0]
        by_month = []
        for s in range(SLEEVES):
            single = portfolio_values(growth_by_mark, n, members, rng, sleeves=[s])[0]
            by_month.append({"month": marks[s * HOLD_MONTHS // SLEEVES].strftime("%B"), "curve": single.round(4).tolist(),
                             "cagr": float(single[-1] ** (1 / years) - 1)})
        full[name] = {"curve": values.round(4).tolist(), "cagr": float(values[-1] ** (1 / years) - 1), "by_month": by_month}
    curves = {name: np.array(f["curve"]) for name, f in full.items()}

    windows = []
    for span in WINDOW_YEARS:
        months = span * 12
        growth = {name: (c[months:] / c[:-months]) ** (1 / span) - 1 for name, c in {**curves, "spy": spy}.items()}
        gap = growth["top"] - growth["bottom"]
        # Start on quarter ends only, as the score dates do, so the bars stay readable.
        quarterly = slice(None, None, HOLD_MONTHS // SLEEVES)
        windows.append({
            "years": span, "count": int(len(gap[quarterly])),
            "top_beats_bottom_pct": float((gap[quarterly] > 0).mean() * 100),
            "top_beats_all_pct": float((growth["top"] > growth["all"])[quarterly].mean() * 100),
            "gap_median": float(np.median(gap[quarterly])), "gap_worst": float(gap[quarterly].min()),
            "gap_best": float(gap[quarterly].max()),
            "gaps": [[str(d.date()), round(float(v), 4)] for d, v in zip(marks[quarterly], gap[quarterly])],
            "spy_gaps": [[str(d.date()), round(float(v), 4)] for d, v in
                         zip(marks[quarterly], (growth["top"] - growth["spy"])[quarterly])],
            "top_beats_spy_pct": float((growth["top"] > growth["spy"])[quarterly].mean() * 100),
        })

    # Against the S&P 500.
    top_paths = paths["top"]
    labels, top_years = calendar_years(top_paths, marks)
    _, spy_years = calendar_years(spy, marks)
    _, rsp_years = calendar_years(bench["RSP"], marks)
    _, full_years = calendar_years(curves["top"], marks)
    year_gap = top_years - spy_years
    top_monthly = curves["top"][1:] / curves["top"][:-1] - 1
    spy_monthly = spy[1:] / spy[:-1] - 1
    active = top_monthly - spy_monthly
    active_late = late[1:]
    histogram = {
        "edges": HISTOGRAM_EDGES.round(3).tolist(),
        "early": np.histogram(np.clip(active[~active_late], HISTOGRAM_EDGES[0], HISTOGRAM_EDGES[-1] - 1e-9), HISTOGRAM_EDGES)[0].tolist(),
        "late": np.histogram(np.clip(active[active_late], HISTOGRAM_EDGES[0], HISTOGRAM_EDGES[-1] - 1e-9), HISTOGRAM_EDGES)[0].tolist(),
        "early_mean": float(active[~active_late].mean()), "late_mean": float(active[active_late].mean()),
        "early_months": int((~active_late).sum()), "late_months": int(active_late.sum()),
        "early_beat_pct": float((active[~active_late] > 0).mean() * 100), "late_beat_pct": float((active[active_late] > 0).mean() * 100),
    }

    def half_cagr(curve, lo, hi):
        return float((curve[hi] / curve[lo]) ** (12 / (hi - lo)) - 1)

    halves = {}
    for label, lo, hi in [("early", 0, split), ("late", split, n - 1)]:
        top_half = (top_paths[:, hi] / top_paths[:, lo]) ** (12 / (hi - lo)) - 1
        halves[label] = {
            "start": str(marks[lo].date()), "end": str(marks[hi].date()),
            "top": [float(x) for x in np.percentile(top_half, [50, 5, 95])],
            "spy": half_cagr(spy, lo, hi), "rsp": half_cagr(bench["RSP"], lo, hi), "all": half_cagr(curves["all"], lo, hi),
            "beat_spy_pct": float((top_half > half_cagr(spy, lo, hi)).mean() * 100),
        }

    spy_vs = {
        "benchmarks": {name: {"curve": c.round(4).tolist(), "stats": summarize(metrics(c, spy, risk_free, years))}
                       for name, c in bench.items() if name in ("SPY", "RSP")},
        "all_stats": summarize(metrics(curves["all"], spy, risk_free, years)),
        "beat_spy_pct": float((top_paths[:, -1] > spy[-1]).mean() * 100),
        "beat_rsp_pct": float((top_paths[:, -1] > bench["RSP"][-1]).mean() * 100),
        "relative_fan": np.percentile(top_paths / spy, FAN_PERCENTILES, axis=0).round(4).tolist(),
        "relative_rsp": (bench["RSP"] / spy).round(4).tolist(),
        "drawdown": {"top": drawdowns(curves["top"]).round(4).tolist(), "spy": drawdowns(spy).round(4).tolist()},
        "years": [{"label": label, "gap": [float(x) for x in np.percentile(year_gap[:, i], [50, 5, 95])],
                   "top": float(np.median(top_years[:, i])), "spy": float(spy_years[0, i]), "rsp": float(rsp_years[0, i]),
                   "full_gap": float(full_years[0, i] - spy_years[0, i])}
                  for i, label in enumerate(labels)],
        "histogram": histogram,
        "halves": halves,
        "split": str(SPLIT_DATE.date()),
        "sectors": sector_mix(scored),
    }

    sizes = pd.Series({m: (g == TOP).sum() for m, (g, _) in growth_by_mark.items()})
    return {
        "dates": [str(d.date()) for d in marks],
        "years": years,
        "paths": SIM_PATHS,
        "pick_fraction": PICK_FRACTION,
        "percentiles": FAN_PERCENTILES,
        "group_size_median": int(sizes.median()),
        "simulated": simulated,
        "full": full,
        "windows": windows,
        "top_beats_bottom_pct": float((metrics(paths["top"], spy, risk_free, years)["cagr"]
                                       > metrics(paths["bottom"], spy, risk_free, years)["cagr"]).mean() * 100),
        "top_beats_all_pct": float((paths["top"][:, -1] > curves["all"][-1]).mean() * 100),
        "vs_spy": spy_vs,
    }
