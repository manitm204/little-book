"""Does the Magic Formula score separate winners from losers better in some sectors than others?

Every quarter, each sector's scored stocks are split into four equal groups by the same score the
rest of the dashboard uses, and each group's average return over the next year is measured. The
gap between a sector's top and bottom quarter, averaged over quarters, is that sector's result.

Sectors average 16 to 61 scored stocks a quarter, so the small ones have only a handful of stocks
per group and are flagged. Testing eight sectors at once also means one is likely to look good by
chance, so each interval is shown both at 90% and widened for the number of sectors (Bonferroni).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import backtest

GROUPS = 4
SMALL_SECTOR = 25
CONFIDENCE = 0.90
SPLIT_DATE = backtest.SPLIT_DATE
ALL = "All sectors"
# Sectors where the gap looked positive, examined start date by start date, with their sector ETFs:
# cap-weighted (SPDR) and equal-weight (Invesco, from November 2006).
FOCUS = ["Healthcare", "Consumer Cyclical"]
SECTOR_ETFS = {"Healthcare": ("XLV", "RSPH"), "Consumer Cyclical": ("XLY", "RSPD")}
# Start-date blocks for the "where did the edge come from" table, as (first year, last year).
PERIODS = [(2005, 2008), (2009, 2012), (2013, 2016), (2017, 2020), (2021, 2026)]
ROLLING_QUARTERS = 12
# A 1-year return starts every quarter, so neighboring gaps share nine months.
TREND_LAGS = 4
INDEX = "Whole index"
# Mixes of long the top 25% and short the bottom 25% in one sector, as shares of the money:
# (key, label, long, short). The rest of the money, and the cash from the short sale, earns T-bills.
MIX_SECTOR = "Healthcare"
MIXES = [("long", "100% long", 1.0, 0.0), ("m75", "75% long / 25% short", 0.75, 0.25),
         ("m50", "50% long / 50% short", 0.5, 0.5), ("ls", "100% long / 100% short", 1.0, 1.0)]
# A portfolio split between one of the mixes and SPY, reset to these shares every month.
BLEND_MIX, BLEND_SHARE = "m75", 0.5
BLEND_WITH = ["SPY", "QQQ"]
# Cost scenarios: (key, label, yearly fee to borrow the shorted stocks, cost per dollar traded one way).
# The trading cost stands for spread plus market impact on S&P 500 stocks.
COST_SCENARIOS = [("none", "No costs", 0.0, 0.0), ("low", "Low", 0.005, 0.0005),
                  ("medium", "Medium", 0.015, 0.0010), ("high", "High", 0.03, 0.0025)]
COST_GRID_BORROW = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
COST_GRID_TRADING = [0.0, 0.0005, 0.0010, 0.0025, 0.0050]


def add_sector_groups(scored):
    """Score order within each sector and date: group 1 is the best quarter, 4 the worst."""
    by = scored.groupby(["score_date", "sector"])["final_rank"]
    order = by.rank(method="first")
    count = by.transform("count")
    scored["sector_group"] = np.ceil(order / count * GROUPS).astype(int)
    scored["sector_score"] = 100 * (1 - (order - 1) / (count - 1).clip(lower=1))
    by_date = scored.groupby("score_date")["final_rank"]
    scored["index_group"] = np.ceil(by_date.rank(method="first") / by_date.transform("count") * GROUPS).astype(int)
    return scored


def gap_summary(per_date, interval, adjusted_level):
    """Top minus bottom group, one observation per quarter, with Newey-West intervals."""
    spread = (per_date[1] - per_date[GROUPS]).dropna()
    low, high, se = interval(spread, CONFIDENCE)
    adj_low, adj_high, _ = interval(spread, adjusted_level)
    return spread, {
        "gap": float(spread.mean()), "ci": [float(low), float(high)], "ci_adjusted": [float(adj_low), float(adj_high)],
        "t_stat": float(spread.mean() / se), "quarters": int(len(spread)), "top_wins_pct": float((spread > 0).mean() * 100),
    }


def sector_curves(scored, prices, sectors):
    """Growth of $1 in each sector's top and bottom group and in the whole sector, staggered yearly rebalance."""
    marks = backtest.month_marks(scored, prices)
    labelled = scored.assign(group=scored["sector"] + "|" + scored["sector_group"].astype(str))
    growth = backtest.holding_growth(labelled, prices, marks)
    rng = np.random.default_rng(7)
    years = (len(marks) - 1) / 12
    curves = {}
    for sector in sectors:
        def members(label):
            return lambda g: g == f"{sector}|{label}"
        whole = lambda g: np.char.startswith(g.astype(str), sector + "|")
        lines = {str(g): backtest.portfolio_values(growth, len(marks), members(g), rng)[0] for g in range(1, GROUPS + 1)}
        lines["all"] = backtest.portfolio_values(growth, len(marks), whole, rng)[0]
        lines["top"], lines["bottom"] = lines["1"], lines[str(GROUPS)]
        curves[sector] = {name: {"curve": v.round(4).tolist(), "cagr": float(v[-1] ** (1 / years) - 1),
                                 "max_drawdown": float(backtest.drawdowns(v).min()),
                                 "volatility": float(np.std(v[1:] / v[:-1] - 1, ddof=1) * np.sqrt(12))}
                          for name, v in lines.items()}
    return marks, growth, curves


def versus_benchmarks(sector, marks, growth, bench, risk_free):
    """The sector's top group, with random half picks as in the backtest, against its sector ETFs and the S&P 500."""
    cap_symbol, equal_symbol = SECTOR_ETFS[sector]
    rng = np.random.default_rng(11)
    n = len(marks)
    years = (n - 1) / 12
    paths = backtest.portfolio_values(growth, n, lambda g: g == f"{sector}|1", rng, backtest.SIM_PATHS, backtest.PICK_FRACTION)
    spy, cap, equal = bench["SPY"], bench[cap_symbol], bench[equal_symbol]
    vs_spy = backtest.metrics(paths, spy, risk_free, years)
    vs_cap = backtest.metrics(paths, cap, risk_free, years)
    split = int(np.argmax(np.asarray(marks >= SPLIT_DATE)))

    def cagr(values, lo=0, hi=n - 1):
        values = np.atleast_2d(values)
        return (values[:, hi] / values[:, lo]) ** (12 / (hi - lo)) - 1

    def spread(values):
        return [float(x) for x in np.percentile(values, [50, 5, 95])]

    # The equal-weight ETF starts later, so it is compared over its own window, with the others re-measured over it.
    equal_start = int(np.argmax(~np.isnan(equal)))
    since = {
        "start": str(marks[equal_start].date()),
        "top": spread(cagr(paths, equal_start)), "cap": float(cagr(cap, equal_start)[0]),
        "equal": float(cagr(equal, equal_start)[0]), "spy": float(cagr(spy, equal_start)[0]),
        "beat_equal_pct": float((cagr(paths, equal_start) > cagr(equal, equal_start)[0]).mean() * 100),
    }
    labels, top_years = backtest.calendar_years(paths, marks)
    _, cap_years = backtest.calendar_years(cap, marks)
    _, spy_years = backtest.calendar_years(spy, marks)
    halves = {}
    for name, lo, hi in [("early", 0, split), ("late", split, n - 1)]:
        top_half = cagr(paths, lo, hi)
        halves[name] = {"start": str(marks[lo].date()), "end": str(marks[hi].date()), "top": spread(top_half),
                        "cap": float(cagr(cap, lo, hi)[0]), "spy": float(cagr(spy, lo, hi)[0]),
                        "beat_cap_pct": float((top_half > cagr(cap, lo, hi)[0]).mean() * 100),
                        "beat_spy_pct": float((top_half > cagr(spy, lo, hi)[0]).mean() * 100)}
    stats = backtest.summarize(vs_spy)
    for key in ("alpha", "beta", "tracking_error", "information_ratio", "months_beating", "up_capture", "down_capture", "r_squared"):
        stats[f"{key}_vs_etf"] = backtest.summarize({key: vs_cap[key]})[key]

    def benchmark_stats(curve):
        measured = backtest.summarize(backtest.metrics(curve, spy, risk_free, years))
        against_cap = backtest.summarize(backtest.metrics(curve, cap, risk_free, years))
        for key in ("alpha", "beta", "tracking_error", "information_ratio", "months_beating", "up_capture", "down_capture", "r_squared"):
            measured[f"{key}_vs_etf"] = against_cap[key]
        return measured

    bottom = backtest.portfolio_values(growth, n, lambda g: g == f"{sector}|{GROUPS}", rng, backtest.SIM_PATHS, backtest.PICK_FRACTION)
    strategies = []
    for key, name, long_side, short_side, hedge in [
        ("long", f"Long top 25% only", paths, None, None),
        ("ls_dollar", "Long top, short bottom 25%", paths, bottom, "dollar"),
        ("ls_beta", "Long top, short bottom 25%, beta-neutral", paths, bottom, "beta"),
        ("etf_dollar", f"Long top, short {cap_symbol}", paths, cap, "dollar"),
        ("etf_beta", f"Long top, short {cap_symbol}, beta-neutral", paths, cap, "beta"),
    ]:
        if short_side is None:
            values, ratio = paths, np.ones((len(paths), n - 1))
        else:
            values, ratio = backtest.long_short(long_side, short_side, spy, risk_free, hedge)
        measures = backtest.metrics(values, spy, risk_free, years)
        strategies.append({
            "key": key, "name": name,
            "stats": backtest.summarize(measures),
            "fan": np.percentile(values, [5, 50, 95], axis=0).round(4).tolist(),
            "hedge": spread(ratio.mean(axis=1)),
            "early": spread(cagr(values, 0, split)), "late": spread(cagr(values, split, n - 1)),
            "positive_pct": float((values[:, -1] > 1).mean() * 100),
            "beat_cash_pct": float((measures["sharpe"] > 0).mean() * 100),
        })

    return {
        "cap_symbol": cap_symbol, "equal_symbol": equal_symbol, "strategies": strategies,
        "fan": np.percentile(paths, backtest.FAN_PERCENTILES, axis=0).round(4).tolist(),
        "cap_curve": cap.round(4).tolist(), "spy_curve": spy.round(4).tolist(),
        # The equal-weight ETF joins the chart at the cap-weighted ETF's value on its first month end.
        "equal_curve": [None if np.isnan(v) else round(float(v * cap[equal_start]), 4) for v in equal],
        "stats": stats, "cap_stats": benchmark_stats(cap), "spy_stats": benchmark_stats(spy),
        "beat_cap_pct": float((paths[:, -1] > cap[-1]).mean() * 100),
        "beat_spy_pct": float((paths[:, -1] > spy[-1]).mean() * 100),
        "relative_cap": np.percentile(paths / cap, backtest.FAN_PERCENTILES, axis=0).round(4).tolist(),
        "relative_spy": np.percentile(paths / spy, backtest.FAN_PERCENTILES, axis=0).round(4).tolist(),
        "years": [{"label": label, "top": float(np.median(top_years[:, i])), "cap": float(cap_years[0, i]), "spy": float(spy_years[0, i]),
                   "gap_cap": spread(top_years[:, i] - cap_years[0, i]), "gap_spy": spread(top_years[:, i] - spy_years[0, i])}
                  for i, label in enumerate(labels)],
        "halves": halves,
        "since_equal": since,
    }


def path_stats(values, marks, spy, etf, risk_free, years, split):
    """Every measure for each path of `values`, against the S&P 500 and against the sector ETF."""
    values = np.atleast_2d(values)
    measures = backtest.metrics(values, spy, risk_free, years)
    vs_etf = backtest.metrics(values, etf, risk_free, years)
    r = values[:, 1:] / values[:, :-1] - 1
    m = spy[1:] / spy[:-1] - 1
    excess, market_excess = r - risk_free, m - risk_free
    # Alpha's t-statistic from the monthly return left after the market part, Newey-West.
    residual = excess - measures["beta"][:, None] * market_excess
    dd = backtest.drawdowns(values)
    underwater = np.zeros(len(values)); run = np.zeros(len(values))
    for t in range(dd.shape[1]):
        run = np.where(dd[:, t] < -1e-9, run + 1, 0)
        underwater = np.maximum(underwater, run)
    down = m < 0
    mm = m[down] - m[down].mean()
    labels, by_year = backtest.calendar_years(values, marks)
    _, spy_years = backtest.calendar_years(spy, marks)
    full_years = [i for i, label in enumerate(labels) if "(" not in label]
    by_year, spy_full = by_year[:, full_years], spy_years[:, full_years]
    centered = r - r.mean(axis=1, keepdims=True)

    def cagr(lo, hi):
        return (values[:, hi] / values[:, lo]) ** (12 / (hi - lo)) - 1

    n = values.shape[1]
    return {
        **measures,
        "alpha_t": backtest.newey_west_t(residual),
        "calmar": measures["cagr"] / -measures["max_drawdown"],
        "underwater_months": underwater,
        "worst_month": r.min(axis=1), "best_month": r.max(axis=1),
        "worst_year": by_year.min(axis=1), "best_year": by_year.max(axis=1),
        "years_positive": (by_year > 0).mean(axis=1), "years_beating_spy": (by_year > spy_full).mean(axis=1),
        "skew": (centered ** 3).mean(axis=1) / centered.std(axis=1) ** 3,
        "downside_beta": (r[:, down] - r[:, down].mean(axis=1, keepdims=True)) @ mm / (mm @ mm),
        "beta_etf": vs_etf["beta"], "correlation_etf": vs_etf["correlation"], "alpha_etf": vs_etf["alpha"],
        "early": cagr(0, split), "late": cagr(split, n - 1),
    }


def with_costs(weights, returns, risk_free, borrow, trading):
    """Monthly returns after costs for a portfolio reset to `weights` every month end.

    `weights` and `returns` are dicts by component: "top" and "bottom" are the sleeve portfolios (a
    negative weight is a short), anything else is a fund. The rest of the money is in T-bills. Trading
    counts the monthly reset back to the weights, and the stock sleeves' yearly rebalances, one sleeve
    each quarter, treated as selling the whole sleeve and buying a new one (an upper bound, since some
    stocks stay in). Borrowing is charged on the short weight.
    Returns (net monthly returns, gross monthly returns, dollars traded per $1 each month)."""
    shape = next(iter(returns.values())).shape
    gross = risk_free + sum(w * (returns[k] - risk_free) for k, w in weights.items())
    traded = np.zeros(shape)
    for k, w in weights.items():
        drifted = w * (1 + returns[k]) / (1 + gross)
        traded += np.abs(drifted - w)
    # The first purchase, then a quarter of each stock side sold and rebought every three months.
    traded[:, 0] += sum(abs(w) for w in weights.values())
    stock_side = sum(abs(weights.get(k, 0)) for k in ("top", "bottom"))
    rotation = np.array([t % (backtest.HOLD_MONTHS // backtest.SLEEVES) == 0 and t > 0 for t in range(shape[1])])
    traded[:, rotation] += 2 * stock_side / backtest.SLEEVES
    short = sum(-w for w in weights.values() if w < 0)
    net = gross - short * borrow / 12 - trading * traded
    return net, gross, traded


def grow(r):
    return np.concatenate([np.ones((len(r), 1)), np.cumprod(1 + r, axis=1)], axis=1)


def cost_tests(marks, returns, strategies, spy, qqq, risk_free, split):
    """Every strategy under each cost scenario, plus a grid of borrow fee against trading cost."""
    n = len(marks)
    years = (n - 1) / 12
    out = []
    for key, name, weights in strategies:
        scenarios = []
        for s_key, s_name, borrow, trading in COST_SCENARIOS:
            net, gross, traded = with_costs(weights, returns, risk_free, borrow, trading)
            values = grow(net)
            stats = backtest.summarize(path_stats(values, marks, spy, qqq, risk_free, years, split))
            gross_cagr = grow(gross)[:, -1] ** (1 / years) - 1
            stats["cost_drag"] = backtest.summarize({"x": gross_cagr - (values[:, -1] ** (1 / years) - 1)})["x"]
            # Ongoing turnover, leaving out the first purchase.
            stats["turnover"] = backtest.summarize({"x": traded[:, 1:].sum(axis=1) / years})["x"]
            scenarios.append({"key": s_key, "name": s_name, "borrow": borrow, "trading": trading, "stats": stats,
                              "curve": np.median(values, axis=0).round(4).tolist()})
        grid = []
        for borrow in COST_GRID_BORROW:
            row = []
            for trading in COST_GRID_TRADING:
                net, _, _ = with_costs(weights, returns, risk_free, borrow, trading)
                measured = backtest.metrics(grow(net), spy, risk_free, years)
                row.append({"alpha": float(np.median(measured["alpha"])), "sharpe": float(np.median(measured["sharpe"])),
                            "cagr": float(np.median(measured["cagr"]))})
            grid.append(row)
        short = sum(-w for w in weights.values() if w < 0)
        out.append({"key": key, "name": name, "short": short, "scenarios": scenarios, "grid": grid})
    return {"strategies": out, "grid_borrow": COST_GRID_BORROW, "grid_trading": COST_GRID_TRADING,
            "spy_curve": spy.round(4).tolist(), "spy_stats": backtest.summarize(path_stats(spy, marks, spy, qqq, risk_free, years, split))}


def mixes(sector, marks, growth, bench, risk_free):
    """Long the sector's top 25% and short its bottom 25% in several proportions, with random half picks as elsewhere."""
    cap_symbol, _ = SECTOR_ETFS[sector]
    rng = np.random.default_rng(13)
    n = len(marks)
    years = (n - 1) / 12
    split = int(np.argmax(np.asarray(marks >= SPLIT_DATE)))
    spy, etf = bench["SPY"], bench[cap_symbol]
    top = backtest.portfolio_values(growth, n, lambda g: g == f"{sector}|1", rng, backtest.SIM_PATHS, backtest.PICK_FRACTION)
    bottom = backtest.portfolio_values(growth, n, lambda g: g == f"{sector}|{GROUPS}", rng, backtest.SIM_PATHS, backtest.PICK_FRACTION)
    long_r, short_r = top[:, 1:] / top[:, :-1] - 1, bottom[:, 1:] / bottom[:, :-1] - 1
    _, top_years = backtest.calendar_years(top, marks)
    labels, spy_years = backtest.calendar_years(spy, marks)
    blend_weights = next((lw, sw) for k, _, lw, sw in MIXES if k == BLEND_MIX)
    mix_r = risk_free + blend_weights[0] * (long_r - risk_free) - blend_weights[1] * (short_r - risk_free)
    rows = []
    for key, name, long_weight, short_weight in MIXES:
        # Rebalanced to the target weights every month.
        r = risk_free + long_weight * (long_r - risk_free) - short_weight * (short_r - risk_free)
        values = np.concatenate([np.ones((len(r), 1)), np.cumprod(1 + r, axis=1)], axis=1)
        _, years_r = backtest.calendar_years(values, marks)
        rows.append({"key": key, "name": name, "long": long_weight, "short": short_weight,
                     "stats": backtest.summarize(path_stats(values, marks, spy, etf, risk_free, years, split)),
                     "fan": np.percentile(values, [5, 50, 95], axis=0).round(4).tolist(),
                     "drawdown": np.median(backtest.drawdowns(values), axis=0).round(4).tolist(),
                     "years": [float(np.median(years_r[:, i])) for i in range(len(labels))],
                     "beat_spy_pct": float((values[:, -1] > spy[-1]).mean() * 100)})
    benchmarks = []
    for key, name, curve in [("spy", "S&P 500 (SPY)", spy), ("etf", f"Healthcare ETF ({cap_symbol})", etf)]:
        _, years_b = backtest.calendar_years(curve, marks)
        benchmarks.append({"key": key, "name": name, "curve": curve.round(4).tolist(),
                           "stats": backtest.summarize(path_stats(curve, marks, spy, etf, risk_free, years, split)),
                           "drawdown": backtest.drawdowns(curve).round(4).tolist(), "years": years_b[0].tolist()})
    # Half the 75/25 mix and half an index fund, reset every month. Measures marked "_etf" are against that fund.
    blends = {}
    for symbol in BLEND_WITH:
        fund = bench[symbol]
        r = BLEND_SHARE * mix_r + (1 - BLEND_SHARE) * (fund[1:] / fund[:-1] - 1)
        values = np.concatenate([np.ones((len(r), 1)), np.cumprod(1 + r, axis=1)], axis=1)
        blends[symbol] = {
            "symbol": symbol, "name": f"{BLEND_SHARE:.0%} 75/25 + {1 - BLEND_SHARE:.0%} {symbol}",
            "stats": backtest.summarize(path_stats(values, marks, spy, fund, risk_free, years, split)),
            "fan": np.percentile(values, [5, 50, 95], axis=0).round(4).tolist(),
            "drawdown": np.median(backtest.drawdowns(values), axis=0).round(4).tolist(),
            "mix_stats": backtest.summarize(path_stats(np.concatenate([np.ones((len(mix_r), 1)), np.cumprod(1 + mix_r, axis=1)], axis=1),
                                                       marks, spy, fund, risk_free, years, split)),
            "fund": {"curve": fund.round(4).tolist(), "drawdown": backtest.drawdowns(fund).round(4).tolist(),
                     "stats": backtest.summarize(path_stats(fund, marks, spy, fund, risk_free, years, split))},
        }
    monthly = lambda c: np.broadcast_to(c[1:] / c[:-1] - 1, long_r.shape)
    returns = {"top": long_r, "bottom": short_r, "SPY": monthly(spy), "QQQ": monthly(bench["QQQ"])}
    lw, sw = blend_weights
    strategies = [(key, name, {"top": l, "bottom": -sh} if sh else {"top": l}) for key, name, l, sh in MIXES]
    for symbol in BLEND_WITH:
        strategies.append((f"blend_{symbol}", blends[symbol]["name"],
                           {"top": BLEND_SHARE * lw, "bottom": -BLEND_SHARE * sw, symbol: 1 - BLEND_SHARE}))
    costs = cost_tests(marks, returns, strategies, spy, bench["QQQ"], risk_free, split)
    return {"sector": sector, "cap_symbol": cap_symbol, "mixes": rows, "blends": blends, "costs": costs, "blend_share": BLEND_SHARE, "benchmarks": benchmarks, "year_labels": labels,
            "split": str(SPLIT_DATE.date())}


def trend(spread):
    """Change in the gap per decade of start dates, with errors that allow for the overlap."""
    years = np.asarray((spread.index - spread.index[0]).days / 365.25)
    fit = sm.OLS(spread.to_numpy(), sm.add_constant(years)).fit(cov_type="HAC", cov_kwds={"maxlags": TREND_LAGS})
    low, high = fit.conf_int(alpha=1 - CONFIDENCE)[1]
    return {"per_decade": float(fit.params[1] * 10), "ci": [float(low * 10), float(high * 10)], "p_value": float(fit.pvalues[1]),
            "start": float(fit.params[0]), "end": float(fit.params[0] + fit.params[1] * years[-1])}


def focus(sector, returns, scored, curves, interval):
    """Everything for one sector, start date by start date."""
    part = returns[returns["sector"] == sector]
    per_date = part.groupby(["score_date", "sector_group"])["forward_return"].mean().unstack()
    spread = (per_date[1] - per_date[GROUPS]).dropna()
    sector_mean = per_date.mean(axis=1)
    counts = scored[scored["sector"] == sector].groupby("score_date").size()
    groups = []
    for g in range(1, GROUPS + 1):
        rows = part[part["sector_group"] == g]
        curve = curves[sector][str(g)]
        groups.append({
            "group": g, "mean_return": float(per_date[g].mean()), "median_stock": float(rows["forward_return"].median()),
            "beat_sector_pct": float((per_date[g] > sector_mean).mean() * 100),
            "stocks_per_quarter": float(rows.groupby("score_date").size().mean()), "stock_quarters": int(len(rows)),
            "cagr": curve["cagr"], "max_drawdown": curve["max_drawdown"], "volatility": curve["volatility"],
        })
    periods = []
    for first, last in PERIODS:
        mask = (spread.index.year >= first) & (spread.index.year <= last)
        chunk = spread[mask]
        if len(chunk) < 3:
            continue
        low, high, _ = interval(chunk, CONFIDENCE)
        periods.append({"label": f"{first}–{min(last, spread.index.max().year)}", "gap": float(chunk.mean()),
                        "ci": [float(low), float(high)], "quarters": int(len(chunk)),
                        "top_wins_pct": float((chunk > 0).mean() * 100),
                        "top": float(per_date.loc[chunk.index, 1].mean()), "bottom": float(per_date.loc[chunk.index, GROUPS].mean())})
    rolling = spread.rolling(ROLLING_QUARTERS).mean()
    return {
        "sector": sector,
        "counts": [[str(d.date()), int(n)] for d, n in counts.items()],
        "groups": groups,
        "by_date": [[str(d.date()), *[round(float(per_date.loc[d, g]), 4) for g in range(1, GROUPS + 1)], round(float(v), 4),
                     None if np.isnan(rolling[d]) else round(float(rolling[d]), 4)] for d, v in spread.items()],
        "periods": periods,
        "trend": trend(spread),
        "best_quarters": [[str(d.date()), round(float(v), 4)] for d, v in spread.nlargest(5).items()],
        # How much of the total gap the five best start dates account for.
        "gap_without_best5": float(spread.drop(spread.nlargest(5).index).mean()),
        "gap_without_best5_pct": float(spread.drop(spread.nlargest(5).index).mean() / spread.mean() * 100),
    }


def build(returns, scored, prices, interval):
    """`returns` are next-year returns per stock and quarter. `interval(series, level)` gives (low, high, se)."""
    sectors = sorted(returns["sector"].dropna().unique())
    adjusted_level = 1 - (1 - CONFIDENCE) / len(sectors)
    sizes = scored.groupby(["sector", "score_date"]).size().groupby("sector").mean()
    rows = []
    for sector in [*sectors, ALL, INDEX]:
        part = returns if sector in (ALL, INDEX) else returns[returns["sector"] == sector]
        if sector == ALL:
            # Each sector's groups pooled with equal weight: the top quarter of every sector, held together.
            per_date = part.groupby(["score_date", "sector", "sector_group"])["forward_return"].mean() \
                .groupby(["score_date", "sector_group"]).mean().unstack()
        elif sector == INDEX:
            # The whole index split into quarters regardless of sector, so sector tilts count too.
            per_date = part.groupby(["score_date", "index_group"])["forward_return"].mean().unstack()
        else:
            per_date = part.groupby(["score_date", "sector_group"])["forward_return"].mean().unstack()
        spread, summary = gap_summary(per_date, interval, adjusted_level)
        sector_mean = per_date.mean(axis=1)
        score, by = ("score", ["score_date"]) if sector == INDEX else ("sector_score", ["score_date", "sector"])
        correlation = part.groupby(by).apply(
            lambda d: d[score].corr(d["forward_return"], method="spearman"), include_groups=False).dropna()
        halves = {}
        for name, mask in [("early", spread.index < SPLIT_DATE), ("late", spread.index >= SPLIT_DATE)]:
            low, high, _ = interval(spread[mask], CONFIDENCE)
            halves[name] = {"gap": float(spread[mask].mean()), "ci": [float(low), float(high)], "quarters": int(mask.sum())}
        rows.append({
            "sector": sector,
            "stocks": float(sizes.sum() if sector in (ALL, INDEX) else sizes[sector]),
            "small": bool(sector in sizes and sizes[sector] < SMALL_SECTOR),
            "reference": sector in (ALL, INDEX),
            "groups": per_date.mean().round(5).tolist(),
            "relative": per_date.sub(sector_mean, axis=0).mean().round(5).tolist(),
            "sector_mean": float(sector_mean.mean()),
            "rank_correlation": float(correlation.mean()),
            "halves": halves,
            **summary,
        })
    rows.sort(key=lambda r: (r["reference"], -r["gap"]))
    marks, growth, curves = sector_curves(scored, prices, sectors)
    bench, risk_free = backtest.benchmark_curves(marks)
    return {
        "rows": rows, "groups": GROUPS, "small_threshold": SMALL_SECTOR, "confidence": CONFIDENCE,
        "adjusted_level": adjusted_level, "sector_count": len(sectors), "split": str(SPLIT_DATE.date()),
        "dates": [str(d.date()) for d in marks], "curves": curves, "rolling_quarters": ROLLING_QUARTERS,
        "focus": [{**focus(sector, returns, scored, curves, interval),
                   "versus": versus_benchmarks(sector, marks, growth, bench, risk_free)} for sector in FOCUS],
        "mixes": mixes(MIX_SECTOR, marks, growth, bench, risk_free),
    }
