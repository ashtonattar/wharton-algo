"""
country_sleeve.py - the foreign equity sleeve as country index ETFs, chosen and weighted by the country screener.

    countries  : the screener's top 22 (GDP >= $500bn, GDP per capita >= $10k, ranked by score), minus the U.S.
    selection  : walk down that list, keep the first N countries with an approved market-tracking ETF that has
                 12 months of prices on the rebalance date; log every country skipped and why
    ETF        : per country, best trailing 12-month total return (prices on or before asof only);
                 ties -> lower expense ratio, then the older fund
    weights    : score / sum(scores), capped per country (Config.country_cap, 40%); the excess goes to the
                 uncapped countries by score. If fewer countries than 1/cap qualify, the cap cannot hold with
                 countries alone, and what is left goes to a broad approved developed-markets fund (VEA)
These ETFs skip the stock funnel (gates A-C, Monte Carlo) completely.

LOOK-AHEAD: the screener is one 2026 snapshot. Any backtest that uses it before 2025 picks countries with
knowledge of how they turned out (see lookahead_warning).
"""
from __future__ import annotations

import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENER = os.path.join(HERE, "..", "Book 9(Country Screener).csv")
APPROVED = os.path.join(HERE, "approved_etfs.csv")
MIN_MONTHS = 13            # 13 month-end prices = 12 monthly returns = a trailing 12-month return


# =============================================================================== screener
def _read_rows(path):
    for enc in ("utf-8-sig", "cp1252"):        # Excel exports "Türkiye" in Windows encoding
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _num(x: str) -> float:
    return float(str(x).replace("$", "").replace(",", "").strip())


def _tables(path):
    """Header positions of the two side-by-side tables: left (full) and right (top-22 summary)."""
    rows = _read_rows(path)
    head = [h.strip() for h in rows[0]]
    left_end = head.index("GDP") + 1                       # the full table ends at its GDP column
    left = {h: i for i, h in enumerate(head[:left_end]) if h}
    right = {h: i + left_end for i, h in enumerate(head[left_end:]) if h}
    for need, tab in ((("Country", "ISO3", "Score", "GDP Per Capita", "GDP"), left), (("Rank", "Country", "Score"), right)):
        missing = [c for c in need if c not in tab]
        if missing:
            raise ValueError(f"screener columns missing: {missing}")
    return rows, left, right


def load_country_scores(path: str = SCREENER) -> pd.DataFrame:
    """The screener's full (left) table: country, iso3, score, gdp_per_capita, gdp ($bn). One row per country."""
    rows, L, _ = _tables(path)
    out = []
    for r in rows[1:]:
        cell = lambda c: r[L[c]].strip() if L[c] < len(r) else ""
        if not cell("Country"):
            continue                                       # blank row (or only the right-hand table)
        try:
            score = _num(cell("Score"))
        except ValueError:
            raise ValueError(f"screener: Score missing or not a number for {cell('Country')!r}: {cell('Score')!r}")
        out.append(dict(country=cell("Country"), iso3=cell("ISO3"), score=score,
                        gdp_per_capita=_num(cell("GDP Per Capita")) if cell("GDP Per Capita") else float("nan"),
                        gdp=_num(cell("GDP")) if cell("GDP") else float("nan")))
    df = pd.DataFrame(out)
    if df.country.duplicated().any():
        raise ValueError(f"screener: duplicate countries {list(df.country[df.country.duplicated()])}")
    return df


def load_screener_summary(path: str = SCREENER) -> pd.DataFrame:
    """The right-hand Rank/Country/Score summary (only used to check our ranking)."""
    rows, _, R = _tables(path)
    out = [dict(rank=int(r[R["Rank"]]), country=r[R["Country"]].strip(), score=_num(r[R["Score"]]))
           for r in rows[1:] if len(r) > R["Rank"] and r[R["Rank"]].strip()]
    return pd.DataFrame(out)


def ranked_countries(scores: pd.DataFrame, plan) -> pd.DataFrame:
    """The screener's top list (size filter, then score, highest first), the domestic country included."""
    big = scores[(scores.gdp >= plan.country_min_gdp_bn) & (scores.gdp_per_capita >= plan.country_min_gdp_per_capita)]
    return big.sort_values("score", ascending=False, kind="stable").head(plan.country_universe_size).reset_index(drop=True)


# =============================================================================== ETFs
_CACHE = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def load_approved(path: str = APPROVED) -> pd.DataFrame:
    return _cached(("approved", path), lambda: pd.read_csv(path, dtype={"notes": str}).fillna({"notes": ""}))


def etf_prices(folder: str) -> pd.DataFrame:
    f = os.path.join(folder, "country_etfs.csv")
    return _cached(("px", f), lambda: pd.read_csv(f, index_col=0, parse_dates=True) if os.path.exists(f) else pd.DataFrame())


def pick_etf(country: str, asof, approved: pd.DataFrame, prices: pd.DataFrame):
    """(chosen row dict, None) or (None, reason). Only prices on or before `asof` are looked at."""
    asof = pd.Timestamp(asof)
    cands = approved[(approved.market_tracking == "Y") & (approved.target_country == country)]
    if cands.empty:
        return None, "no approved market-tracking ETF"
    ok, why = [], []
    for r in cands.itertuples():
        p = prices[r.ticker].loc[:asof].dropna() if r.ticker in prices.columns else pd.Series(dtype=float)
        if p.empty or p.index[-1] < asof - pd.offsets.MonthEnd(1):
            why.append(f"{r.ticker}: no price at {asof.date()}")
        elif len(p) < MIN_MONTHS:
            why.append(f"{r.ticker}: only {len(p) - 1} months of prices")
        else:
            ok.append(dict(ticker=r.ticker, name=r.name, ret_12m=float(p.iloc[-1] / p.iloc[-MIN_MONTHS] - 1),
                           expense_ratio=getattr(r, "expense_ratio", float("nan")), first=p.index[0], notes=r.notes))
    if not ok:
        return None, "; ".join(why)
    er = lambda x: x["expense_ratio"] if pd.notna(x["expense_ratio"]) else float("inf")
    return sorted(ok, key=lambda x: (-round(x["ret_12m"], 10), er(x), x["first"]))[0], None


# =============================================================================== weights
def capped_score_weights(scores: dict, cap: float) -> tuple:
    """score-proportional weights, each <= cap; the excess is spread over the uncapped by score until nothing is
    over. Returns (weights, leftover): leftover > 0 only when len(scores) * cap < 1 (every country at the cap)."""
    if any(s <= 0 for s in scores.values()):
        raise ValueError(f"scores must be positive: {scores}")
    w, fixed, room = {}, set(), 1.0
    while True:
        free = {k: s for k, s in scores.items() if k not in fixed}
        if not free:
            break
        tot = sum(free.values())
        trial = {k: room * s / tot for k, s in free.items()}
        over = {k for k, x in trial.items() if x > cap + 1e-12}
        if not over:
            w.update(trial)
            room = 0.0
            break
        for k in over:
            w[k] = cap
            fixed.add(k)
            room -= cap
    return w, max(0.0, room)


# =============================================================================== the sleeve
def country_sleeve(folder: str, asof, plan, cap: float) -> dict:
    """Weights inside the foreign sleeve (sum to 1, or {} if no country qualifies), plus a report."""
    scores = _cached(("scores", plan.country_screener), lambda: load_country_scores(plan.country_screener))
    approved, prices = load_approved(plan.approved_etfs), etf_prices(folder)
    picked, skipped = [], []
    for r in ranked_countries(scores, plan).itertuples():
        if r.country == plan.domestic_country:
            continue
        if len(picked) >= plan.n_foreign_countries:
            break
        if r.score <= 0:
            skipped.append(dict(country=r.country, score=r.score, reason="score not positive"))
            continue
        etf, why = pick_etf(r.country, asof, approved, prices)
        if etf is None:
            skipped.append(dict(country=r.country, score=r.score, reason=why))
        else:
            picked.append(dict(country=r.country, score=r.score, **etf))
    if not picked:
        return dict(weights={}, rows=[], skipped=skipped, filler=0.0)
    cw, left = capped_score_weights({p["country"]: p["score"] for p in picked}, cap)
    weights = {}
    for p in picked:
        p["weight"] = cw[p["country"]]
        weights[p["ticker"]] = weights.get(p["ticker"], 0.0) + p["weight"]
    filler = 0.0
    if left > 1e-12:
        fp = prices[plan.foreign_filler].loc[:pd.Timestamp(asof)].dropna() if plan.foreign_filler in prices.columns else []
        if len(fp):
            weights[plan.foreign_filler] = weights.get(plan.foreign_filler, 0.0) + left
            filler = left
        else:                                              # no filler price: let the countries go over the cap
            weights = {t: x / (1 - left) for t, x in weights.items()}
            for p in picked:
                p["weight"] /= (1 - left)
    return dict(weights=weights, rows=picked, skipped=skipped, filler=filler)


def lookahead_warning(plan, asof) -> str | None:
    if plan.foreign_mode == "country_etf" and plan.country_scores_static and pd.Timestamp(asof) < pd.Timestamp("2025-01-01"):
        return (f"WARNING look-ahead: foreign countries chosen with the 2026 screener snapshot on {pd.Timestamp(asof).date()}; "
                "country selection before 2025 is not point-in-time.")
    return None
