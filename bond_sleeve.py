"""
bond_sleeve.py - the bond share of an account as specific approved Treasuries, chosen by the bond workbook's scores.

    data      : approved_bonds.csv + bond_settings.json (build_approved_bonds.py, from "(C) Bond Model (Fixed).xlsx").
                One snapshot: every metric is computed at the settlement date (2026-10-23)
    metrics   : the workbook's formulas (Excel PRICE / DURATION / COUP* with actual/actual day count), recomputed here
    scores    : WITHIN-BUCKET (same bucket + currency, 0.5+ yrs left): 40% return per unit of duration, 30% 1-yr return,
                15% roll-down, 15% (lower) default risk, each a percentile in the group.
                ACROSS-MATURITIES (same currency, has excess return / duration): percentile of excess return per unit of duration
    growth / pre-gift / post-gift : best within-bucket bond per maturity bucket, kept if its modified duration is under the
                phase's limit min(years to phase end, bond_max_duration) (floor 1.0); weights = across score + 1, capped
    reserve   : a liability-matching ladder for the $50k payments on Jan 1 2033-2042: the bond for Jan 1 of year Y
                matures in Y-1; face sized backward with 0% reinvestment; payments with no such bond are a gap, covered by
                the latest bond that matures before the gap (its proceeds held in T-bills)

LIVE-ONLY: single bonds have no price history, so backtests keep IEF as the bond return proxy.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import pandas as pd

from country_sleeve import capped_score_weights
from funnel_v2 import CASH

HERE = os.path.dirname(os.path.abspath(__file__))
BONDS = os.path.join(HERE, "approved_bonds.csv")
SETTINGS = "bond_settings.json"                # next to the bond csv
LIVE_NOTE = "bond picks are live-only; backtests hold IEF as the proxy"
FACE_LOT = 1_000.0                             # face rounding (check against StockTrak's lot rules)


@dataclass
class BondSettings:
    settle: str = "2026-10-23"                 # Settings B1: yields as of
    rf_usd: float | None = 0.0444              # B3-B5: 1-yr risk-free by currency (blank = no across score)
    rf_gbp: float | None = None
    rf_eur: float | None = None
    recovery: float = 0.40                     # B6
    w_ret_per_dur: float = 0.40                # B7-B10: within-bucket weights
    w_ret: float = 0.30
    w_roll: float = 0.15
    w_default: float = 0.15
    cut1: float = 3.0                          # B11-B13: bucket cutoffs (years left)
    cut2: float = 7.0
    cut3: float = 10.0
    horizon_months: int = 12                   # B2 = EDATE(B1, 12)
    min_years_within: float = 0.5              # col AB: scored within bucket if 0.5+ years left
    min_years_across: float = 1.0              # col AA: excess return per duration needs 1+ year left
    lone_score: float = 50.0                   # a group of one scores 50
    stale_days: int = 14                       # warn when asof is further than this from settle

    @classmethod
    def load(cls, bond_csv: str = BONDS) -> "BondSettings":
        f = os.path.join(os.path.dirname(os.path.abspath(bond_csv)), SETTINGS)
        return cls(**json.load(open(f))) if os.path.exists(f) else cls()

    def rf(self, ccy: str):
        return {"USD": self.rf_usd, "GBP": self.rf_gbp, "EUR": self.rf_eur}.get(ccy)


# =============================================================================== Excel bond functions (basis 1)
def _ts(d):
    return pd.Timestamp(d).normalize()


def edate(d, months: int):
    return _ts(d) + pd.DateOffset(months=months)


def _coupon_date(mat, k: int, freq: int):
    """k-th coupon date before maturity (Excel steps back whole months; month-end maturities stay month-end)."""
    d = mat - pd.DateOffset(months=k * 12 // freq)
    return d + pd.offsets.MonthEnd(0) if mat.is_month_end else d


def _pcd_ncd(settle, mat, freq):
    settle, mat = _ts(settle), _ts(mat)
    k = 0
    while _coupon_date(mat, k + 1, freq) > settle:
        k += 1
    return _coupon_date(mat, k + 1, freq), _coupon_date(mat, k, freq), k + 1      # PCD, NCD, COUPNUM


def coupnum(settle, mat, freq) -> int:
    return _pcd_ncd(settle, mat, freq)[2]


def coupdaybs(settle, mat, freq) -> int:
    return (_ts(settle) - _pcd_ncd(settle, mat, freq)[0]).days


def coupdays(settle, mat, freq) -> int:
    pcd, ncd, _ = _pcd_ncd(settle, mat, freq)
    return (ncd - pcd).days


def accrued(settle, mat, coupon, freq) -> float:
    return 100 * coupon / freq * coupdaybs(settle, mat, freq) / coupdays(settle, mat, freq)


def _cash_times(settle, mat, coupon, ytm, freq):
    """(cash flow, discount exponent in coupon periods) for every payment after settle."""
    pcd, ncd, n = _pcd_ncd(settle, mat, freq)
    dsc_e = (ncd - _ts(settle)).days / (ncd - pcd).days
    cpn = 100 * coupon / freq
    return [(cpn + (100 if k == n else 0), k - 1 + dsc_e) for k in range(1, n + 1)], dsc_e


def price(settle, mat, coupon, ytm, freq) -> float:
    """Excel PRICE(settle, mat, rate, yld, 100, freq, 1): clean price per 100 face."""
    flows, dsc_e = _cash_times(settle, mat, coupon, ytm, freq)
    a = accrued(settle, mat, coupon, freq)
    if len(flows) == 1:                                    # Excel: simple-interest discount in the last period
        return flows[0][0] / (1 + dsc_e * ytm / freq) - a
    return sum(cf / (1 + ytm / freq) ** t for cf, t in flows) - a


def duration(settle, mat, coupon, ytm, freq) -> float:
    """Excel DURATION(settle, mat, rate, yld, freq, 1): Macaulay duration in years."""
    flows, _ = _cash_times(settle, mat, coupon, ytm, freq)
    pv = [cf / (1 + ytm / freq) ** t for cf, t in flows]
    return sum(p * t for p, (_, t) in zip(pv, flows)) / sum(pv) / freq


# =============================================================================== metrics and scores
def load_bonds(path: str = BONDS) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["maturity"])
    if df.empty:
        raise ValueError(f"no bonds in {path}")
    return df


def bond_key(r) -> str:
    pre = "UST" if r["currency"] == "USD" else r["currency"]
    return f"{pre}_{pd.Timestamp(r['maturity']).date()}_{100 * r['coupon']:.3f}"


def bucket(years: float, s: BondSettings) -> str:
    if years < s.cut1:
        return f"Under {s.cut1:g}"
    if years <= s.cut2:
        return f"{s.cut1:g}-{s.cut2:g}"
    if years <= s.cut3:
        return f"{s.cut2:g}-{s.cut3:g}"
    return f"Over {s.cut3:g}"


def _pct_lower(vals: pd.Series) -> pd.Series:
    """per member: (# members with a LOWER value) / (n - 1), as the workbook's COUNTIFS(...,"<"&x)."""
    n = len(vals)
    return vals.apply(lambda x: (vals < x).sum() / (n - 1))


def _rank(scores: pd.Series) -> pd.Series:
    return scores.apply(lambda x: int((scores > x).sum()) + 1)


def score_bonds(settings: BondSettings | None = None, path: str = BONDS) -> pd.DataFrame:
    """Every workbook column for every bond (one row per bond, same order as the workbook)."""
    s = settings or BondSettings.load(path)
    df = load_bonds(path).copy()
    settle, horizon = _ts(s.settle), edate(s.settle, s.horizon_months)
    rows = []
    for r in df.to_dict("records"):
        m, c, y, f = _ts(r["maturity"]), r["coupon"], r["ytm"], int(r["freq"])
        o = dict(r, key=bond_key(r))
        o["default_risk"] = r["cds_bp"] / 10_000 / (1 - s.recovery)                        # I
        o["years_left"] = (m - settle).days / 365.25                                         # J
        o["bucket"] = bucket(o["years_left"], s)                                             # K
        o["clean"] = price(settle, m, c, y, f)                                               # L
        o["accrued"] = accrued(settle, m, c, f)                                              # M
        o["dirty"] = o["clean"] + o["accrued"]                                               # N
        o["mac_dur"] = duration(settle, m, c, y, f)                                          # O
        o["mod_dur"] = o["mac_dur"] / (1 + y / f)                                            # P
        o["dv01_100k"] = o["mod_dur"] * o["dirty"] / 10                                      # R
        if m <= horizon:                                                                     # S, T, U
            o["price_1y"] = o["coupons_1y"] = float("nan")
            o["carry_1y"] = y
        else:
            o["price_1y"] = price(horizon, m, c, y, f) + accrued(horizon, m, c, f)
            o["coupons_1y"] = 100 * c / f * (coupnum(settle, m, f) - coupnum(horizon, m, f))
            o["carry_1y"] = (o["price_1y"] + o["coupons_1y"]) / o["dirty"] - 1
        o["ret_1y"] = o["carry_1y"] + r["rolldown"]                                          # W
        rf = s.rf(r["currency"])
        o["rf_1y"] = float("nan") if rf is None else rf                                      # Y
        o["excess"] = o["ret_1y"] - o["rf_1y"]                                               # Z
        o["excess_per_dur_bp"] = (o["excess"] / o["mod_dur"] * 10_000                        # AA
                                  if pd.notna(o["excess"]) and o["years_left"] >= s.min_years_across else float("nan"))
        o["ret_per_dur_bp"] = o["ret_1y"] / o["mod_dur"] * 10_000                            # AC
        rows.append(o)
    out = pd.DataFrame(rows)
    out["within_score"] = out["within_rank"] = out["group_within"] = float("nan")
    ok = out.years_left >= s.min_years_within                                                # AB
    for _, g in out[ok].groupby(["bucket", "currency"]):                                     # AD, AE, AF
        n = len(g)
        if n == 1:
            sc = pd.Series(s.lone_score, index=g.index)
        else:
            sc = 100 * (s.w_ret_per_dur * _pct_lower(g.ret_per_dur_bp) + s.w_ret * _pct_lower(g.ret_1y)
                        + s.w_roll * _pct_lower(g.rolldown) + s.w_default * (1 - _pct_lower(g.default_risk)))
        out.loc[g.index, "group_within"] = n
        out.loc[g.index, "within_score"] = sc
        out.loc[g.index, "within_rank"] = _rank(sc)
    out["across_score"] = out["across_rank"] = out["group_across"] = float("nan")
    for _, g in out[out.excess_per_dur_bp.notna()].groupby("currency"):                      # AG, AH, AI, AJ
        n = len(g)
        sc = pd.Series(s.lone_score, index=g.index) if n < 2 else 100 * _pct_lower(g.excess_per_dur_bp)
        out.loc[g.index, "group_across"] = n
        out.loc[g.index, "across_score"] = sc
        out.loc[g.index, "across_rank"] = _rank(sc)
    return out


_CACHE = {}


def scored(plan) -> tuple:
    """(BondSettings, scored DataFrame) for the plan's bond file, computed once."""
    key = plan.bond_data
    if key not in _CACHE:
        s = BondSettings.load(key)
        _CACHE[key] = (s, score_bonds(s, key))
    return _CACHE[key]


def stale_warning(asof, settings: BondSettings) -> str | None:
    gap = abs((_ts(asof) - _ts(settings.settle)).days)
    if gap > settings.stale_days:
        return (f"WARNING stale bond data: yields are as of {settings.settle}, {gap} days from {_ts(asof).date()}; "
                "re-export the workbook inputs and re-run build_approved_bonds.py")
    return None


# =============================================================================== return-seeking / pre-gift sleeve
def _alive(df, asof, plan, min_years):
    asof = _ts(asof)
    left = (df.maturity - asof).dt.days / 365.25
    return df[df.currency.isin(plan.bond_currencies) & (df.maturity > asof) & (left >= min_years)]


def _fallback(plan, phase, why):
    ph = plan.phases[phase]
    t = CASH if ph.allow_cash is True else ph.bond_proxy
    return dict(weights={t: 1.0}, rows=[], skipped=[], status=f"NO BOND ELIGIBLE -> {'T-bills' if t == CASH else t}: {why}")


def duration_limit(plan, phase, year) -> float:
    return max(1.0, min(plan.phase_end(phase) - year, plan.bond_max_duration))


def _sleeve_weights(chosen: pd.DataFrame, cap: float, log: list) -> dict:
    sc = {r.key: (0.0 if pd.isna(r.across_score) else r.across_score) + 1 for r in chosen.itertuples()}
    if len(sc) * cap < 1 - 1e-12:
        log.append(f"cap {cap:.0%} relaxed: only {len(sc)} bond(s), cannot hold with fewer than {math.ceil(1 / cap)}")
        tot = sum(sc.values())
        return {k: v / tot for k, v in sc.items()}
    return capped_score_weights(sc, cap)[0]


def bond_sleeve(asof, plan, phase: str, year: float) -> dict:
    """Weights inside the bond sleeve (sum to 1) plus a report: weights, rows, skipped, status (and the ladder for reserve)."""
    if phase == "reserve":
        return reserve_ladder(asof, plan)
    s, df = scored(plan)
    limit = duration_limit(plan, phase, year)
    uni = _alive(df, asof, plan, s.min_years_within)
    log = []
    if uni.empty:
        rep = _fallback(plan, phase, f"no {'/'.join(plan.bond_currencies)} bond with {s.min_years_within}+ yrs left")
        return dict(rep, limit=limit, wavg_dur=0.0, log=log, warning=stale_warning(asof, s))
    picks, skipped = [], []
    for b, g in uni.groupby("bucket", sort=False):
        g = g.assign(_a=g.across_score.fillna(-1.0))
        best = g.sort_values(["within_score", "_a", "maturity"], ascending=[False, False, True]).iloc[0]
        if best.mod_dur > limit:
            skipped.append(dict(bucket=b, bond=best.bond, mod_dur=best.mod_dur,
                                reason=f"modified duration {best.mod_dur:.2f} over the {limit:.2f} limit"))
        else:
            picks.append(best)
    chosen = pd.DataFrame(picks)
    while len(chosen):
        w = _sleeve_weights(chosen, plan.bond_cap, log)
        wavg = sum(w[r.key] * r.mod_dur for r in chosen.itertuples())
        if wavg <= limit + 1e-12:
            break
        worst = chosen.sort_values("mod_dur").iloc[-1]
        skipped.append(dict(bucket=worst.bucket, bond=worst.bond, mod_dur=worst.mod_dur,
                            reason=f"dropped: sleeve duration {wavg:.2f} over the {limit:.2f} limit"))
        chosen = chosen[chosen.key != worst.key]
    if chosen.empty:
        rep = _fallback(plan, phase, f"every bucket's best bond is over the {limit:.2f} duration limit")
        return dict(rep, skipped=skipped, limit=limit, wavg_dur=0.0, log=log, warning=stale_warning(asof, s))
    rows = []
    for r in chosen.sort_values("maturity").to_dict("records"):
        r["weight"] = w[r["key"]]
        rows.append(r)
    return dict(weights={r["key"]: r["weight"] for r in rows}, rows=rows, skipped=skipped, limit=limit, wavg_dur=wavg,
                log=log, status=f"SCORED BONDS ({len(rows)})", warning=stale_warning(asof, s))


# =============================================================================== reserve ladder
def _flows(r, face: float, asof):
    """(date, cash) for every coupon and the principal paid after asof, for `face` of bond row r."""
    m, f = _ts(r["maturity"]), int(r["freq"])
    cpn = face * r["coupon"] / f
    out, k = [], 0
    while (d := _coupon_date(m, k, f)) > _ts(asof):
        out.append((d, cpn + (face if k == 0 else 0.0)))
        k += 1
    return out


def _best(cands: pd.DataFrame):
    """highest across-maturities score; exact ties -> earlier maturity."""
    c = cands.assign(_a=cands.across_score.fillna(-1.0))
    return c.sort_values(["_a", "maturity"], ascending=[False, True]).iloc[0]


def _simulate(rungs: dict, gap_bond, asof, pay_years: list, payment: float, first_window: int) -> list:
    """Forward cash at each Jan 1 payment, 0% reinvestment. A cash flow in calendar year Y-1 funds Jan 1 of Y.
    Ladder coupons paid before `first_window` are not credited (cushion); the gap bond's flows all count."""
    cash = {y: 0.0 for y in pay_years}
    for r, face in list(rungs.values()) + ([gap_bond] if gap_bond is not None else []):
        is_gap = gap_bond is not None and r is gap_bond[0]
        for d, x in _flows(r, face, asof):
            y = d.year + 1
            if y < pay_years[0]:
                if not is_gap and d.year < first_window:
                    continue
                y = pay_years[0]
            if y in cash:
                cash[y] += x
    out, carry = [], 0.0
    for y in pay_years:
        avail = carry + cash[y]
        out.append(dict(year=y, received=cash[y], carried_in=carry, available=avail, coverage=avail / payment))
        carry = avail - payment
    return out


def reserve_ladder(asof, plan) -> dict:
    s, df = scored(plan)
    asof = _ts(asof)
    base_year = 2026                                       # year 0 of the case
    pay_years = [base_year + y for y in plan.payment_years]
    uni = df[df.currency.isin(plan.bond_currencies) & (df.maturity > asof)]
    funded, gaps = {}, []
    for Y in pay_years:
        c = uni[uni.maturity.dt.year == Y - 1]
        if c.empty:
            gaps.append(Y)
        else:
            funded[Y] = _best(c).to_dict()
    # size faces backward: cash on Jan 1 Y = own principal + coupons in Y-1 + coupons in Y-1 from later rungs
    faces = {}
    for Y in sorted(funded, reverse=True):
        r = funded[Y]
        later = sum(x for Z in faces for d, x in _flows(funded[Z], faces[Z], asof) if d.year == Y - 1)
        own = sum(x for d, x in _flows(r, 1.0, asof) if d.year == Y - 1)          # per $1 face
        faces[Y] = max(0.0, (plan.payment - later) / own)
    faces = {Y: round(f / FACE_LOT) * FACE_LOT for Y, f in faces.items()}
    first_window = pay_years[0] - 1

    def rungs():
        return {Y: (funded[Y], faces[Y]) for Y in funded}

    # rounding to the nearest lot can leave a funded payment a few dollars short: add one lot until covered
    # (checked over the funded payments only; the gap before them is covered separately below)
    fy = [Y for Y in pay_years if funded and Y >= min(funded)]
    for _ in range(100):
        sim = _simulate(rungs(), None, asof, fy, plan.payment, fy[0] - 1) if fy else []
        short = [x["year"] for x in sim if x["coverage"] < 1 - 1e-9]
        if not short:
            break
        faces[short[0]] += FACE_LOT
    no_gap_bond = _simulate(rungs(), None, asof, pay_years, plan.payment, first_window)
    gross_gap = plan.payment * len(gaps)
    # net gap: the most the cumulative cash falls below cumulative payments up to the last gap payment
    net_gap = max([0.0] + [-(x["available"] - plan.payment) for x in no_gap_bond if x["year"] <= (max(gaps) if gaps else 0)])
    gap_bond, cash_needed = None, 0.0
    if gaps and net_gap > 0:
        c = uni[uni.maturity < pd.Timestamp(year=gaps[0], month=1, day=1)]
        if not c.empty:
            last = c[c.maturity.dt.year == c.maturity.dt.year.max()]
            g = _best(last).to_dict()
            per_face = sum(x for _, x in _flows(g, 1.0, asof))
            gf = math.ceil(net_gap / per_face / FACE_LOT) * FACE_LOT
            for _ in range(100):
                sim = _simulate(rungs(), (g, gf), asof, pay_years, plan.payment, first_window)
                if all(x["coverage"] >= 1 - 1e-9 for x in sim):
                    break
                gf += FACE_LOT
            gap_bond = (g, gf)
        else:
            cash_needed = net_gap                          # no bond matures before the gap: T-bills hold it today
    sim = _simulate(rungs(), gap_bond, asof, pay_years, plan.payment, first_window)
    if cash_needed:
        for x in sim:
            x["available"] += cash_needed
            x["coverage"] = x["available"] / plan.payment
    rows = []
    for x in sim:
        r, face = (funded[x["year"]], faces[x["year"]]) if x["year"] in funded else (None, 0.0)
        rows.append(dict(x, payment_date=pd.Timestamp(year=x["year"], month=1, day=1).date(), gap=x["year"] in gaps,
                         key=r["key"] if r else "", bond=r["bond"] if r else "(gap)", stocktrak_no=r["stocktrak_no"] if r else None,
                         maturity=r["maturity"].date() if r else None, face=face, cost=face * r["dirty"] / 100 if r else 0.0))
    holdings = [(r["key"], r, f) for r, f in [(funded[Y], faces[Y]) for Y in funded] + ([gap_bond] if gap_bond else [])]
    costs = {k: f * r["dirty"] / 100 for k, r, f in holdings}
    if cash_needed:
        costs[CASH] = cash_needed
    total = sum(costs.values())
    sleeve_dollars = plan.reserve_amount * plan.phases["reserve"].mix["bonds"]
    gb = None
    if gap_bond:
        r, f = gap_bond
        gb = dict(key=r["key"], bond=r["bond"], stocktrak_no=r["stocktrak_no"], maturity=r["maturity"].date(), face=f,
                  cost=f * r["dirty"] / 100, proceeds=sum(x for _, x in _flows(r, f, asof)))
    return dict(weights={k: c / total for k, c in costs.items()}, rows=rows, skipped=[dict(year=Y, reason="no bond matures in "
                f"{Y - 1}") for Y in gaps], status=f"LADDER ({len(funded)} rungs, gap {len(gaps)} payments)",
                gap_years=gaps, gross_gap=gross_gap, net_gap=net_gap, gap_bond=gb, cash_needed=cash_needed,
                total_cost=total, sleeve_dollars=sleeve_dollars, sleeve_shortfall=max(0.0, total - sleeve_dollars),
                limit=None, wavg_dur=sum(costs.get(k, 0) / total * r["mod_dur"] for k, r, _ in holdings), log=[],
                warning=stale_warning(asof, s))


def capital_plan(rows: list, capital: float) -> tuple:
    """Face to buy per bond (price incl. accrued, rounded DOWN to the lot so it never overspends) for `capital`
    dollars of bond sleeve."""
    out, spent = [], 0.0
    for r in rows:
        face = math.floor(capital * r["weight"] / (r["dirty"] / 100) / FACE_LOT + 1e-9) * FACE_LOT
        cost = face * r["dirty"] / 100
        out.append(dict(key=r["key"], bond=r["bond"], stocktrak_no=r["stocktrak_no"], face=face, cost=cost))
        spent += cost
    return out, capital - spent
