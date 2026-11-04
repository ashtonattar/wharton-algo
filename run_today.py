"""
run_today.py - what each phase would buy today, choosing from the real S&P 500 (data_us/).

    python run_today.py                 # growth phase, latest month-end in the data
    python run_today.py post_gift       # any phase: growth, pre_gift_y5, pre_gift, post_gift, reserve
    python run_today.py growth 2015-12-31   # any past month-end (only data up to that date is used)
    python run_today.py growth --capital 300000   # face of each bond to buy for $300k of account
    python run_today.py reserve --bonds ief       # bond sleeve as IEF (the old behaviour); default: scored Treasuries
    python run_today.py growth --sector off       # no sector tilt on the domestic stocks (the old behaviour)

Note: data_us/ has no company statements, so gates A3-A5 (valuation, debt, earnings quality)
let every stock through; check those by hand before trading.
The foreign sleeve holds the screener's country ETFs (country_sleeve.py); security phases keep it in T-bills.
"""
import argparse
from funnel_v2 import MarketData
from laura_strategy import Plan, decide, market_up
from build_us_universe import universe_fn
from country_sleeve import lookahead_warning
import bond_sleeve as bs

ap = argparse.ArgumentParser()
ap.add_argument("phase", nargs="?", default="growth")
ap.add_argument("asof", nargs="?")
ap.add_argument("--bonds", choices=("scored", "ief"), default="scored")
ap.add_argument("--sector", choices=("tilt", "off"), default="tilt")
ap.add_argument("--capital", type=float, help="account dollars: face of each bond to buy")
args = ap.parse_args()
phase = args.phase
data = MarketData("data_us")
asof = args.asof or str(data.price_usd.dropna(how="all").index[-1].date())
year = {"growth": 1, "pre_gift_y5": 5, "pre_gift": 6, "post_gift": 7, "reserve": 5}[phase]
plan = Plan(universe_at=universe_fn("data_us"), bond_mode=args.bonds, sector_mode=args.sector)

print(f"Phase: {phase}   Date: {asof}   Market above 10-month average: {market_up(data, asof, plan)}")
if lookahead_warning(plan, asof):
    print(lookahead_warning(plan, asof))
dec = decide(data, asof, plan, phase, year, backtest_mode=True)
if "bonds_chosen" in dec["info"]:
    print(f"Bond ladder chose {dec['info']['bonds_chosen']:.0%} bonds")
for sleeve, i in dec["info"]["sleeves"].items():
    p = "" if i["p_hit"] is None else f" (chance of hitting target: {i['p_hit']:.0%})"
    print(f"  {sleeve} {'ETFs' if 'country' in i else 'stocks'}: {i['status']}{p}")

domestic = dec["info"]["sleeves"].get("domestic", {})
sec = domestic.get("sector")
if sec:
    meta_sector = data.meta["sector"]
    actual = {}
    for t, x in domestic["weights"].items():
        s_ = meta_sector.get(t, t)
        actual[s_] = actual.get(s_, 0.0) + x
    print(f"\nSector tilt on the domestic stocks (cap = share of the account's equities)")
    print(f"  {'Sector':24s} {'Score':>5s} {'Risk':>5s}  {'Tier':12s} {'Cap':>4s} {'sleeve cap':>10s} {'in sleeve':>9s}")
    for s_, r in sec["table"].iterrows():
        print(f"  {s_:24s} {r.score:5.1f} {r.risk:5.1f}  {r.tier.capitalize():12s} {sec['caps'][s_]:4.0%} {sec['sleeve_caps'][s_]:10.0%} {actual.pop(s_, 0.0):9.1%}")
    for s_, x in sorted(actual.items(), key=lambda x: -x[1]):
        print(f"  {s_:24s} {'':5s} {'':5s}  {'(no score)':12s} {'':4s} {'':10s} {x:9.1%}")
    if sec["relaxed"]:
        print(f"  {sec['relaxed']} (flat caps used)")

foreign = dec["info"]["sleeves"].get("foreign", {})
rep = foreign.get("country")
if rep:
    share = sum(dec["weights"].get(t, 0.0) for t in foreign["weights"])     # foreign sleeve's share of the account
    print(f"\nForeign sleeve: country ETFs ({share:.1%} of the account)")
    print(f"  {'ETF':6s} {'Country':12s} {'Score':>6s} {'12m ret':>8s} {'in sleeve':>9s} {'in account':>10s}")
    for r in rep["rows"]:
        print(f"  {r['ticker']:6s} {r['country']:12s} {r['score']:6.2f} {r['ret_12m']:8.1%} {r['weight']:9.1%} {r['weight'] * share:10.1%}")
    if rep["filler"]:
        print(f"  {plan.foreign_filler:6s} {'(40% cap)':12s} {'':6s} {'':8s} {rep['filler']:9.1%} {rep['filler'] * share:10.1%}"
              "   developed-markets fund holds what the country cap leaves")
    if rep["skipped"]:
        print("  Skipped:")
        for s in rep["skipped"]:
            print(f"    {s['country']:22s} {s['score']:6.2f}  {s['reason']}")

bonds = dec["info"].get("bonds")
if bonds:
    share = bonds["share"]                                                   # bond sleeve's share of the account
    print(f"\nBond sleeve: {bonds['status']} ({share:.1%} of the account)   note: {bs.LIVE_NOTE}")
    if bonds.get("warning"):
        print("  " + bonds["warning"])
    if phase == "reserve":
        print(f"  Ladder for the $50k payments on Jan 1 (0% reinvestment, face rounded to ${bs.FACE_LOT:,.0f}):")
        print(f"  {'Payment':10s} {'Bond':24s} {'ST#':>4s} {'Face':>9s} {'Cost today':>10s} {'Available':>10s} {'vs $50,000':>10s} {'Coverage':>8s}")
        for r in bonds["rows"]:
            st = "" if r["stocktrak_no"] is None else f"{r['stocktrak_no']:.0f}"
            print(f"  {r['payment_date']!s:10s} {r['bond']:24s} {st:>4s} {r['face']:9,.0f} {r['cost']:10,.0f} "
                  f"{r['available']:10,.0f} {r['available'] - 50_000:+10,.0f} {r['coverage']:8.2f}")
        g = bonds["gap_bond"]
        if g:
            print(f"  Gap bond: {g['bond']} (#{g['stocktrak_no']:.0f}) face {g['face']:,.0f}, cost {g['cost']:,.0f}, "
                  f"pays {g['proceeds']:,.0f} by {g['maturity']} -> held in T-bills for the gap payments")
        print(f"  Gap: {len(bonds['gap_years'])} payments with no prior-year maturity ({', '.join(map(str, bonds['gap_years']))}) "
              f"= ${bonds['gross_gap']:,.0f} gross, ${bonds['net_gap']:,.0f} after ladder coupons")
        print(f"  Ladder cost ${bonds['total_cost']:,.0f} vs reserve bond sleeve ${bonds['sleeve_dollars']:,.0f} "
              f"({plan.phases['reserve'].mix['bonds']:.0%} of ${plan.reserve_amount:,.0f}): short ${bonds['sleeve_shortfall']:,.0f}")
    elif bonds["rows"]:
        print(f"  {'#':>2s} {'Bond':24s} {'ST#':>4s} {'Maturity':10s} {'YTM':>6s} {'ModDur':>6s} {'1y ret':>6s} "
              f"{'Within':>6s} {'Across':>6s} {'in sleeve':>9s} {'in account':>10s}")
        for i, r in enumerate(sorted(bonds["rows"], key=lambda r: -r["weight"]), 1):
            print(f"  {i:2d} {r['bond']:24s} {r['stocktrak_no']:4.0f} {r['maturity'].date()!s:10s} {r['ytm']:6.2%} "
                  f"{r['mod_dur']:6.2f} {r['ret_1y']:6.2%} {r['within_score']:6.1f} {r['across_score']:6.1f} "
                  f"{r['weight']:9.1%} {r['weight'] * share:10.1%}")
    for sk in bonds["skipped"]:
        what = f"{sk['bucket']} bucket, {sk['bond']}" if "bucket" in sk else f"Jan {sk['year']} payment"
        print(f"  Skipped {what}: {sk['reason']}")
    for line in bonds.get("log", []):
        print("  " + line)
    if bonds.get("limit") is not None:
        print(f"  Sleeve modified duration {bonds['wavg_dur']:.2f} vs limit {bonds['limit']:.2f}")
    if args.capital and bonds["rows"] and phase != "reserve":
        buys, left = bs.capital_plan(bonds["rows"], args.capital * share)
        print(f"  To buy with ${args.capital:,.0f} (bond sleeve ${args.capital * share:,.0f}):")
        for b in buys:
            print(f"    #{b['stocktrak_no']:<3.0f} {b['bond']:24s} face {b['face']:9,.0f}  cost {b['cost']:10,.0f}")
        print(f"    unspent ${left:,.0f}")
elif args.bonds == "scored":
    print(f"\nBond sleeve: 0% of the account (bond ladder chose no bonds)   note: {bs.LIVE_NOTE}")

print("\nHoldings:")
labels = {"IEF": "Treasury bonds (IEF)", "EFA": "International index fund (EFA)", "SPY": "S&P 500 index fund (SPY)",
          "CASH": "T-bills", "VEA": "Developed-markets index fund (VEA)", "VOO": "S&P 500 index fund (VOO)"}
countries = {r["ticker"]: r["country"] for r in (rep or {}).get("rows", [])}
bond_names = {r["key"]: r["bond"] for r in (bonds or {}).get("rows", []) if r.get("key")}
if bonds and bonds.get("gap_bond"):
    bond_names[bonds["gap_bond"]["key"]] = bonds["gap_bond"]["bond"] + " (gap)"
labels.update({k: f"{n} ({k})" for k, n in bond_names.items()})
for t, w in sorted(dec["weights"].items(), key=lambda x: -x[1]):
    label = labels.get(t, f"{t} ({countries[t]} ETF)" if t in countries else t)
    print(f"  {w:6.1%}  {label}")
