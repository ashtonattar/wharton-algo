"""
fetch_country_etfs.py - monthly adjusted prices (total return) for the foreign country-ETF sleeve.

    python fetch_country_etfs.py            # writes data_us/country_etfs.csv, adds the ETFs to data_us/meta.csv
    python fetch_country_etfs.py data       # same for another data folder

Downloads every market-tracking ETF in approved_etfs.csv plus the approved index funds the plan holds as
fallbacks (VEA foreign, VOO domestic). Before VEA / VOO existed (2007 / 2010), their history is spliced from
EFA / SPY in funds.csv, the same way fetch_data.py extends EFA and IEF; this is a proxy and is flagged.
Re-run after build_approved_etfs.py and after build_us_universe.py (which rewrites meta.csv).
"""
import os, sys
import pandas as pd
import yfinance as yf

from fetch_data import retry

HERE = os.path.dirname(os.path.abspath(__file__))
SPLICE = {"VEA": "EFA", "VOO": "SPY"}      # approved fund -> older proxy in funds.csv


def monthly(sym):
    h = retry(lambda: yf.Ticker(sym).history(period="max", interval="1mo", auto_adjust=True)["Close"])
    h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
    return h[~h.index.duplicated(keep="last")]


def main(folder):
    etfs = pd.read_csv(os.path.join(HERE, "approved_etfs.csv"))
    y = etfs[etfs.market_tracking == "Y"]
    funds = pd.read_csv(os.path.join(folder, "funds.csv"), index_col=0, parse_dates=True)
    last = funds.index.max()
    px = {}
    for t in list(y.ticker) + list(SPLICE):
        px[t] = monthly(t)
        print(f"{t:6s} {px[t].index.min().date()} - {px[t].index.max().date()}  {len(px[t])} months")
    f = pd.DataFrame(px).sort_index().loc[:last]          # same last month as the rest of the data folder
    for t, proxy in SPLICE.items():
        p = funds[proxy].dropna()
        first = f[t].first_valid_index()
        f = f.reindex(f.index.union(p.index[p.index < first]))
        f.loc[f.index < first, t] = (p / p.loc[first] * f.loc[first, t])[p.index < first]
        print(f"{t}: before {first.date()} spliced from {proxy}")
    f.sort_index().to_csv(os.path.join(folder, "country_etfs.csv"))
    meta_f = os.path.join(folder, "meta.csv")
    meta = pd.read_csv(meta_f)
    meta = meta[~meta.ticker.isin(f.columns)]
    add = [dict(ticker=r.ticker, name=r.name, country=r.target_country, sector="Country ETF", currency="USD",
                financial_currency="USD") for r in y.itertuples()]
    add += [dict(ticker="VEA", name="VANGUARD FTSE DEVELOPED ETF", country="International", sector="Index fund",
                 currency="USD", financial_currency="USD"),
            dict(ticker="VOO", name="VANGUARD S&P 500 ETF", country="United States", sector="Index fund",
                 currency="USD", financial_currency="USD")]
    pd.concat([meta, pd.DataFrame(add)]).to_csv(meta_f, index=False)
    print("saved", os.path.join(folder, "country_etfs.csv"), "and", len(add), "rows in meta.csv")


if __name__ == "__main__":
    main(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "data_us"))
