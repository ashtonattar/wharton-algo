# Laura Gao strategy: Funnel v3 code

v3 is the default. `Plan()` uses the v3 phase settings; `Plan(phases=default_phases())` gives the old v2 settings.

## Setup (once)
```
python3 -m venv .venv && source .venv/bin/activate
pip install yfinance pandas numpy scipy scikit-learn matplotlib reportlab openpyxl pytest pypdf
python build_approved_etfs.py   # approved ETF list (PDF) -> approved_etfs.csv
python fetch_country_etfs.py    # country ETF + VEA/VOO prices -> data_us/country_etfs.csv, rows in data_us/meta.csv
python -m pytest -q             # test_funnel_v2.py and one live test skip when the demo data/ folder is missing
```

## Files
| File | What it is |
|---|---|
| `funnel_v2.py` | The stock-picking algorithm: gates A1-A9, rank B1, portfolio gates C1-C6, Monte Carlo, data loading. (The file name is historical; it holds the v3 options.) |
| `laura_strategy.py` | Laura's plan: the five phases, gate M1 (market switch), the bond ladder, the 2031 split / 2033 gift / $50k payments, and the full-plan backtest. `v3_phases()` = final settings |
| `us_backtest.py` | Real-market backtests on the S&P 500 as it stood on each date (`data_us/`) |
| `run_v3.py` | One full-plan replay: `python run_v3.py v3 2001-12-31` (or `v2`) |
| `phase_backtest.py` | Each phase on its own over its real length |
| `build_us_universe.py` | Builds `data_us/` from the S&P 500 membership history and Yahoo prices |
| `country_sleeve.py` | Foreign sleeve: the country screener (`../Book 9(Country Screener).csv`) picks and weights approved country ETFs |
| `bond_sleeve.py` | Bond sleeve (`Plan(bond_mode="scored")`): the bond workbook's metrics and scores, best bond per maturity bucket under a duration limit, and the reserve's $50k-payment ladder. Live-only; backtests keep IEF |
| `build_approved_bonds.py` | Reads the input cells of `../(C) Bond Model (Fixed).xlsx` into `approved_bonds.csv` + `bond_settings.json` (re-run when yields change) |
| `bond_report.py` | All 60 bonds with every metric and both scores -> `bond_report.csv` (IPS appendix) |
| `sector_sleeve.py` | Sector tilt on the domestic stocks: the sector screener (`../(C) Sector Screener Inputs.csv`) scores the 11 sectors, tiers them, and gives the domestic funnel per-sector caps and a B1 `sector` pillar |
| `sector_report.py` | All 11 sectors with every input, each test's pass/fail and points, score, risk and tier -> `sector_report.csv` (IPS appendix) |
| `build_approved_etfs.py` | Parses the competition's approved ETF PDF into `approved_etfs.csv` (re-run with the 26-27 URL) |
| `fetch_country_etfs.py` | Downloads monthly prices for the country ETFs and the approved fallback funds (VEA, VOO) |
| `legacy_baseline.json` | Decisions recorded with the code before the country-ETF sleeve; the old modes must still match it |
| `fetch_data.py` | Downloads prices, statements, bonds and index funds for any ticker list (`python fetch_data.py my_universe.csv`) |
| `case_study.py`, `enrich_case.py` | Replays the 31 Dec 2003 decision used in the PDF's case studies |
| `build_results_xlsx.py` | Rebuilds the results spreadsheet from saved runs |
| `test_*.py` | Tests: no look-ahead, caps hold, the spreadsheet plan reproduced to the cent, M1 and bond ladder behave as described |
| `data/` | 41-stock demo universe with company statements (live demo) |
| `data_us/` | S&P 500 point-in-time universe, 1990-2026 monthly prices (backtests) |

## Foreign sleeve (country ETFs)
- Equity is split 70% U.S. / 30% foreign in every phase (`Plan.foreign_share_of_equity`); cash and bond shares are unchanged.
- Countries: the screener's top 22 (GDP >= $500bn, GDP per capita >= $10k, ranked by score), minus the U.S. Walk down the list and keep the first 5 (`n_foreign_countries`) with an approved single-country ETF that has 12 months of prices. Skipped countries are listed with the reason.
- Weights: score / sum of scores, capped at 40% per country. With fewer than 3 countries the cap cannot hold, and the rest goes to VEA (approved developed-markets fund). On the 25-26 list only Canada (BBCA) and Australia (EWA) qualify: 40% / 40% / VEA 20%.
- The ETFs skip the stock funnel. Gate M1 still applies (EFA below its 10-month average -> the foreign sleeve goes to T-bills). Security phases (reserve, pre-gift) keep their foreign sleeve in T-bills.
- Fallback holdings are approved funds: VOO (U.S.) and VEA (foreign). Gate M1 still reads SPY and EFA prices as its signal.
- Old behaviour: `Plan(foreign_mode="funnel")` (foreign stocks) or `"index"` (= the old `foreign_index_only=True`); add `foreign_share_of_equity=None, fallback_holdings={"domestic": "SPY", "foreign": "EFA"}` for the exact old numbers.

## Sector tilt (domestic stocks)
- The sector screener (Book 9.xlsx, Sector Screener sheet) scores each sector 0-10 (11 tests, higher is better) and gives it a risk score 0-10 (10 penalties, higher is riskier). Every sector's EPS growth is compared to the S&P's 16% (the sheet's `G44` reference is not locked; fixed here).
- Tiers: sectors by score (ties: lower risk). Overweight = the first 3 with risk below 7; Underweight = the last 3; the rest Neutral. Today: Overweight Technology, Healthcare, Utilities; Underweight Financial Services, Real Estate, Consumer Defensive.
- Caps (share of the account's equities, same basis as the old flat 35%): Overweight 35%, Neutral 25%, Underweight 10%; sectors the screener doesn't cover use the Neutral cap. They go through the same account -> sleeve formula as `sector_cap`.
- B1 gets a `sector` pillar (10%; the phase's other pillars are scaled by 0.90). Stocks in sectors the screener doesn't cover get the median sector score.
- Domestic sleeve only; the foreign sleeve is untouched. If the tilted caps can't be met (`NOT ENOUGH STOCKS` / `SOLVER FAILED`), the sleeve is re-run with the flat cap ("sector tilt relaxed"), then the usual VOO fallback.
- Old behaviour: `Plan(sector_mode="off")` or `python run_today.py growth --sector off`.

## Common runs
```
python run_v3.py v3 2006-12-31                    # whole plan, history from 2007 playing 2027-2042
python us_backtest.py phases growth post_gift     # each phase on its own, every start year (slow: ~1 hr)
python laura_strategy.py                          # what each phase would hold today (demo universe)
python run_today.py growth --capital 300000       # live picks incl. scored Treasuries (--bonds ief = old IEF)
```
A start date is the month-end that plays the role of Jan 2027; it needs 16 years of data after it (latest: 2009-12-31).

## Limits
- Free data: Yahoo lacks prices for some failed companies. The members we have beat the real equal-weight S&P 500 by about 1.5% a year.
- Company statements exist only from 2021, so in backtests gates A3-A5 let every stock through.
- The foreign part holds country ETFs picked with the 2026 screener snapshot; any backtest before 2025 therefore has look-ahead bias in country selection (a warning is printed), and before 2019 Canada has no approved ETF with history (BBCA started in 2018).
- VEA and VOO prices before 2007 / 2010 are spliced from EFA / SPY.
- The sector tilt uses one October 2026 snapshot (forward P/E, EPS revisions, analyst numbers), so it is skipped on any date before 2026-09-30; `Plan(sector_tilt_in_backtests=True)` forces it on with a look-ahead warning.
