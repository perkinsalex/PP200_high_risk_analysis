"""
Trend analysis: are new homes increasingly being built in high fire-risk areas?

Reads  ~/Desktop/ocr_analysis/ca_homes_with_risk.parquet  (created by ca_high_risk_new_homes.py)
Writes CSVs + a PNG chart into the same folder.

Install once:  conda install -n firerisk -c conda-forge matplotlib
Run:           python trend_high_risk_new_homes.py
"""
from pathlib import Path
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- settings -----------------------------------
WORKDIR = Path.home() / "Desktop" / "ocr_analysis"
DATA = WORKDIR / "ca_homes_with_risk.parquet"

RISK_COL = "rps_2011"        # or "rps_2047" for the future-climate scenario
THRESHOLDS = {"score>=8": 0.5, "score>=9": 1.0, "score=10": 3.0}
MAIN = "score>=9"            # definition used for the chart and headline numbers

FIRST_YEAR = 1980            # start of the trend window (try 1970 or 1950 for longer history)
PRINT_LAST_N = 30            # how many recent years to print in the year-by-year table
RECENT = (2024, 2025)        # "last 2 years" (2026 is partial -> shown separately)
PRIOR = (2022, 2023)         # comparison window

# --------------------------------- load -----------------------------------
df = pd.read_parquet(DATA)
df = df[df[RISK_COL].notna() & df["yearbuilt"].between(FIRST_YEAR, 2026)].copy()
print(f"Homes in analysis (matched to a building, valid year {FIRST_YEAR}+): {len(df):,}")


def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else np.nan


all_tables = []
for label, cut in THRESHOLDS.items():
    df["high"] = df[RISK_COL] >= cut

    yr = (df.groupby(["yearbuilt", "high"]).size().unstack(fill_value=0)
            .rename(columns={True: "high_risk", False: "other"}))
    yr["total"] = yr["high_risk"] + yr["other"]
    yr["pct_high"] = 100 * yr["high_risk"] / yr["total"]
    yr["yoy_high_%"] = 100 * yr["high_risk"].pct_change()
    yr["yoy_other_%"] = 100 * yr["other"].pct_change()
    base = yr.loc[2015, ["high_risk", "other"]] if 2015 in yr.index else yr.iloc[0][["high_risk", "other"]]
    yr["index_high_(2015=100)"] = 100 * yr["high_risk"] / base["high_risk"]
    yr["index_other_(2015=100)"] = 100 * yr["other"] / base["other"]
    yr.insert(0, "threshold", label)
    all_tables.append(yr.reset_index())

    # recent vs prior window
    r = yr.loc[RECENT[0]:RECENT[1]]
    p = yr.loc[PRIOR[0]:PRIOR[1]]
    hi_r, hi_p = r["high_risk"].sum(), p["high_risk"].sum()
    ot_r, ot_p = r["other"].sum(), p["other"].sum()
    z = two_prop_z(hi_r, hi_r + ot_r, hi_p, hi_p + ot_p)
    print(f"\n=== {label} ({RISK_COL}) ===")
    print(f"High-risk homes built {RECENT[0]}-{RECENT[1]}: {hi_r:,}  vs {PRIOR[0]}-{PRIOR[1]}: {hi_p:,}"
          f"  ({100*(hi_r/hi_p-1):+.1f}%)")
    print(f"All other homes built {RECENT[0]}-{RECENT[1]}: {ot_r:,}  vs {PRIOR[0]}-{PRIOR[1]}: {ot_p:,}"
          f"  ({100*(ot_r/ot_p-1):+.1f}%)")
    print(f"High-risk share of new homes: {100*hi_p/(hi_p+ot_p):.2f}% -> {100*hi_r/(hi_r+ot_r):.2f}%"
          f"  (change {100*(hi_r/(hi_r+ot_r)-hi_p/(hi_p+ot_p)):+.2f} pts, z={z:.1f})")

    if label == MAIN:
        main_yr = yr
        print(f"\nYear-by-year, last {PRINT_LAST_N} years:")
        print(yr.tail(PRINT_LAST_N)[["high_risk", "other", "pct_high", "yoy_high_%", "yoy_other_%"]]
              .round(1).to_string())

pd.concat(all_tables).to_csv(WORKDIR / "trend_by_year.csv", index=False)

# ----------------------- where is the recent building? -----------------------
cut = THRESHOLDS[MAIN]
rec = df[(df[RISK_COL] >= cut) & df["yearbuilt"].between(*RECENT)]
cty = rec.groupby("county").size().sort_values(ascending=False).head(15)
print(f"\nTop counties for high-risk homes built {RECENT[0]}-{RECENT[1]} ({MAIN}):")
print(cty.to_string())
cty.to_csv(WORKDIR / "recent_high_risk_by_county.csv")

# --------------------------------- chart ------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
yrs = main_yr.index
ax[0].plot(yrs, main_yr["index_high_(2015=100)"], label=f"High risk ({MAIN})", lw=2.2, color="#c0392b")
ax[0].plot(yrs, main_yr["index_other_(2015=100)"], label="All other", lw=2.2, color="#2c3e50")
ax[0].axhline(100, color="grey", lw=.6, ls=":")
ax[0].set_title("Homes built per year (indexed, 2015 = 100)")
ax[0].set_yscale("log")
ax[1].plot(yrs, main_yr["pct_high"], lw=2.2, color="#c0392b")
ax[1].set_title("High-risk share of homes built that year (%)")
for a in ax:
    a.axvspan(2024.5, 2026.5, color="grey", alpha=.15)  # recent years: assessor data may lag
    a.set_xlabel("Year built")
    a.grid(alpha=.25)
ax[0].legend(frameon=False)
fig.text(0.5, -0.02, "Shaded: 2025-26 may be incomplete (assessor records lag new construction).",
         ha="center", fontsize=9, color="grey")
fig.tight_layout()
fig.savefig(WORKDIR / "trend_chart.png", dpi=160, bbox_inches="tight")
print(f"\nSaved trend_by_year.csv, recent_high_risk_by_county.csv, trend_chart.png to {WORKDIR}")