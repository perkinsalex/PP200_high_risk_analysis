#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:32:55 2026

@author: alexandraperkins
"""

"""
Four more charts from ca_homes_with_risk.parquet (made by ca_high_risk_new_homes.py).

  fig1_risk_distribution.png  - are new homes skewed toward higher risk scores than older homes?
  fig2_map_new_homes.png      - where are new homes being built, and where is it high risk?
  fig3_counties.png           - which counties have the most new high-risk homes?
  fig4_by_decade.png          - what share of each decade's homes sits in high-risk areas?

Run:  python more_charts.py     (needs matplotlib, pandas, numpy, pyarrow)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = Path.home() / "Desktop" / "ocr_analysis"
DATA = WORKDIR / "ca_homes_with_risk.parquet"

NEW_FROM = 2006            # "new" = built this year or later
HIGH_CUT = 1.0             # rps >= 1  -> score 9-10 ("high risk")
RISK_COL = "rps_2011"
RED, NAVY, GREY = "#c0392b", "#2c3e50", "#95a5a6"

df = pd.read_parquet(DATA)
df = df[df[RISK_COL].notna() & df["yearbuilt"].between(1900, 2026)].copy()
df["new"] = df["yearbuilt"] >= NEW_FROM
df["high"] = df[RISK_COL] >= HIGH_CUT

# CarbonPlan score (0-10) from rps, using their bin edges
edges = [0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.5, 1, 3]
df["score"] = np.where(df[RISK_COL] <= 0, 0, 1 + np.searchsorted(edges, df[RISK_COL].values))
print(f"{len(df):,} homes; {df['new'].sum():,} built {NEW_FROM}+; "
      f"{df['high'].mean():.1%} of all homes are high risk")

# ---- fig 1: score distribution, new vs older -----------------------------
d_new = df[df["new"]]["score"].value_counts(normalize=True).sort_index() * 100
d_old = df[~df["new"]]["score"].value_counts(normalize=True).sort_index() * 100
scores = np.arange(0, 11)
w = 0.4
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(scores - w/2, d_old.reindex(scores, fill_value=0), w, color=GREY, label=f"Built before {NEW_FROM}")
ax.bar(scores + w/2, d_new.reindex(scores, fill_value=0), w, color=RED, label=f"Built {NEW_FROM} or later")
ax.set_xticks(scores)
ax.set_xlabel("CarbonPlan wildfire risk score (0 = lowest, 10 = highest)")
ax.set_ylabel("% of homes in group")
ax.set_title("Are new homes skewed toward higher fire risk?")
ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(WORKDIR / "fig1_risk_distribution.png", dpi=160); plt.close(fig)

# ---- fig 2: maps ---------------------------------------------------------
n = df[df["new"]]
aspect = 1 / np.cos(np.radians(37))
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
hb = ax[0].hexbin(n["lon"], n["lat"], C=n["high"].astype(float), reduce_C_function=np.mean,
                  gridsize=90, mincnt=10, cmap="YlOrRd", vmin=0, vmax=1)
ax[0].set_title(f"Share of homes built {NEW_FROM}+ that are high risk")
fig.colorbar(hb, ax=ax[0], shrink=.7, label="share high risk")
h = n[n["high"]]
hb2 = ax[1].hexbin(h["lon"], h["lat"], gridsize=90, mincnt=1, cmap="Reds", bins="log")
ax[1].set_title(f"Count of high-risk homes built {NEW_FROM}+")
fig.colorbar(hb2, ax=ax[1], shrink=.7, label="homes (log scale)")
for a in ax:
    a.set_aspect(aspect); a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
    a.set_xlim(-124.6, -114.0); a.set_ylim(32.4, 42.1)
fig.tight_layout(); fig.savefig(WORKDIR / "fig2_map_new_homes.png", dpi=160); plt.close(fig)

# ---- fig 3: counties -----------------------------------------------------
g = n.groupby("county").agg(new_total=("high", "size"), new_high=("high", "sum"))
g["pct_high"] = 100 * g["new_high"] / g["new_total"]
top = g.sort_values("new_high", ascending=False).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top.index.astype(str), top["new_high"], color=RED)
for b, (cnt, pct) in zip(bars, zip(top["new_high"], top["pct_high"])):
    ax.text(b.get_width(), b.get_y() + b.get_height()/2, f"  {pct:.0f}% of county's new homes",
            va="center", fontsize=8.5, color=NAVY)
ax.set_xlim(0, top["new_high"].max() * 1.35)
ax.set_xlabel(f"High-risk homes built {NEW_FROM}+")
ax.set_title("Counties with the most new homes in high fire-risk areas")
ax.grid(axis="x", alpha=.25)
fig.tight_layout(); fig.savefig(WORKDIR / "fig3_counties.png", dpi=160); plt.close(fig)
g.sort_values("new_high", ascending=False).to_csv(WORKDIR / "counties_new_homes_risk.csv")

# ---- fig 4: share high risk by decade built, 2011 vs 2047 climate --------
df["decade"] = (df["yearbuilt"] // 10) * 10
dec = df[df["decade"] >= 1940].groupby("decade").agg(
    hi_2011=("rps_2011", lambda s: 100 * (s >= HIGH_CUT).mean()),
    hi_2047=("rps_2047", lambda s: 100 * (s >= HIGH_CUT).mean()) if "rps_2047" in df else ("rps_2011", "size"))
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(dec)); w = 0.4
ax.bar(x - w/2, dec["hi_2011"], w, color=NAVY, label="Risk under 2011 climate")
if "rps_2047" in df:
    ax.bar(x + w/2, dec["hi_2047"], w, color=RED, label="Risk under projected 2047 climate")
ax.set_xticks(x); ax.set_xticklabels([f"{d}s" for d in dec.index])
ax.set_ylabel("% of homes in high-risk areas")
ax.set_title("Share of homes in high fire-risk areas, by decade built")
ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(WORKDIR / "fig4_by_decade.png", dpi=160); plt.close(fig)

print(f"Saved fig1-fig4 PNGs and counties_new_homes_risk.csv to {WORKDIR}")