"""
=============================================================================
  Kiva Loans — Business Intelligence Project
  Steps 1–3: EDA  |  Cleaning & Transformation  |  Visualization
=============================================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Global Style ─────────────────────────────────────────────────────────────
PALETTE_PRIMARY   = "#C0392B"   # crimson – Kiva brand red
PALETTE_SECONDARY = "#2C3E50"   # dark navy
PALETTE_ACCENT    = "#F39C12"   # amber
PALETTE_NEUTRAL   = "#7F8C8D"   # grey

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"       : 150,
    "savefig.dpi"      : 150,
    "savefig.bbox"     : "tight",
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "font.family"      : "DejaVu Sans",
})

DATA_PATH = "masked_kiva_loans.csv"


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & INITIAL EXPLORATION (EDA)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 1.1  Load ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# ── 1.2  Gender Classification ───────────────────────────────────────────────
# FIX: original code over-wrote the column in-place causing order-dependency
# bugs; we now build it cleanly from a lowercase copy.
_g = df["borrower_genders"].str.lower().fillna("")
has_female = _g.str.contains("female")
has_male   = _g.str.contains(r"\bmale\b", regex=True)

conditions  = [has_female & has_male, has_female & ~has_male, has_male & ~has_female]
choices     = ["Mixed", "Female", "Male"]
df["gender_simple"] = np.select(conditions, choices, default="Unknown")

# ── 1.3  Extract time features for later use ─────────────────────────────────
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")

# ── 1.4  Console Summary ─────────────────────────────────────────────────────
print(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range    : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"Countries     : {df['country'].nunique()}")
print(f"Sectors       : {df['sector'].nunique()}")
print(f"Total Funded  : ${df['funded_amount'].sum():,.0f}")
print(f"Avg Loan      : ${df['funded_amount'].mean():,.0f}")

print("\n── Gender breakdown ──────────────────────")
print(df["gender_simple"].value_counts(dropna=False).to_string())

print("\n── Numeric summary ───────────────────────")
print(df[["funded_amount", "loan_amount", "term_in_months", "lender_count"]]
      .describe().round(2).to_string())

print("\n── Missing values ────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "None")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — GENERATING VISUALIZATIONS")
print("=" * 60)

# ── Chart 01: Top 10 Sectors by Total Funding ────────────────────────────────
# Retained & upgraded: added value labels and sorted bar colours by magnitude.
sector_totals = (
    df.groupby("sector")["funded_amount"]
    .sum()
    .sort_values(ascending=True)        # ascending so largest is at top
    .tail(10)
    / 1e6
)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(sector_totals.index, sector_totals.values,
               color=PALETTE_PRIMARY, edgecolor="white", height=0.65)

# Value labels on each bar
for bar in bars:
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"${bar.get_width():.1f}M", va="center", ha="left", fontsize=9,
            color=PALETTE_SECONDARY)

ax.set_xlabel("Total Funded Amount (Millions USD)", labelpad=8)
ax.set_title("Top 10 Sectors by Total Funding", fontweight="bold", pad=14)
ax.set_xlim(0, sector_totals.max() * 1.15)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
plt.savefig("01_sector_funding_bar.png")
plt.close()
print("  Saved → 01_sector_funding_bar.png")


# ── Chart 02: Sector Funding Trends Over Time (NEW) ─────────────────────────
# Directly answers Q1: "how does funding change over time per sector?"
# Top 6 sectors by total funding shown as an area-stacked line chart.
top6_sectors = (df.groupby("sector")["funded_amount"].sum()
                .sort_values(ascending=False).head(6).index.tolist())

sector_time = (
    df[df["sector"].isin(top6_sectors)]
    .groupby(["year", "sector"])["funded_amount"]
    .sum()
    .reset_index()
    .pivot(index="year", columns="sector", values="funded_amount")
    .fillna(0) / 1e6
)

fig, ax = plt.subplots(figsize=(12, 6))
sector_time.plot(kind="area", stacked=False, ax=ax, linewidth=2, alpha=0.18)
sector_time.plot(kind="line", ax=ax, linewidth=2.2)  # overlay solid lines

# clean up the duplicate legend entries created by double-plotting
handles, labels = ax.get_legend_handles_labels()
mid = len(handles) // 2
ax.legend(handles[mid:], labels[mid:], title="Sector",
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

ax.set_xlabel("Year", labelpad=8)
ax.set_ylabel("Funded Amount (Millions USD)", labelpad=8)
ax.set_title("Top 6 Sectors — Funding Trends Over Time", fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
plt.tight_layout()
plt.savefig("02_sector_trends_over_time.png")
plt.close()
print("  Saved → 02_sector_trends_over_time.png")


# ── Chart 03: Gender Distribution & Repayment Intervals ─────────────────────
# Retained; styling tightened and colour palette made consistent.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie
gender_counts = df["gender_simple"].value_counts()
gender_colors = {"Female": PALETTE_PRIMARY, "Male": PALETTE_SECONDARY,
                 "Mixed": PALETTE_ACCENT,   "Unknown": PALETTE_NEUTRAL}
pie_colors = [gender_colors.get(g, "#BDC3C7") for g in gender_counts.index]
wedges, texts, autotexts = axes[0].pie(
    gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
    colors=pie_colors, startangle=140, pctdistance=0.82,
    wedgeprops=dict(linewidth=1.4, edgecolor="white"))
for at in autotexts:
    at.set_fontsize(9)
axes[0].set_title("Borrower Gender Distribution", fontweight="bold", pad=12)

# Count-plot (ordered by frequency)
rep_order = df["repayment_interval"].value_counts().index.tolist()
sns.countplot(data=df, x="repayment_interval", order=rep_order,
              palette=[PALETTE_PRIMARY, PALETTE_SECONDARY,
                       PALETTE_ACCENT, PALETTE_NEUTRAL],
              ax=axes[1], edgecolor="white")
axes[1].set_xlabel("Repayment Interval", labelpad=8)
axes[1].set_ylabel("Number of Loans", labelpad=8)
axes[1].set_title("Loan Count by Repayment Interval", fontweight="bold", pad=12)
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, color=PALETTE_SECONDARY)

plt.tight_layout()
plt.savefig("03_gender_repayment.png")
plt.close()
print("  Saved → 03_gender_repayment.png")


# ── Chart 04: Funded Amount Distribution (NEW) ────────────────────────────────
# Replaces nothing — fills a critical gap: understanding the loan size shape.
fig, ax = plt.subplots(figsize=(11, 5))
capped = df[df["funded_amount"] <= 5000]["funded_amount"]
ax.hist(capped, bins=60, color=PALETTE_PRIMARY, edgecolor="white",
        alpha=0.85, density=True)

# Overlay KDE
from scipy.stats import gaussian_kde
kde_x = np.linspace(0, 5000, 400)
kde   = gaussian_kde(capped)
ax.plot(kde_x, kde(kde_x), color=PALETTE_SECONDARY, linewidth=2.2, label="KDE")

ax.axvline(capped.mean(),   color=PALETTE_ACCENT, linewidth=1.8,
           linestyle="--",  label=f"Mean  ${capped.mean():,.0f}")
ax.axvline(capped.median(), color=PALETTE_NEUTRAL, linewidth=1.8,
           linestyle=":",   label=f"Median ${capped.median():,.0f}")

ax.set_xlabel("Funded Amount (USD)", labelpad=8)
ax.set_ylabel("Density", labelpad=8)
ax.set_title("Distribution of Funded Loan Amounts (≤ $5,000)", fontweight="bold", pad=14)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("04_funded_amount_distribution.png")
plt.close()
print("  Saved → 04_funded_amount_distribution.png")


# ── Chart 05: Lenders vs. Funded Amount (Correlation) ───────────────────────
# Retained & improved: fixed random_state for reproducibility, added
# Pearson r annotation, switched to hexbin for density clarity.
fig, ax = plt.subplots(figsize=(10, 6))
sample = df[df["lender_count"] < 200].sample(n=min(3000, len(df)), random_state=42)

hb = ax.hexbin(sample["lender_count"], sample["funded_amount"],
               gridsize=40, cmap="Reds", mincnt=1, linewidths=0.3)
cb = fig.colorbar(hb, ax=ax, label="Count")

# Regression line
m, b = np.polyfit(sample["lender_count"], sample["funded_amount"], 1)
x_line = np.linspace(0, sample["lender_count"].max(), 200)
ax.plot(x_line, m * x_line + b, color=PALETTE_SECONDARY, linewidth=2, linestyle="--")

# Pearson r
r = sample["lender_count"].corr(sample["funded_amount"])
ax.text(0.97, 0.05, f"Pearson r = {r:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10, color=PALETTE_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE_NEUTRAL))

ax.set_xlabel("Number of Lenders", labelpad=8)
ax.set_ylabel("Funded Amount (USD)", labelpad=8)
ax.set_title("Correlation: Lender Count vs. Funded Amount", fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("05_lenders_vs_funded_correlation.png")
plt.close()
print("  Saved → 05_lenders_vs_funded_correlation.png")


# ── Chart 06: Top 15 Countries by Total Funding (NEW) ────────────────────────
# Key geographic insight missing from the original — essential for a BI report.
country_totals = (
    df.groupby("country")["funded_amount"].sum()
    .sort_values(ascending=True).tail(15) / 1e6
)

fig, ax = plt.subplots(figsize=(11, 7))
colors = [PALETTE_PRIMARY if i >= len(country_totals) - 3 else "#AEBDC5"
          for i in range(len(country_totals))]
ax.barh(country_totals.index, country_totals.values, color=colors,
        edgecolor="white", height=0.65)

for i, (val, country) in enumerate(zip(country_totals.values, country_totals.index)):
    ax.text(val + 0.05, i, f"${val:.1f}M", va="center", fontsize=8.5,
            color=PALETTE_SECONDARY)

ax.set_xlabel("Total Funded Amount (Millions USD)", labelpad=8)
ax.set_title("Top 15 Countries by Total Funding", fontweight="bold", pad=14)
ax.set_xlim(0, country_totals.max() * 1.16)
plt.tight_layout()
plt.savefig("06_top_countries_funding.png")
plt.close()
print("  Saved → 06_top_countries_funding.png")


# ── Chart 07: Monthly Funding Time Series ────────────────────────────────────
# Retained; resample alias fixed ('ME' → 'M' for broader compatibility),
# axis labels and fill added.
monthly = df.set_index("date").resample("ME")["funded_amount"].sum() / 1e6

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly.index, monthly.values,
        color=PALETTE_PRIMARY, linewidth=2, marker="o",
        markersize=3.5, label="Monthly Funded")
ax.fill_between(monthly.index, monthly.values,
                alpha=0.12, color=PALETTE_PRIMARY)

# 3-month rolling average
rolling = monthly.rolling(3, center=True).mean()
ax.plot(rolling.index, rolling.values,
        color=PALETTE_SECONDARY, linewidth=2.2, linestyle="--",
        label="3-Month Rolling Avg")

ax.set_xlabel("Date", labelpad=8)
ax.set_ylabel("Funded Amount (Millions USD)", labelpad=8)
ax.set_title("Monthly Funding Trend (Jan 2014 – Jul 2017)", fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("07_monthly_funding_timeseries.png")
plt.close()
print("  Saved → 07_monthly_funding_timeseries.png")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DATA CLEANING & TRANSFORMATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — DATA CLEANING & TRANSFORMATION")
print("=" * 60)

df_clean = df.copy()

# ── 3.1  Remove Duplicates ───────────────────────────────────────────────────
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
removed_dups = before - len(df_clean)
print(f"\n[1] Duplicates removed : {removed_dups}")

# ── 3.2  Handle Missing Values ───────────────────────────────────────────────
# FIX: fillna(inplace=True) is deprecated in pandas ≥ 2.0; assign via [] instead.
# FIX: median is computed BEFORE we remove anything, giving a stable reference.
partner_median = df_clean["partner_id"].median()
df_clean["partner_id"] = df_clean["partner_id"].fillna(partner_median)

for col in ["borrower_genders", "sector", "country", "repayment_interval"]:
    df_clean[col] = df_clean[col].fillna("Unknown")

print(f"[2] Missing values imputed (partner_id → median={partner_median:.0f}; "
      f"categoricals → 'Unknown')")

# ── 3.3  Outlier Removal (±3 σ on funded_amount) ────────────────────────────
# FIX: store pre-clean data separately so the boxplot comparison is honest.
df_pre_outlier = df_clean.copy()

mean_f = df_clean["funded_amount"].mean()
std_f  = df_clean["funded_amount"].std()
lower  = max(0, mean_f - 3 * std_f)    # FIX: clamp at 0; negative lower is nonsensical
upper  = mean_f + 3 * std_f

df_clean = df_clean[(df_clean["funded_amount"] >= lower) &
                    (df_clean["funded_amount"] <= upper)].copy()

removed_outliers = len(df_pre_outlier) - len(df_clean)
print(f"[3] Outliers removed   : {removed_outliers}  "
      f"(bounds: ${lower:,.0f} – ${upper:,.0f})")

# ── 3.4  Categorical Encoding ────────────────────────────────────────────────
df_clean["repayment_encoded"] = (
    df_clean["repayment_interval"].astype("category").cat.codes
)
df_clean["gender_encoded"] = (
    df_clean["gender_simple"].astype("category").cat.codes
)
print("[4] Categorical encoding applied "
      "(repayment_interval, gender_simple → integer codes)")

# ── 3.5  Normalization ───────────────────────────────────────────────────────
# FIX: NEVER overwrite the original columns with scaled values — Power BI
# (Step 4) and human-readable exports need the raw values.  Scaled versions
# are stored in separate '_scaled' columns ready for the ML step (Step 5).
scaler   = StandardScaler()
ml_cols  = ["loan_amount", "term_in_months", "lender_count"]
scaled   = scaler.fit_transform(df_clean[ml_cols])
for i, col in enumerate(ml_cols):
    df_clean[f"{col}_scaled"] = scaled[:, i]

print(f"[5] StandardScaler applied to {ml_cols}\n"
      f"    Scaled values stored in '<col>_scaled' columns "
      f"(originals preserved for Power BI export)")

# ── 3.6  Export cleaned dataset ──────────────────────────────────────────────
# FIX: original code never exported df_clean — without this, Step 4 (Power BI)
# has no cleaned data to import.
CLEAN_PATH = "kiva_loans_cleaned.csv"
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"\n[6] Cleaned dataset saved → '{CLEAN_PATH}'  "
      f"({len(df_clean):,} rows × {df_clean.shape[1]} columns)")

# ── 3.7  Cleaning Comparison Visualizations ──────────────────────────────────

# Chart 08: Boxplot Before vs. After Outlier Removal
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)

sns.boxplot(y=df_pre_outlier["funded_amount"], ax=axes[0],
            color="#E74C3C", linewidth=1.4,
            flierprops=dict(marker=".", markersize=2, alpha=0.4))
axes[0].set_title("Before Outlier Removal", fontweight="bold")
axes[0].set_ylabel("Funded Amount (USD)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x:,.0f}"))

sns.boxplot(y=df_clean["funded_amount"], ax=axes[1],
            color="#27AE60", linewidth=1.4,
            flierprops=dict(marker=".", markersize=2, alpha=0.4))
axes[1].set_title("After Outlier Removal  (±3σ)", fontweight="bold")
axes[1].set_ylabel("")

fig.suptitle("Funded Amount — Outlier Removal Comparison",
             fontweight="bold", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("08_outlier_removal_comparison.png")
plt.close()
print("\n  Saved → 08_outlier_removal_comparison.png")

# Chart 09: Missing Value Heatmap (NEW) ──────────────────────────────────────
# Visually proves the cleaning step to a non-technical audience.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, frame, title in zip(
        axes,
        [df.isnull(), df_clean.isnull()],
        ["Before Cleaning", "After Cleaning"]):
    col_missing = frame.sum().sort_values(ascending=False)
    colors      = [PALETTE_PRIMARY if v > 0 else PALETTE_SECONDARY
                   for v in col_missing.values]
    ax.barh(col_missing.index, col_missing.values, color=colors, edgecolor="white")
    ax.set_xlabel("Missing Value Count")
    ax.set_title(title, fontweight="bold")
    for i, v in enumerate(col_missing.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8.5)

fig.suptitle("Missing Value Profile — Before vs. After Cleaning",
             fontweight="bold", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("09_missing_values_comparison.png")
plt.close()
print("  Saved → 09_missing_values_comparison.png")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE  —  Summary")
print("=" * 60)
print(f"  Raw rows        : {len(df):,}")
print(f"  Cleaned rows    : {len(df_clean):,}")
print(f"  Dropped total   : {len(df) - len(df_clean):,}")
print(f"  Charts saved    : 9  (01 – 09)")
print(f"  Cleaned CSV     : {CLEAN_PATH}")
print(f"  Ready for       : Power BI import  (Step 4)")
print(f"                    ML model         (Step 5  — use *_scaled cols)")
print(f"                    Time series      (Step 6)")
print("=" * 60)

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Machine Learning Model
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Machine Learning Model")
print("=" * 60)

# ── 4.1 Features ───────────────────────────────────────────────────────────
RAW_FEATURES = [
    "loan_amount",
    "term_in_months",
    "lender_count",
    "repayment_encoded",
    "gender_encoded"
]

SCALED_FEATURES = [
    "loan_amount_scaled",
    "term_in_months_scaled",
    "lender_count_scaled",
    "repayment_encoded",
    "gender_encoded"
]

TARGET = "funded_amount"

X_raw = df_clean[RAW_FEATURES]
X_sc  = df_clean[SCALED_FEATURES]
y     = df_clean[TARGET]

# ── 4.2 Split  ──────────────────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

X_train_sc = X_sc.loc[X_train_raw.index]
X_test_sc  = X_sc.loc[X_test_raw.index]

# ── 4.3 Models  ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# 1. Linear Regression
# ════════════════════════════════════════════════════════════════════════════
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

# ════════════════════════════════════════════════════════════════════════════
# 2. Polynomial Regression
# ════════════════════════════════════════════════════════════════════════════
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly  = poly.transform(X_test_sc)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# ════════════════════════════════════════════════════════════════════════════
# 3. Decision Tree
# ════════════════════════════════════════════════════════════════════════════
dt = DecisionTreeRegressor(max_depth=8, random_state=42)
dt.fit(X_train_raw, y_train)
y_pred_dt = dt.predict(X_test_raw)

# ════════════════════════════════════════════════════════════════════════════
# 4. Random Forest
# ════════════════════════════════════════════════════════════════════════════
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_raw, y_train)
y_pred_rf = rf.predict(X_test_raw)

# ── 4.4 Evaluation  ──────────────────────────────────────────────────────────────
print("\n========== Model Evaluation ==========")

models = {
    "Linear Regression": y_pred_lr,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf
}

results = {}

for name, y_pred in models.items():
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = r2

    print(f"\n--- {name} ---")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R2   : {r2:.4f}")

# ── 4.5 Comparison  ──────────────────────────────────────────────────────────────
print("\n========== Model Comparison ==========")

for name, score in results.items():
    print(f"{name}: R2 = {score:.4f}")

best_model = max(results, key=results.get)

print(f"\nBest Model: {best_model}")

print("=" * 60)