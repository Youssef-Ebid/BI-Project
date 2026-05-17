import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────
PRIMARY   = "crimson"
SECONDARY = "midnightblue"
ACCENT    = "orange"
NEUTRAL   = "gray"

sns.set_theme(style="whitegrid", font_scale=1.05)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"
})

DATA_PATH = "masked_kiva_loans.csv"

# ── STEP 1 — EDA
print("\nSTEP 1 — EDA")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

g = df["borrower_genders"].str.lower().fillna("")

female = g.str.contains("female")
male   = g.str.contains(r"\bmale\b", regex=True)

df["gender_simple"] = np.select(
    [
        female & male,
        female & ~male,
        male & ~female
    ],
    ["Mixed", "Female", "Male"],
    default="Unknown"
)

df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")

print(f"Rows : {len(df):,}")
print(f"Countries : {df['country'].nunique()}")
print(f"Sectors : {df['sector'].nunique()}")
print("=" * 60)

# ── STEP 2 — VISUALIZATIONS
print("\nSTEP 2 — Visualizations")

def save_plot(name):
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

# 1 ─ Top Sectors
sector_totals = (
    df.groupby("sector")["funded_amount"]
    .sum()
    .sort_values(ascending=True)
    .tail(10) / 1e6
)

plt.figure(figsize=(11, 6))

bars = plt.barh(
    sector_totals.index,
    sector_totals.values,
    color=PRIMARY,
    edgecolor="white"
)

for b in bars:
    plt.text(
        b.get_width() + 0.05,
        b.get_y() + b.get_height()/2,
        f"${b.get_width():.1f}M",
        va="center"
    )

plt.xlabel("Funded Amount (Millions USD)")
plt.title("Top 10 Sectors by Total Funding", fontweight="bold")

save_plot("01_sector_funding_bar.png")

# 2 ─ Sector Trends
top6 = (
    df.groupby("sector")["funded_amount"]
    .sum()
    .sort_values(ascending=False)
    .head(6)
    .index
)

sector_time = (
    df[df["sector"].isin(top6)]
    .groupby(["year", "sector"])["funded_amount"]
    .sum()
    .reset_index()
    .pivot(index="year", columns="sector", values="funded_amount")
    .fillna(0) / 1e6
)

fig, ax = plt.subplots(figsize=(12, 6))

sector_time.plot(kind="area", stacked=False, alpha=0.18, ax=ax)
sector_time.plot(kind="line", linewidth=2, ax=ax)

handles, labels = ax.get_legend_handles_labels()
mid = len(handles) // 2

ax.legend(handles[mid:], labels[mid:], bbox_to_anchor=(1.01, 1))
ax.set_title("Top 6 Sectors — Funding Trends", fontweight="bold")

save_plot("02_sector_trends_over_time.png")

# 3 ─ Gender & Repayment
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

gender_counts = df["gender_simple"].value_counts()

axes[0].pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    colors=[PRIMARY, SECONDARY, ACCENT, NEUTRAL][:len(gender_counts)],
    startangle=140
)

axes[0].set_title("Borrower Gender Distribution", fontweight="bold")

rep_order = df["repayment_interval"].value_counts().index

sns.countplot(
    data=df,
    x="repayment_interval",
    order=rep_order,
    palette=[PRIMARY, SECONDARY, ACCENT, NEUTRAL],
    ax=axes[1]
)

axes[1].set_title("Loan Count by Repayment Interval", fontweight="bold")

save_plot("03_gender_repayment.png")

# 4 ─ Distribution
plt.figure(figsize=(11, 5))

capped = df[df["funded_amount"] <= 5000]["funded_amount"]

plt.hist(
    capped,
    bins=60,
    color=PRIMARY,
    edgecolor="white",
    alpha=0.85,
    density=True
)

x = np.linspace(0, 5000, 400)

plt.plot(
    x,
    gaussian_kde(capped)(x),
    color=SECONDARY,
    linewidth=2
)

plt.axvline(capped.mean(), color=ACCENT, linestyle="--")
plt.axvline(capped.median(), color=NEUTRAL, linestyle=":")

plt.title("Distribution of Funded Loan Amounts", fontweight="bold")

save_plot("04_funded_amount_distribution.png")

# 5 ─ Correlation
fig, ax = plt.subplots(figsize=(10, 6))

sample = df[df["lender_count"] < 200].sample(
    n=min(3000, len(df)),
    random_state=42
)

hb = ax.hexbin(
    sample["lender_count"],
    sample["funded_amount"],
    gridsize=40,
    cmap="Reds",
    mincnt=1
)

fig.colorbar(hb, ax=ax)

m, b = np.polyfit(
    sample["lender_count"],
    sample["funded_amount"],
    1
)

x = np.linspace(0, sample["lender_count"].max(), 200)

ax.plot(x, m*x+b, color=SECONDARY, linestyle="--")

r = sample["lender_count"].corr(sample["funded_amount"])

ax.text(
    0.97,
    0.05,
    f"r = {r:.3f}",
    transform=ax.transAxes,
    ha="right"
)

ax.set_title("Lender Count vs Funded Amount", fontweight="bold")

save_plot("05_lenders_vs_funded_correlation.png")

# 6 ─ Top Countries
country_totals = (
    df.groupby("country")["funded_amount"]
    .sum()
    .sort_values(ascending=True)
    .tail(15) / 1e6
)

plt.figure(figsize=(11, 7))

plt.barh(
    country_totals.index,
    country_totals.values,
    color=[
        PRIMARY if i >= len(country_totals)-3 else "lightgray"
        for i in range(len(country_totals))
    ],
    edgecolor="white"
)

plt.title("Top 15 Countries by Total Funding", fontweight="bold")

save_plot("06_top_countries_funding.png")

# 7 ─ Monthly Trend
monthly = (
    df.set_index("date")
    .resample("ME")["funded_amount"]
    .sum() / 1e6
)

plt.figure(figsize=(13, 5))

plt.plot(
    monthly.index,
    monthly.values,
    color=PRIMARY,
    linewidth=2,
    marker="o",
    markersize=3
)

plt.fill_between(
    monthly.index,
    monthly.values,
    alpha=0.12,
    color=PRIMARY
)

rolling = monthly.rolling(3, center=True).mean()

plt.plot(
    rolling.index,
    rolling.values,
    color=SECONDARY,
    linestyle="--",
    linewidth=2
)

plt.title("Monthly Funding Trend", fontweight="bold")

save_plot("07_monthly_funding_timeseries.png")
print("=" * 60)

# ── STEP 3 — CLEANING
print("\nSTEP 3 — Cleaning")

df_clean = df.drop_duplicates().copy()

# Fill missing partner_id
partner_median = df_clean["partner_id"].median()

df_clean["partner_id"] = (
    df_clean["partner_id"]
    .fillna(partner_median)
)

# Fill categorical missing values
for c in [
    "borrower_genders",
    "sector",
    "country",
    "repayment_interval"
]:
    df_clean[c] = df_clean[c].fillna("Unknown")

# Save before outlier removal
df_before = df_clean.copy()

# ── Outlier Removal ─────────────────────────────────────────────
mean_f = df_clean["funded_amount"].mean()
std_f  = df_clean["funded_amount"].std()

lower = max(0, mean_f - 3 * std_f)
upper = mean_f + 3 * std_f

df_clean = df_clean[
    (df_clean["funded_amount"] >= lower) &
    (df_clean["funded_amount"] <= upper)
].copy()

# ── Encoding ────────────────────────────────────────────────────
df_clean["repayment_encoded"] = (
    df_clean["repayment_interval"]
    .astype("category")
    .cat.codes
)

df_clean["gender_encoded"] = (
    df_clean["gender_simple"]
    .astype("category")
    .cat.codes
)

# ── Save Cleaned Dataset ────────────────────────────────────────
CLEAN_PATH = "kiva_loans_cleaned.csv"

df_clean.to_csv(CLEAN_PATH, index=False)

# ── Visualization 1: Outlier Comparison ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

sns.boxplot(
    y=df_before["funded_amount"],
    ax=axes[0],
    color="red"
)

axes[0].set_title("Before Outlier Removal")

sns.boxplot(
    y=df_clean["funded_amount"],
    ax=axes[1],
    color="green"
)

axes[1].set_title("After Outlier Removal")

save_plot("08_outlier_removal_comparison.png")

# ── Visualization 2: Missing Values ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, frame, title in zip(
    axes,
    [df.isnull(), df_clean.isnull()],
    ["Before Cleaning", "After Cleaning"]
):

    missing = frame.sum().sort_values(ascending=False)

    ax.barh(
        missing.index,
        missing.values,
        color=[
            PRIMARY if v > 0 else SECONDARY
            for v in missing.values
        ]
    )

    ax.set_title(title)

save_plot("09_missing_values_comparison.png")

print(f"Cleaned rows : {len(df_clean):,}")
print("=" * 60)

# ── STEP 4 — MACHINE LEARNING
print("\nSTEP 4 — Machine Learning")

# ── 1 Features ──────────────────────────────────────────────────
RAW_FEATURES = [
    "loan_amount",
    "term_in_months",
    "lender_count",
    "repayment_encoded",
    "gender_encoded"
]

TARGET = "funded_amount"

X_raw = df_clean[RAW_FEATURES]
y = df_clean[TARGET]

# ── 2 Split ─────────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.2,
    random_state=42
)

# ── 3 Scaling ───────────────────────────────────────────────────
scale_cols = [
    "loan_amount",
    "term_in_months",
    "lender_count"
]

X_train_sc = X_train_raw.copy()
X_test_sc  = X_test_raw.copy()

scaler = StandardScaler()

X_train_sc[scale_cols] = scaler.fit_transform(
    X_train_sc[scale_cols]
)

X_test_sc[scale_cols] = scaler.transform(
    X_test_sc[scale_cols]
)

# ── 4 Models ────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# 1. Linear Regression
# ════════════════════════════════════════════════════════════════
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

# ════════════════════════════════════════════════════════════════
# 2. Polynomial Regression
# ════════════════════════════════════════════════════════════════
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly  = poly.transform(X_test_sc)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# ════════════════════════════════════════════════════════════════
# 3. Decision Tree
# ════════════════════════════════════════════════════════════════
dt = DecisionTreeRegressor(
    max_depth=8,
    random_state=42
)
dt.fit(X_train_raw, y_train)
y_pred_dt = dt.predict(X_test_raw)

# ════════════════════════════════════════════════════════════════
# 4. Random Forest
# ════════════════════════════════════════════════════════════════
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train_raw, y_train)
y_pred_rf = rf.predict(X_test_raw)

# ── 5 Evaluation ────────────────────────────────────────────────
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

# ── 6 Comparison ────────────────────────────────────────────────
print("\n========== Model Comparison ==========")

for name, score in results.items():
    print(f"{name}: R2 = {score:.4f}")

best_model = max(results, key=results.get)

print(f"\nBest Model: {best_model}")

print("=" * 60)