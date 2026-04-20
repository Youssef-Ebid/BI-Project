import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("masked_kiva_loans.csv", parse_dates=["date"])

# check gender
g = df["borrower_genders"].str.lower()
df["gender_simple"] = np.nan
has_female = g.str.contains("female", na=False)
has_male = g.str.contains(r"\bmale\b", na=False)
df.loc[has_female & has_male, "gender_simple"] = "Mixed"
df.loc[has_female & ~has_male, "gender_simple"] = "Female"
df.loc[has_male & ~has_female, "gender_simple"] = "Male"

# STATS
print(f"Dataset: {df.shape[0]:,} rows")
print("Gender column check:")
print(df["gender_simple"].value_counts(dropna=False))
print(df[["funded_amount", "loan_amount", "term_in_months", "lender_count"]].describe().round(2))

print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nTotal Loans    : {len(df):,}")
print(f"Total Funded   : ${df['funded_amount'].sum():,.0f}")
print(f"Avg Loan Amount: ${df['funded_amount'].mean():,.0f}")
print(f"Countries      : {df['country'].nunique()}")
print(f"Sectors        : {df['sector'].nunique()}")

#VISUALIZATIONS
# Chart 1: Top 10 Sectors
plt.figure(figsize=(10, 5))
sector_data = df.groupby("sector")["funded_amount"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=sector_data.values / 1e6, y=sector_data.index, color="#C0392B")
plt.title("Top 10 Sectors by Funding (Millions USD)", fontweight="bold")
plt.show()

# Chart 2: Gender & Repayment
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# Gender Pie
counts = df["gender_simple"].value_counts()
color_map = {"Female": "#C0392B", "Male": "#2C3E50", "Mixed": "#95A5A6"}
pie_colors = [color_map[label] for label in counts.index]
ax[0].pie(counts, labels=counts.index, autopct='%1.1f%%',colors=pie_colors, startangle=140)
ax[0].set_title("Gender Distribution", fontweight="bold")

# Repayment Intervals
sns.countplot(data=df, x="repayment_interval", ax=ax[1], palette="viridis")
ax[1].set_title("Repayment Intervals", fontweight="bold")
plt.show()

# Chart 3: Lenders vs Funded
plt.figure(figsize=(10, 5))
sample = df[df["lender_count"] < 200].sample(min(2000, len(df)))
sns.regplot(data=sample, x="lender_count", y="funded_amount",scatter_kws={'alpha':0.3, 'color':'#C0392B'}, line_kws={'color':'#2C3E50'})
plt.title("Correlation: Lenders vs. Funded Amount", fontweight="bold")
plt.show()

# Chart 4: Time Series
monthly = df.set_index('date').resample('M')['funded_amount'].sum() / 1e6
plt.figure(figsize=(12, 4))
monthly.plot(color="#C0392B", linewidth=2, marker='o')
plt.fill_between(monthly.index, monthly.values, alpha=0.1)
plt.title("Monthly Funding Trends (Millions USD)", fontweight="bold")
plt.show()