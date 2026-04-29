import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

df = pd.read_csv("masked_kiva_loans.csv", parse_dates=["date"])

# PART 1: INITIAL ANALYSIS
print("=== Initial Analysis (EDA) ===")

# check gender
g = df["borrower_genders"].str.lower()
df["gender_simple"] = "" # Modified ONLY to avoid TypeError
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

# VISUALIZATIONS
# Chart 1: Top 10 Sectors
plt.figure(figsize=(10, 5))
sector_data = df.groupby("sector")["funded_amount"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=sector_data.values / 1e6, y=sector_data.index, color="#C0392B")
plt.title("Top 10 Sectors by Funding (Millions USD)", fontweight="bold")
plt.savefig('01_initial_sectors.png')
plt.close()

# Chart 2: Gender & Repayment
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
counts = df["gender_simple"].value_counts()
color_map = {"Female": "#C0392B", "Male": "#2C3E50", "Mixed": "#95A5A6"}
pie_colors = [color_map.get(label, "#F1C40F") for label in counts.index]
ax[0].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=140)
ax[0].set_title("Gender Distribution", fontweight="bold")

sns.countplot(data=df, x="repayment_interval", ax=ax[1], palette="viridis")
ax[1].set_title("Repayment Intervals", fontweight="bold")
plt.savefig('02_initial_gender_repayment.png')
plt.close()

# Chart 3: Lenders vs Funded
plt.figure(figsize=(10, 5))
sample = df[df["lender_count"] < 200].sample(min(2000, len(df)))
sns.regplot(data=sample, x="lender_count", y="funded_amount", scatter_kws={'alpha':0.3, 'color':'#C0392B'}, line_kws={'color':'#2C3E50'})
plt.title("Correlation: Lenders vs. Funded Amount", fontweight="bold")
plt.savefig('03_initial_correlation.png')
plt.close()

# Chart 4: Time Series
monthly = df.set_index('date').resample('ME')['funded_amount'].sum() / 1e6
plt.figure(figsize=(12, 4))
monthly.plot(color="#C0392B", linewidth=2, marker='o')
plt.fill_between(monthly.index, monthly.values, alpha=0.1)
plt.title("Monthly Funding Trends (Millions USD)", fontweight="bold")
plt.savefig('04_initial_timeseries.png')
plt.close()


# PART 2: DATA CLEANING & TRANSFORMATION (Abdallah's Contribution)
print("\n=== Phase 2: Data Cleaning & Transformation ===")

# 1. Remove Duplicates
before_dup = len(df)
df.drop_duplicates(inplace=True)
print(f"1: Removed {before_dup - len(df)} duplicate rows.")

# 2. Handle Missing Values
df['partner_id'].fillna(df['partner_id'].median(), inplace=True)
for col in ['borrower_genders', 'sector', 'country', 'repayment_interval']:
    df[col].fillna("unknown", inplace=True)
print("2: Missing values imputed using Median and 'unknown' tags.")

# 3. Outlier Removal (Using 3 Standard Deviations)
mean_f = df['funded_amount'].mean()
std_f = df['funded_amount'].std()
upper = mean_f + (3 * std_f)
lower = mean_f - (3 * std_f)

df_clean = df[(df['funded_amount'] >= lower) & (df['funded_amount'] <= upper)].copy()
print(f"3: Removed {len(df) - len(df_clean)} outliers (Beyond 3 Sigma).")

# 4. Data Transformation (Label Encoding & Date Preparation)
df_clean['repayment_encoded'] = df_clean['repayment_interval'].astype('category').cat.codes
df_clean['date'] = pd.to_datetime(df_clean['date']) # ضروري عشان جراف الـ Time Series الجديد يشتغل صح
print("4: Categorical encoding and date transformation completed.")

# 5. Normalization (Scaling)
scaler = StandardScaler()
num_cols = ['loan_amount', 'term_in_months', 'lender_count']
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
print(f"5: Features scaled using StandardScaler: {num_cols}")

# 6. SAVE CLEANED VISUALIZATIONS
# Chart 5: Outliers Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['funded_amount'], color='red').set_title("Before Cleaning")
plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['funded_amount'], color='green').set_title("After Cleaning")
plt.savefig('05_cleaning_outliers.png')
plt.close()

# Chart 6: Cleaned Time Series
plt.figure(figsize=(12, 5))
monthly_clean = df_clean.set_index('date').resample('ME')['funded_amount'].sum() / 1e6
monthly_clean.plot(color="green", linewidth=2, marker='s')
plt.title("Monthly Funding Trends (AFTER CLEANING)")
plt.ylabel("Millions USD")
plt.savefig('06_cleaned_timeseries.png')
plt.close()

