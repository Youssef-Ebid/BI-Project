# PROJECT OBJECTIVE:
# Analyze the Kiva microfinance loan dataset to generate meaningful business
# insights that help the Kiva organization understand their lending patterns,
# identify high-impact sectors, and predict future loan funding amounts.
#
# KEY QUESTIONS TO ANSWER:
#   1. Which sectors receive the highest funding, and how does this change over time?
#   2. Is there a correlation between the number of lenders and the funded amount?
#   3. Can we predict future loan amounts based on historical data?
#
# PROJECT STEPS:
#   Part 1  - Project setup: import libraries and load the dataset
#   Part 2  - Data Exploration: summary statistics & visualizations
#   Part 3  - Data Cleaning: handle missing values, duplicates, outliers
#   Part 4  - Advanced Visualization (PowerBI-ready exports)
#   Part 5  - Machine Learning: predict funded_amount
#   Part 6  - Time Series Analysis: funded_amount trends over time
#
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
df = pd.read_csv("masked_kiva_loans.csv", parse_dates=["date"])

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")