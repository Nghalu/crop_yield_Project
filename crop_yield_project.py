import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
pesticides_df = pd.read_csv('pesticides.csv')
rainfaill_df = pd.read_csv('rainfall.csv')
temp_df = pd.read_csv('temp.csv')
yield_df = pd.read_csv('yield.csv')
yield1_df = pd.read_csv('yield1.csv')
print(pesticides_df.head())
print(rainfaill_df.head())
print(temp_df.head())
print(yield_df.head())
print(yield1_df.head())
# Explore the datasets
print(pesticides_df.info())
print(rainfaill_df.info())
print(temp_df.info())
print(yield_df.info())
print(yield1_df.info())
print(pesticides_df.describe())
print(rainfaill_df.describe())
print(temp_df.describe())
print(yield_df.describe())
print(yield1_df.describe())
# Data Cleaning and Preprocessing
print(pesticides_df.isnull().sum())
print(rainfaill_df.isnull().sum())
print(temp_df.isnull().sum())
print(yield_df.isnull().sum())
print(yield1_df.isnull().sum())
# Distribution analysis of key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Features', fontsize=16)
# Helper: find the most likely column name for a desired value
def find_column(df, keywords):
	"""Return first column name that contains any of the keywords (case-insensitive).
	If none found, return the first numeric column. Raises KeyError if no candidate.
	"""
	cols = df.columns.tolist()
	for k in keywords:
		for c in cols:
			if k.lower() in c.lower():
				return c
	# fallback: any numeric column
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if num_cols:
		return num_cols[0]
	raise KeyError(f"No column found matching keywords={keywords} in columns={cols}")
# crop_yield distribution (robust to different column names)
try:
	yield_col = find_column(yield_df, ['yield', 'value', 'hg/ha'])
	print(f"Using yield column: {yield_col}")
	sns.histplot(pd.to_numeric(yield_df[yield_col], errors='coerce').dropna(), bins=30, ax=axes[0, 0], color='skyblue')
	axes[0, 0].set_title('Crop Yield Distribution')
	axes[0, 0].set_xlabel('Yield (hg/ha)')
	axes[0, 0].set_ylabel('Frequency')
except KeyError as e:
	axes[0, 0].text(0.5, 0.5, 'Yield column not found', ha='center')
	axes[0, 0].set_title('Crop Yield Distribution (missing)')
# pesticide_used distribution (robust to different column names)
try:
	pesticides_col = find_column(pesticides_df, ['pesticide', 'value', 'pesticides_tonnes'])
	print(f"Using pesticides column: {pesticides_col}")
	sns.histplot(pd.to_numeric(pesticides_df[pesticides_col], errors='coerce').dropna(), bins=30, ax=axes[0, 1], color='salmon')
	axes[0, 1].set_title('Pesticide Used Distribution')
	axes[0, 1].set_xlabel('Pesticide Used')
	axes[0, 1].set_ylabel('Frequency')
except KeyError:
	axes[0, 1].text(0.5, 0.5, 'Pesticides column not found', ha='center')
	axes[0, 1].set_title('Pesticide Used Distribution (missing)')
 #select numeric column for correlation analysis
numeric_cols = yield_df.select_dtypes(include=[np.number]).columns.tolist()
# calculate correlation matrix
try:
    pesticides_col = find_column(pesticides_df, ['pesticide', 'value', 'pesticides_tonnes'])
    print(f"Using pesticides column: {pesticides_col}")
    sns.histplot(pd.to_numeric(pesticides_df[pesticides_col], errors='coerce').dropna(), bins=30, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Pesticide Used Distribution')
    axes[0, 1].set_xlabel('Pesticide Used')
    axes[0, 1].set_ylabel('Frequency')
except KeyError:
    axes[0, 1].text(0.5, 0.5, 'Pesticides column not found', ha='center')
    axes[0, 1].set_title('Pesticide Used Distribution (missing)')
if len(numeric_cols) >= 2:
    corr_matrix = yield_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0, 2])
    axes[0, 2].set_title('Correlation Matrix')