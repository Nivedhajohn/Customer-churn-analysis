# src/eda.py
"""
Exploratory Data Analysis (EDA) with visualizations for Customer Churn dataset.
Generates bar charts, pie charts, histograms, correlation heatmaps, etc.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Config
# -------------------
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "customer churn data.csv")

# -------------------
# Load Data
# -------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")
    
df = pd.read_csv(data_path)

# Ensure target column exists
if "Churn" not in df.columns:
    raise KeyError("Expected a target column named 'Churn' in dataset")

# -------------------
# Visualization 1: Churn Distribution (Pie + Bar)
# -------------------
plt.figure()
df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"])
plt.title("Churn Distribution (Pie Chart)")
plt.ylabel("")  # remove y-label
plt.show()

plt.figure()
sns.countplot(x="Churn", data=df, palette="Set2")
plt.title("Churn Distribution (Bar Chart)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# -------------------
# Visualization 2: Contract Type vs Churn (Bar)
# -------------------
if "Contract" in df.columns:
    plt.figure()
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set1")
    plt.title("Contract Type vs Churn")
    plt.xlabel("Contract Type")
    plt.ylabel("Count")
    plt.show()

# -------------------
# Visualization 3: Monthly Charges by Churn (Boxplot)
# -------------------
if "MonthlyCharges" in df.columns:
    plt.figure()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3")
    plt.title("Monthly Charges vs Churn")
    plt.show()

# -------------------
# Visualization 4: Tenure Distribution (Histogram)
# -------------------
if "tenure" in df.columns:
    plt.figure()
    sns.histplot(df, x="tenure", hue="Churn", bins=30, kde=True, palette="husl")
    plt.title("Tenure Distribution by Churn")
    plt.show()

# -------------------
# Visualization 5: Correlation Heatmap (Numeric Features)
# -------------------
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()
