import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("C:/Users/Kartik/Desktop/Electric_Vehicle_Population_By_County.csv")


print("\n==============================")
print("✅ Dataset Loaded Successfully")
print("==============================\n")


df.head() # top 5 rows
# Show first few rows
print(df.head())
print("==============================\n")


# Check number of rows and columns
print("📏 Dataset Shape (Rows, Columns):")
print(df.shape)
print("\n------------------------------\n")


# Check data types and memory usage
print("🧠 Dataset Info:")
df.info()
print("\n------------------------------\n")


# Check missing values
print("🔍 Missing Values Per Column:")
print(df.isnull().sum())
print("\n------------------------------\n")


# Compute Q1 and Q3 for outlier detection
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("📊 Outlier Bounds for 'Percent Electric Vehicles':")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("\n------------------------------\n")


# Identify outliers
outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("🚨 Number of Outliers in 'Percent Electric Vehicles':", outliers.shape[0])
print("\n------------------------------\n")


# Data Preprocessing
print("🔧 Starting Data Preprocessing...\n")

# Convert "Date" to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows with invalid dates
df = df[df['Date'].notnull()]

# Remove rows with missing target values
df = df[df['Electric Vehicle (EV) Total'].notnull()]

# Fill missing County and State
df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')

# Confirm missing values are filled
print("✅ Missing Values After Fill:")
print(df[['County', 'State']].isnull().sum())
print("\n------------------------------\n")


# Show first few rows
print("🔎 Sample Data After Cleaning:")
print(df.head())
print("\n------------------------------\n")


# Cap outliers using IQR bounds
df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                 np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))

# Confirm outliers removed
outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("✅ Number of Outliers After Capping:", outliers.shape[0])
print("\n==============================")
print("🎉 Data Cleaning Complete")
print("==============================\n")