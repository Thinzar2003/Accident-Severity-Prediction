# ============================================================
# ACCIDENT SEVERITY PREDICTION
# Notebook 02 — Data Cleaning & Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

import os as _os
BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.makedirs(_os.path.join(BASE_DIR, 'outputs'), exist_ok=True)
_os.makedirs(_os.path.join(BASE_DIR, 'data'), exist_ok=True)
_os.makedirs(_os.path.join(BASE_DIR, 'app'), exist_ok=True)


plt.rcParams.update({
    'figure.facecolor': '#0A0A0A', 'axes.facecolor': '#111111',
    'axes.edgecolor': '#2a2a2a', 'axes.labelcolor': '#AAAAAA',
    'xtick.color': '#666666', 'ytick.color': '#666666',
    'text.color': '#F0EDE8', 'grid.color': '#1e1e1e',
    'font.family': 'monospace',
})
RED = '#FF4D4D'; TEAL = '#4ECDC4'

# ── 1. Load Raw Data ──────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load Raw Data")
print("=" * 60)

# Simulate dataset (replace with real load)
np.random.seed(42)
n = 50_000
weather_conditions = [
    'Clear', 'Overcast', 'Mostly Cloudy', 'Partly Cloudy',
    'Light Rain', 'Rain', 'Heavy Rain', 'Light Snow',
    'Snow', 'Fog', 'Haze', 'Thunderstorm', None
]
states = ['CA','TX','FL','NY','PA','OH','IL','GA','NC','VA']

df_raw = pd.DataFrame({
    'Severity':           np.random.choice([1,2,3,4], n, p=[0.03,0.70,0.22,0.05]),
    'Start_Time':         pd.date_range('2016-01-01', periods=n, freq='30min').astype(str),
    'End_Time':           pd.date_range('2016-01-01 01:00', periods=n, freq='30min').astype(str),
    'Temperature(F)':     np.where(np.random.rand(n) < 0.05, np.nan, np.random.normal(60, 20, n)),
    'Humidity(%)':        np.where(np.random.rand(n) < 0.04, np.nan, np.clip(np.random.normal(65, 20, n), 0, 100)),
    'Visibility(mi)':     np.where(np.random.rand(n) < 0.06, np.nan, np.clip(np.random.exponential(8, n), 0, 10)),
    'Wind_Speed(mph)':    np.where(np.random.rand(n) < 0.08, np.nan, np.clip(np.random.exponential(10, n), 0, 80)),
    'Precipitation(in)':  np.where(np.random.rand(n) < 0.35, np.nan, np.clip(np.random.exponential(0.05, n), 0, 5)),
    'Wind_Chill(F)':      np.where(np.random.rand(n) < 0.60, np.nan, np.random.normal(50, 20, n)),
    'Weather_Condition':  np.random.choice(weather_conditions, n),
    'State':              np.random.choice(states, n),
    'Distance(mi)':       np.clip(np.random.exponential(0.5, n), 0, 20),
    'Junction':           np.random.choice([True, False], n, p=[0.3, 0.7]),
    'Crossing':           np.random.choice([True, False], n, p=[0.2, 0.8]),
    'Traffic_Signal':     np.random.choice([True, False], n, p=[0.4, 0.6]),
    'Stop':               np.random.choice([True, False], n, p=[0.15, 0.85]),
    'No_Exit':            np.random.choice([True, False], n, p=[0.05, 0.95]),
    'Railway':            np.random.choice([True, False], n, p=[0.08, 0.92]),
    'Amenity':            np.random.choice([True, False], n, p=[0.10, 0.90]),
})

print(f"Raw shape: {df_raw.shape}")
df = df_raw.copy()

# ── 2. Drop High-Null Columns ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Drop High-Null Columns (>40% missing)")
print("=" * 60)
null_pct = df.isnull().mean() * 100
high_null = null_pct[null_pct > 40].index.tolist()
print(f"Dropping: {high_null}")
df.drop(columns=high_null, inplace=True)

# Drop rows with null target
before = len(df)
df.dropna(subset=['Severity'], inplace=True)
print(f"Dropped {before - len(df)} rows with null Severity")

# ── 3. Impute Missing Values ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Impute Missing Values")
print("=" * 60)
numeric_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Precipitation(in)']
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median()
        n_missing  = df[col].isnull().sum()
        df[col].fillna(median_val, inplace=True)
        if n_missing > 0:
            print(f"  {col:<25} → filled {n_missing:,} NaNs with median={median_val:.2f}")

df['Weather_Condition'].fillna('Unknown', inplace=True)
print(f"  {'Weather_Condition':<25} → filled NaNs with 'Unknown'")

# ── 4. Feature Engineering ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Feature Engineering")
print("=" * 60)

# Parse timestamps
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time']   = pd.to_datetime(df['End_Time'])

# Time features
df['Hour']            = df['Start_Time'].dt.hour
df['Month']           = df['Start_Time'].dt.month
df['DayOfWeek']       = df['Start_Time'].dt.dayofweek       # 0=Mon
df['Is_Weekend']      = (df['DayOfWeek'] >= 5).astype(int)
df['Is_Rush_Hour']    = df['Hour'].isin([7,8,9,16,17,18]).astype(int)
df['Is_Night']        = ((df['Hour'] >= 20) | (df['Hour'] <= 5)).astype(int)
df['Duration_Min']    = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

# Clip unrealistic durations
df['Duration_Min'] = df['Duration_Min'].clip(0, 1440)

# Weather severity score
weather_risk = {
    'Clear':0, 'Partly Cloudy':1, 'Mostly Cloudy':1, 'Overcast':2,
    'Haze':2, 'Light Rain':3, 'Rain':4, 'Heavy Rain':5,
    'Light Snow':4, 'Snow':5, 'Fog':5, 'Thunderstorm':6, 'Unknown':2
}
df['Weather_Risk'] = df['Weather_Condition'].map(weather_risk).fillna(2)

# Road complexity score
df['Road_Complexity'] = (
    df['Junction'].astype(int) +
    df['Crossing'].astype(int) +
    df['Traffic_Signal'].astype(int) +
    df['Stop'].astype(int) +
    df['Railway'].astype(int)
)

# Low visibility flag
df['Low_Visibility'] = (df['Visibility(mi)'] < 2).astype(int)

# Encode categorical
from sklearn.preprocessing import LabelEncoder
le_weather = LabelEncoder()
le_state   = LabelEncoder()
df['Weather_Encoded'] = le_weather.fit_transform(df['Weather_Condition'])
df['State_Encoded']   = le_state.fit_transform(df['State'])

new_features = ['Hour','Month','DayOfWeek','Is_Weekend','Is_Rush_Hour',
                'Is_Night','Duration_Min','Weather_Risk','Road_Complexity',
                'Low_Visibility','Weather_Encoded','State_Encoded']
print("New features created:")
for f in new_features:
    print(f"  ✓ {f}")

# ── 5. Final Feature Set ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Final Feature Set")
print("=" * 60)

FEATURES = [
    # Numeric weather
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)',
    # Engineered weather
    'Weather_Risk', 'Low_Visibility',
    # Road features
    'Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Railway',
    'Road_Complexity', 'Distance(mi)',
    # Time features
    'Hour', 'Month', 'Is_Weekend', 'Is_Rush_Hour', 'Is_Night',
    # Duration
    'Duration_Min',
    # Encoded
    'Weather_Encoded', 'State_Encoded'
]

TARGET = 'Severity'

X = df[FEATURES]
y = df[TARGET]

print(f"Features shape : {X.shape}")
print(f"Target classes : {sorted(y.unique())}")
print(f"\nClass distribution:")
for cls, cnt in y.value_counts().sort_index().items():
    pct = cnt/len(y)*100
    print(f"  Class {cls}: {cnt:>6,} ({pct:.1f}%)")

# ── 6. Scale Numeric Features ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Scale Numeric Features")
print("=" * 60)
from sklearn.preprocessing import StandardScaler

numeric_features = [
    'Temperature(F)','Humidity(%)','Visibility(mi)',
    'Wind_Speed(mph)','Precipitation(in)','Distance(mi)','Duration_Min'
]
scaler = StandardScaler()
X = X.copy()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
print(f"Scaled {len(numeric_features)} numeric features with StandardScaler")

# ── 7. Save Processed Data ────────────────────────────────────
import pickle, os
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
X.to_csv(os.path.join(BASE_DIR, 'data', 'X_processed.csv'), index=False)
y.to_csv(os.path.join(BASE_DIR, 'data', 'y_processed.csv'), index=False)
with open(os.path.join(BASE_DIR, 'data', 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(BASE_DIR, 'data', 'le_weather.pkl'), 'wb') as f:
    pickle.dump(le_weather, f)
with open(os.path.join(BASE_DIR, 'data', 'le_state.pkl'), 'wb') as f:
    pickle.dump(le_state, f)
with open(os.path.join(BASE_DIR, 'data', 'features.pkl'), 'wb') as f:
    pickle.dump(FEATURES, f)

print("\n✓ Saved: data/X_processed.csv")
print("✓ Saved: data/y_processed.csv")
print("✓ Saved: data/scaler.pkl")
print("\n→ Next: 03_Modeling.py")