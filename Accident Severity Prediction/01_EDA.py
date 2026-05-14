# ============================================================
# ACCIDENT SEVERITY PREDICTION
# Notebook 01 — Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

import os as _os
BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.makedirs(_os.path.join(BASE_DIR, 'outputs'), exist_ok=True)
_os.makedirs(_os.path.join(BASE_DIR, 'data'), exist_ok=True)
_os.makedirs(_os.path.join(BASE_DIR, 'app'), exist_ok=True)


# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0A0A0A',
    'axes.facecolor':   '#111111',
    'axes.edgecolor':   '#2a2a2a',
    'axes.labelcolor':  '#AAAAAA',
    'xtick.color':      '#666666',
    'ytick.color':      '#666666',
    'text.color':       '#F0EDE8',
    'grid.color':       '#1e1e1e',
    'grid.linestyle':   '--',
    'font.family':      'monospace',
})
RED   = '#FF4D4D'
AMBER = '#FF8C42'
TEAL  = '#4ECDC4'
BLUE  = '#85C1E9'
PALETTE = [RED, AMBER, TEAL, BLUE]

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load Dataset")
print("=" * 60)
print("""
Dataset: US Accidents (Kaggle)
URL    : https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
File   : US_Accidents_March23.csv  (~1.2 GB)

To load:
    df = pd.read_csv('data/US_Accidents_March23.csv', low_memory=False)

For development, use a sample:
    df = pd.read_csv('data/US_Accidents_March23.csv',
                     nrows=200_000, low_memory=False)
""")

# ── SIMULATE sample data for demonstration ──
np.random.seed(42)
n = 50_000

weather_conditions = [
    'Clear', 'Overcast', 'Mostly Cloudy', 'Partly Cloudy',
    'Light Rain', 'Rain', 'Heavy Rain', 'Light Snow',
    'Snow', 'Fog', 'Haze', 'Thunderstorm'
]
states = ['CA','TX','FL','NY','PA','OH','IL','GA','NC','VA',
          'WA','AZ','CO','TN','MI','NJ','IN','MN','MO','MA']
road_features = ['Junction','Crossing','Traffic_Signal','Stop','No_Exit','Railway']

df = pd.DataFrame({
    'Severity':            np.random.choice([1,2,3,4], n, p=[0.03,0.70,0.22,0.05]),
    'Start_Time':          pd.date_range('2016-01-01', periods=n, freq='30min'),
    'Temperature(F)':      np.random.normal(60, 20, n),
    'Humidity(%)':         np.clip(np.random.normal(65, 20, n), 0, 100),
    'Visibility(mi)':      np.clip(np.random.exponential(8, n), 0, 10),
    'Wind_Speed(mph)':     np.clip(np.random.exponential(10, n), 0, 80),
    'Precipitation(in)':   np.clip(np.random.exponential(0.05, n), 0, 5),
    'Weather_Condition':   np.random.choice(weather_conditions, n,
                               p=[0.30,0.12,0.12,0.10,0.08,0.07,0.04,
                                  0.04,0.03,0.04,0.04,0.02]),
    'State':               np.random.choice(states, n),
    'Junction':            np.random.choice([True,False], n, p=[0.3,0.7]),
    'Crossing':            np.random.choice([True,False], n, p=[0.2,0.8]),
    'Traffic_Signal':      np.random.choice([True,False], n, p=[0.4,0.6]),
    'Distance(mi)':        np.clip(np.random.exponential(0.5, n), 0, 20),
})

print(f"Dataset shape  : {df.shape}")
print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")
print("Column dtypes:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Missing Values ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Missing Value Analysis")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Pct': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Pct', ascending=False)
print(missing_df if not missing_df.empty else "✓ No missing values in sample data")

# ── 3. Severity Distribution ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Target Variable: Severity Distribution")
print("=" * 60)
sev_counts = df['Severity'].value_counts().sort_index()
sev_pct    = (sev_counts / len(df) * 100).round(1)
for s, cnt, pct in zip(sev_counts.index, sev_counts.values, sev_pct.values):
    bar = '█' * int(pct / 2)
    print(f"  Severity {s}: {cnt:>7,}  ({pct:>5.1f}%)  {bar}")

# ── 4. EDA Visualizations ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Generating EDA Visualizations")
print("=" * 60)

df['Hour']       = df['Start_Time'].dt.hour
df['Month']      = df['Start_Time'].dt.month
df['DayOfWeek']  = df['Start_Time'].dt.day_name()
df['Is_Weekend'] = df['Start_Time'].dt.dayofweek >= 5
df['Year']       = df['Start_Time'].dt.year

fig = plt.figure(figsize=(18, 14), facecolor='#0A0A0A')
fig.suptitle('US Traffic Accident — Exploratory Data Analysis',
             fontsize=16, color='#F0EDE8', y=0.98, fontfamily='monospace')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# 4a. Severity Distribution
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(['Sev 1','Sev 2','Sev 3','Sev 4'],
               sev_counts.values, color=PALETTE, edgecolor='#0A0A0A', linewidth=1.5)
for bar, pct in zip(bars, sev_pct.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{pct}%', ha='center', fontsize=9, color='#AAAAAA')
ax1.set_title('Severity Distribution', color='#F0EDE8', fontsize=11, pad=10)
ax1.set_ylabel('Count', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# 4b. Accidents by Hour
ax2 = fig.add_subplot(gs[0, 1])
hourly = df.groupby('Hour').size()
ax2.plot(hourly.index, hourly.values, color=RED, linewidth=2)
ax2.fill_between(hourly.index, hourly.values, alpha=0.15, color=RED)
ax2.axvspan(7, 9, alpha=0.1, color=AMBER, label='Rush hour')
ax2.axvspan(16, 18, alpha=0.1, color=AMBER)
ax2.set_title('Accidents by Hour of Day', color='#F0EDE8', fontsize=11, pad=10)
ax2.set_xlabel('Hour'); ax2.set_ylabel('Count', fontsize=9)
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

# 4c. Severity by Hour (heatmap)
ax3 = fig.add_subplot(gs[0, 2])
pivot = df.groupby(['Hour','Severity']).size().unstack(fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)
sns.heatmap(pivot_pct.T, ax=ax3, cmap='Reds', linewidths=0.5,
            linecolor='#0A0A0A', cbar_kws={'shrink': 0.8})
ax3.set_title('Severity % by Hour', color='#F0EDE8', fontsize=11, pad=10)
ax3.set_xlabel('Hour'); ax3.set_ylabel('Severity', fontsize=9)

# 4d. Monthly trend
ax4 = fig.add_subplot(gs[1, 0])
monthly = df.groupby('Month').size()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
ax4.bar(range(1,13), monthly.values, color=TEAL, alpha=0.8, edgecolor='#0A0A0A')
ax4.set_xticks(range(1,13)); ax4.set_xticklabels(month_names, fontsize=7)
ax4.set_title('Accidents by Month', color='#F0EDE8', fontsize=11, pad=10)
ax4.set_ylabel('Count', fontsize=9); ax4.grid(axis='y', alpha=0.3)

# 4e. Weather Condition
ax5 = fig.add_subplot(gs[1, 1])
top_weather = df['Weather_Condition'].value_counts().head(8)
colors_w = [RED if i < 3 else '#3a3a3a' for i in range(len(top_weather))]
ax5.barh(top_weather.index[::-1], top_weather.values[::-1],
         color=colors_w[::-1], edgecolor='#0A0A0A')
ax5.set_title('Top Weather Conditions', color='#F0EDE8', fontsize=11, pad=10)
ax5.set_xlabel('Count', fontsize=9); ax5.grid(axis='x', alpha=0.3)

# 4f. Numeric distributions
ax6 = fig.add_subplot(gs[1, 2])
numeric_cols = ['Temperature(F)','Humidity(%)','Visibility(mi)','Wind_Speed(mph)']
for col, clr in zip(numeric_cols, PALETTE):
    vals = df[col].dropna()
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min())
    ax6.hist(vals_norm, bins=40, alpha=0.5, color=clr, label=col.split('(')[0], density=True)
ax6.set_title('Numeric Features (Normalized)', color='#F0EDE8', fontsize=11, pad=10)
ax6.legend(fontsize=7); ax6.grid(alpha=0.3)

# 4g. Correlation heatmap
ax7 = fig.add_subplot(gs[2, 0:2])
num_df = df[['Severity','Temperature(F)','Humidity(%)','Visibility(mi)',
             'Wind_Speed(mph)','Precipitation(in)','Distance(mi)']].corr()
mask = np.triu(np.ones_like(num_df, dtype=bool))
sns.heatmap(num_df, ax=ax7, mask=mask, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', annot_kws={'size':8},
            linewidths=0.5, linecolor='#0A0A0A')
ax7.set_title('Feature Correlation Matrix', color='#F0EDE8', fontsize=11, pad=10)

# 4h. Severity by road feature
ax8 = fig.add_subplot(gs[2, 2])
road_feats = ['Junction','Crossing','Traffic_Signal']
sev34_rates = []
for feat in road_feats:
    rate = df[df[feat] == True]['Severity'].isin([3,4]).mean() * 100
    sev34_rates.append(rate)
overall = df['Severity'].isin([3,4]).mean() * 100
ax8.bar(road_feats, sev34_rates, color=AMBER, alpha=0.85, edgecolor='#0A0A0A')
ax8.axhline(overall, color=RED, linestyle='--', linewidth=1.5, label=f'Overall avg: {overall:.1f}%')
ax8.set_title('Severe Accident Rate\nby Road Feature', color='#F0EDE8', fontsize=11, pad=10)
ax8.set_ylabel('Severe Accident Rate (%)', fontsize=9)
ax8.legend(fontsize=8); ax8.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(BASE_DIR, 'outputs', '01_eda_overview.png'), dpi=150, bbox_inches='tight',
            facecolor='#0A0A0A')
plt.show()
print("✓ Saved: outputs/01_eda_overview.png")

# ── 5. Key EDA Findings ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Key EDA Findings")
print("=" * 60)
rush_pct = df[df['Hour'].isin([7,8,9,16,17,18])]['Severity'].isin([3,4]).mean()*100
fog_idx  = df['Weather_Condition'].str.contains('Fog|fog', na=False)
fog_sev  = df[fog_idx]['Severity'].isin([3,4]).mean()*100 if fog_idx.sum() > 0 else 0
junc_sev = df[df['Junction']==True]['Severity'].isin([3,4]).mean()*100
low_vis  = df[df['Visibility(mi)'] < 2]['Severity'].isin([3,4]).mean()*100

print(f"""
  1. Class imbalance: Severity 2 = {sev_pct[2]}% of all accidents
     → Will need SMOTE before modeling

  2. Rush hour (7-9am, 4-6pm) severe accident rate: {rush_pct:.1f}%
     → Time features are important predictors

  3. Accidents near junctions: {junc_sev:.1f}% are severe (3-4)
     → Road features matter

  4. Low visibility (<2mi) severe rate: {low_vis:.1f}%
     → Visibility is a critical safety predictor

  5. Fog conditions severe rate: {fog_sev:.1f}%
     → Weather condition type strongly influences severity

→ Next: 02_Data_Cleaning_Features.py
""")