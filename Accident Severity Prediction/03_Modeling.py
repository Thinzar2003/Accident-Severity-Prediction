# ============================================================
# ACCIDENT SEVERITY PREDICTION
# Notebook 03 — Modeling, Evaluation & Insights
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle, os, warnings
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
RED='#FF4D4D'; AMBER='#FF8C42'; TEAL='#4ECDC4'; BLUE='#85C1E9'
PALETTE=[RED,AMBER,TEAL,BLUE]

# ── 1. Load Processed Data ────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load Processed Data")
print("=" * 60)

# Regenerate data (in real project: load from CSV)
np.random.seed(42)
n = 50_000

FEATURES = [
    'Temperature(F)','Humidity(%)','Visibility(mi)','Wind_Speed(mph)',
    'Precipitation(in)','Weather_Risk','Low_Visibility','Junction',
    'Crossing','Traffic_Signal','Stop','Railway','Road_Complexity',
    'Distance(mi)','Hour','Month','Is_Weekend','Is_Rush_Hour',
    'Is_Night','Duration_Min','Weather_Encoded','State_Encoded'
]

X = pd.DataFrame(np.random.randn(n, len(FEATURES)), columns=FEATURES)
X['Junction']       = np.random.randint(0, 2, n)
X['Crossing']       = np.random.randint(0, 2, n)
X['Traffic_Signal'] = np.random.randint(0, 2, n)
X['Stop']           = np.random.randint(0, 2, n)
X['Railway']        = np.random.randint(0, 2, n)
X['Is_Weekend']     = np.random.randint(0, 2, n)
X['Is_Rush_Hour']   = np.random.randint(0, 2, n)
X['Is_Night']       = np.random.randint(0, 2, n)
X['Low_Visibility'] = np.random.randint(0, 2, n)
X['Weather_Risk']   = np.random.randint(0, 7, n)
X['Road_Complexity']= np.random.randint(0, 5, n)
X['Weather_Encoded']= np.random.randint(0, 12, n)
X['State_Encoded']  = np.random.randint(0, 10, n)
X['Hour']           = np.random.randint(0, 24, n)
X['Month']          = np.random.randint(1, 13, n)
y = pd.Series(np.random.choice([1,2,3,4], n, p=[0.03,0.70,0.22,0.05]))

print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# ── 2. Train/Test Split ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Train/Test Split")
print("=" * 60)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 3. Handle Class Imbalance with SMOTE ──────────────────────
print("\n" + "=" * 60)
print("STEP 3 — SMOTE Oversampling")
print("=" * 60)
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("✓ SMOTE applied")
except ImportError:
    print("⚠  imbalanced-learn not installed — using raw data")
    print("   Install: pip install imbalanced-learn")
    X_train_res, y_train_res = X_train, y_train

print("Resampled class distribution:")
for cls, cnt in pd.Series(y_train_res).value_counts().sort_index().items():
    print(f"  Class {cls}: {cnt:,}")

# ── 4. Train Multiple Models ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Train & Compare Models")
print("=" * 60)
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree           import DecisionTreeClassifier
from sklearn.metrics        import f1_score, accuracy_score
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("⚠  XGBoost not installed — skipping. pip install xgboost")

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, max_depth=10,
                                                    random_state=42, n_jobs=-1),
}
if USE_XGB:
    models['XGBoost'] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss', verbosity=0,
        # XGBoost requires 0-indexed labels
        # We'll handle remapping inside the loop
    )

results = {}
best_model = None
best_f1    = 0

for name, model in models.items():
    print(f"\n  Training: {name}...", end=' ', flush=True)
    # XGBoost needs 0-indexed labels
    if name == 'XGBoost':
        model.fit(X_train_res, y_train_res - 1)
        y_pred = model.predict(X_test) + 1
    else:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': acc, 'f1_weighted': f1, 'model': model, 'pred': y_pred, 'is_xgb': name=='XGBoost'}
    print(f"Acc={acc:.3f}  F1={f1:.3f}")
    if f1 > best_f1:
        best_f1   = f1
        best_model = name

print(f"\n★ Best model: {best_model}  (Weighted F1 = {best_f1:.3f})")
best = results[best_model]

# ── 5. Detailed Evaluation ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Detailed Evaluation")
print("=" * 60)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)

print(f"\nClassification Report — {best_model}:")
print(classification_report(y_test, best['pred'],
      target_names=['Severity 1','Severity 2','Severity 3','Severity 4']))

# ── 6. Visualization Dashboard ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Generating Evaluation Dashboard")
print("=" * 60)
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

fig = plt.figure(figsize=(18, 12), facecolor='#0A0A0A')
fig.suptitle(f'Model Evaluation Dashboard — {best_model}',
             fontsize=15, color='#F0EDE8', y=0.98, fontfamily='monospace')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 6a. Model Comparison
ax1 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
f1s   = [results[n]['f1_weighted'] for n in names]
accs  = [results[n]['accuracy'] for n in names]
x = np.arange(len(names))
w = 0.35
bars1 = ax1.bar(x - w/2, accs, w, label='Accuracy', color=TEAL, alpha=0.85, edgecolor='#0A0A0A')
bars2 = ax1.bar(x + w/2, f1s,  w, label='F1 (weighted)', color=RED, alpha=0.85, edgecolor='#0A0A0A')
ax1.set_xticks(x); ax1.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=7)
ax1.set_ylim(0, 1); ax1.set_title('Model Comparison', color='#F0EDE8', fontsize=11, pad=10)
ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{bar.get_height():.2f}', ha='center', fontsize=7, color='#AAAAAA')

# 6b. Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, best['pred'])
sns.heatmap(cm, ax=ax2, cmap='Reds', annot=True, fmt='d',
            linewidths=0.5, linecolor='#0A0A0A',
            xticklabels=['Sev 1','Sev 2','Sev 3','Sev 4'],
            yticklabels=['Sev 1','Sev 2','Sev 3','Sev 4'],
            cbar_kws={'shrink': 0.8})
ax2.set_title('Confusion Matrix', color='#F0EDE8', fontsize=11, pad=10)
ax2.set_xlabel('Predicted', fontsize=9); ax2.set_ylabel('Actual', fontsize=9)

# 6c. Per-Class F1
ax3 = fig.add_subplot(gs[0, 2])
from sklearn.metrics import f1_score
per_class_f1 = f1_score(y_test, best['pred'], average=None, labels=[1,2,3,4])
bars = ax3.bar(['Sev 1','Sev 2','Sev 3','Sev 4'], per_class_f1,
               color=PALETTE, edgecolor='#0A0A0A', alpha=0.9)
ax3.axhline(best_f1, color='#555', linestyle='--', linewidth=1, label=f'Weighted avg: {best_f1:.2f}')
ax3.set_title('F1-Score per Severity Class', color='#F0EDE8', fontsize=11, pad=10)
ax3.set_ylim(0, 1); ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, per_class_f1):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{val:.2f}', ha='center', fontsize=9, color='#F0EDE8')

# 6d. Feature Importance
ax4 = fig.add_subplot(gs[1, 0:2])
model_obj = results[best_model]['model']
if hasattr(model_obj, 'feature_importances_'):
    imp = pd.Series(model_obj.feature_importances_, index=FEATURES)
    top_imp = imp.sort_values(ascending=True).tail(12)
    colors_imp = [RED if v > top_imp.quantile(0.75) else AMBER
                  if v > top_imp.median() else '#3a3a3a' for v in top_imp.values]
    ax4.barh(top_imp.index, top_imp.values, color=colors_imp, edgecolor='#0A0A0A')
    ax4.set_title('Feature Importance — Top 12', color='#F0EDE8', fontsize=11, pad=10)
    ax4.set_xlabel('Importance Score', fontsize=9)
    ax4.grid(axis='x', alpha=0.3)
    ax4.axvline(top_imp.median(), color='#555', linestyle='--', linewidth=1,
                label='Median importance')
    ax4.legend(fontsize=8)
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available for\nLogistic Regression',
             ha='center', va='center', transform=ax4.transAxes, color='#666', fontsize=11)
    ax4.set_title('Feature Importance', color='#F0EDE8', fontsize=11, pad=10)

# 6e. ROC Curve (Severe vs Not Severe binary)
ax5 = fig.add_subplot(gs[1, 2])
y_binary = (y_test >= 3).astype(int)
if hasattr(model_obj, 'predict_proba'):
    y_prob = model_obj.predict_proba(X_test)
    # probability of class 3 or 4
    classes = list(model_obj.classes_)
    idx3 = classes.index(3) if 3 in classes else None
    idx4 = classes.index(4) if 4 in classes else None
    if idx3 is not None and idx4 is not None:
        y_score = y_prob[:, idx3] + y_prob[:, idx4]
    elif idx3 is not None:
        y_score = y_prob[:, idx3]
    else:
        y_score = y_prob[:, -1]
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, color=RED, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax5.fill_between(fpr, tpr, alpha=0.1, color=RED)
    ax5.plot([0,1],[0,1], '--', color='#444', linewidth=1)
    ax5.set_title('ROC Curve\n(Severe vs Non-Severe)', color='#F0EDE8', fontsize=11, pad=10)
    ax5.set_xlabel('False Positive Rate', fontsize=9)
    ax5.set_ylabel('True Positive Rate', fontsize=9)
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'ROC not available', ha='center', va='center',
             transform=ax5.transAxes, color='#666')
    ax5.set_title('ROC Curve', color='#F0EDE8', fontsize=11, pad=10)

plt.savefig(os.path.join(BASE_DIR, 'outputs', '03_evaluation_dashboard.png'), dpi=150,
            bbox_inches='tight', facecolor='#0A0A0A')
plt.show()
print("✓ Saved: outputs/03_evaluation_dashboard.png")

# ── 7. Save Best Model ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Save Best Model")
print("=" * 60)
with open(os.path.join(BASE_DIR, 'app', 'model.pkl'), 'wb') as f:
    pickle.dump(results[best_model]['model'], f)
with open(os.path.join(BASE_DIR, 'app', 'features.pkl'), 'wb') as f:
    pickle.dump(FEATURES, f)
print(f"✓ Saved: app/model.pkl  ({best_model})")
print("✓ Saved: app/features.pkl")

# ── 8. Business Insights ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Key Business Insights")
print("=" * 60)

if hasattr(model_obj, 'feature_importances_'):
    imp = pd.Series(model_obj.feature_importances_, index=FEATURES)
    top3 = imp.sort_values(ascending=False).head(3)
    print("\nTop 3 predictors of accident severity:")
    for i, (feat, score) in enumerate(top3.items(), 1):
        print(f"  {i}. {feat:<25} importance = {score:.4f}")

overall_severe_rate = (y_test >= 3).mean() * 100
rush_mask   = X_test['Is_Rush_Hour'] == 1
night_mask  = X_test['Is_Night'] == 1
lowvis_mask = X_test['Low_Visibility'] == 1
junc_mask   = X_test['Junction'] == 1

print(f"""
Accident Severity Analysis — Key Findings:
─────────────────────────────────────────
  Overall severe (3-4) rate   : {overall_severe_rate:.1f}%
  Rush hour severe rate       : {(y_test[rush_mask] >= 3).mean()*100:.1f}%
  Night-time severe rate      : {(y_test[night_mask] >= 3).mean()*100:.1f}%
  Low visibility severe rate  : {(y_test[lowvis_mask] >= 3).mean()*100:.1f}%
  Junction severe rate        : {(y_test[junc_mask] >= 3).mean()*100:.1f}%

Model Performance Summary:
─────────────────────────────────────────
  Best Model    : {best_model}
  Accuracy      : {results[best_model]['accuracy']:.3f}
  Weighted F1   : {results[best_model]['f1_weighted']:.3f}

→ Next: Run app/streamlit_app.py for interactive demo
""")