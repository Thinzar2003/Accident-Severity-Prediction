# 🚗 Accident Severity Prediction

> A machine learning project that classifies road accident severity (1–4) using weather, road, and time features. Built as a data analysis portfolio project using the US Accidents dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

Road accidents are a major public safety concern. This project builds a **multi-class classifier** to predict the severity of a traffic accident (on a scale of 1–4) based on conditions at the time of the accident:

- 🌡️ **Weather** — temperature, humidity, visibility, precipitation
- 🛣️ **Road features** — junctions, crossings, traffic signals
- 🕐 **Time** — hour of day, day of week, month, rush hour
- 📍 **Location** — US state

**Goal:** Help traffic management agencies identify high-risk conditions before an accident happens, enabling proactive safety measures.

---

## 📊 Key Results

| Metric | Score |
|---|---|
| Best Model | XGBoost |
| Accuracy | 81% |
| Weighted F1-Score | 0.79 |
| ROC-AUC (Severe vs Non-Severe) | 0.84 |

---

## 🔍 Key Findings

After training on 700K+ accident records:

1. **Visibility is the #1 predictor** — Accidents with visibility below 2 miles are 2.4× more likely to be Severity 3–4
2. **Rush hour amplifies severity** — 7–9am and 4–6pm windows show 18% higher severe accident rates
3. **Weather compounding effect** — Fog + junction + night-time is the highest-risk combination in the dataset
4. **Severity 2 dominates** — 70% of accidents are Severity 2, requiring SMOTE to prevent model bias
5. **Junctions are dangerous** — Accidents near junctions are 22% more likely to be severe vs open road

### 💡 Recommendations
- Deploy dynamic speed limits during low-visibility or adverse weather conditions
- Increase traffic management personnel at high-risk junctions during rush hours
- Use predictive alerts for fog/heavy rain conditions on high-traffic corridors

---

## 📁 Project Structure

```
accident-severity-prediction/
│
├── notebooks/
│   ├── 01_EDA.py                      # Exploratory data analysis & visualizations
│   ├── 02_Data_Cleaning_Features.py   # Cleaning, imputation, feature engineering
│   └── 03_Modeling.py                 # Model training, evaluation, insights
│
├── app/
│   ├── streamlit_app.py               # Interactive prediction web app
│   ├── model.pkl                      # Saved XGBoost model
│   └── features.pkl                   # Feature list
│
├── data/
│   ├── X_processed.csv                # Processed features (generated)
│   ├── y_processed.csv                # Target variable (generated)
│   └── scaler.pkl                     # StandardScaler
│
├── outputs/
│   ├── 01_eda_overview.png            # EDA visualization dashboard
│   └── 03_evaluation_dashboard.png   # Model evaluation dashboard
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualization |
| **scikit-learn** | Preprocessing, modeling, evaluation |
| **XGBoost** | Best performing classifier |
| **imbalanced-learn** | SMOTE for class imbalance |
| **Streamlit** | Interactive web app |
| **Folium** | Geospatial heatmaps (optional) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/accident-severity-prediction.git
cd accident-severity-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- Download `US_Accidents_March23.csv`
- Place it in the `data/` folder

### 4. Run the notebooks in order
```bash
python notebooks/01_EDA.py
python notebooks/02_Data_Cleaning_Features.py
python notebooks/03_Modeling.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Pipeline

```
Raw Data (3.5M rows)
    ↓
Drop high-null columns (>40%)
    ↓
Impute missing values (median/mode)
    ↓
Feature Engineering
  · Extract hour, month, day from timestamp
  · Create is_rush_hour, is_night, is_weekend flags
  · Calculate weather_risk score (0–6)
  · Calculate road_complexity score (0–5)
    ↓
Encode categoricals (LabelEncoder)
Scale numerics (StandardScaler)
    ↓
Train/Test Split (80/20, stratified)
    ↓
SMOTE (fix class imbalance on train set)
    ↓
Train: Logistic Regression, Decision Tree,
       Random Forest, XGBoost
    ↓
Evaluate: Accuracy, F1, Confusion Matrix, ROC-AUC
    ↓
Best model saved → Streamlit App
```

---

## 📊 Visualizations

The project produces two dashboards:

**EDA Dashboard** (`outputs/01_eda_overview.png`)
- Severity distribution
- Accidents by hour of day
- Severity heatmap by hour
- Monthly accident trends
- Weather condition breakdown
- Feature correlation matrix
- Severe accident rate by road feature

**Evaluation Dashboard** (`outputs/03_evaluation_dashboard.png`)
- Model comparison (Accuracy + F1)
- Confusion matrix
- Per-class F1 scores
- Feature importance chart
- ROC curve (Severe vs Non-Severe)

---

## 🧠 Feature Engineering Details

| Feature | Description |
|---|---|
| `Is_Rush_Hour` | 1 if accident occurred 7–9am or 4–6pm |
| `Is_Night` | 1 if accident occurred between 8pm–5am |
| `Weather_Risk` | Score 0–6 based on weather condition severity |
| `Road_Complexity` | Count of road features present (0–5) |
| `Low_Visibility` | 1 if visibility < 2 miles |
| `Duration_Min` | Accident duration in minutes |

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
folium>=0.14.0
```

---

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 Dataset Citation

Moosavi, Sobhan, et al. *"A Countrywide Traffic Accident Dataset."* 2019.
Available at: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

## 📝 License

This project is open source under the [MIT License](LICENSE).
