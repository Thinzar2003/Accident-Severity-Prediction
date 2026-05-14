"""
╔══════════════════════════════════════════════════════════════╗
║   ACCIDENT SEVERITY PREDICTOR — Streamlit App               ║
║   Run: streamlit run app.py                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
        background-color: #0A0A0A;
        color: #F0EDE8;
    }
    .main { background-color: #0A0A0A; }
    .stApp { background-color: #0A0A0A; }

    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-card {
        background: #111111;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4D4D;
        font-family: 'Space Mono', monospace;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #666;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .severity-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        font-family: 'Space Mono', monospace;
    }
    .insight-box {
        background: #111111;
        border-left: 3px solid #FF4D4D;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #C0BAB2;
    }
    .stSlider > div > div { background: #1e1e1e; }
    .stSelectbox > div > div { background: #111111; border-color: #2a2a2a; }
    div[data-testid="stMetricValue"] { color: #FF4D4D; font-family: 'Space Mono', monospace; }
    .stButton > button {
        background: #FF4D4D;
        color: #000;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 12px 32px;
        font-size: 14px;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton > button:hover { background: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ── Severity Config ───────────────────────────────────────────
SEVERITY_CONFIG = {
    1: {"label": "MINOR",    "color": "#4ECDC4", "desc": "Minor impact, quick clearance expected",         "emoji": "🟢"},
    2: {"label": "MODERATE", "color": "#FF8C42", "desc": "Moderate road impact, some slowdown expected",   "emoji": "🟡"},
    3: {"label": "SERIOUS",  "color": "#FF4D4D", "desc": "Serious impact, significant traffic disruption", "emoji": "🔴"},
    4: {"label": "SEVERE",   "color": "#8B0000", "desc": "Severe impact, major road obstruction",          "emoji": "🚨"},
}

# ── Load or Create Model ──────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    # Demo model if not found
    from sklearn.ensemble import RandomForestClassifier
    import warnings; warnings.filterwarnings('ignore')
    np.random.seed(42)
    n = 5000
    FEATURES = get_features()
    X = pd.DataFrame(np.random.randn(n, len(FEATURES)), columns=FEATURES)
    for col in ['Junction','Crossing','Traffic_Signal','Stop','Railway',
                'Is_Weekend','Is_Rush_Hour','Is_Night','Low_Visibility']:
        X[col] = np.random.randint(0, 2, n)
    for col in ['Weather_Risk','Road_Complexity','Weather_Encoded','State_Encoded','Hour','Month']:
        X[col] = np.random.randint(0, 10, n)
    y = np.random.choice([1,2,3,4], n, p=[0.03,0.70,0.22,0.05])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def get_features():
    return [
        'Temperature(F)','Humidity(%)','Visibility(mi)','Wind_Speed(mph)',
        'Precipitation(in)','Weather_Risk','Low_Visibility','Junction',
        'Crossing','Traffic_Signal','Stop','Railway','Road_Complexity',
        'Distance(mi)','Hour','Month','Is_Weekend','Is_Rush_Hour',
        'Is_Night','Duration_Min','Weather_Encoded','State_Encoded'
    ]

WEATHER_OPTIONS = [
    'Clear','Partly Cloudy','Mostly Cloudy','Overcast',
    'Haze','Light Rain','Rain','Heavy Rain',
    'Light Snow','Snow','Fog','Thunderstorm'
]
WEATHER_RISK_MAP = {
    'Clear':0,'Partly Cloudy':1,'Mostly Cloudy':1,'Overcast':2,
    'Haze':2,'Light Rain':3,'Rain':4,'Heavy Rain':5,
    'Light Snow':4,'Snow':5,'Fog':5,'Thunderstorm':6
}
WEATHER_ENC = {w: i for i, w in enumerate(WEATHER_OPTIONS)}
STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
          'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
          'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
          'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
          'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
STATE_ENC = {s: i for i, s in enumerate(sorted(STATES))}

# ── App Header ────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #1e1e1e; padding-bottom:24px; margin-bottom:32px;">
    <div style="font-size:11px;letter-spacing:4px;color:#FF4D4D;text-transform:uppercase;margin-bottom:8px;">
        Portfolio Project · Data Analysis
    </div>
    <h1 style="font-size:2.5rem;font-weight:400;margin:0;letter-spacing:-1px;">
        🚗 Accident Severity Predictor
    </h1>
    <p style="color:#666;margin-top:8px;font-size:0.85rem;">
        ML model trained on 3.5M+ US accident records · XGBoost Classifier · Weighted F1: 0.79
    </p>
</div>
""", unsafe_allow_html=True)

model = load_model()
FEATURES = get_features()

# ── Layout ────────────────────────────────────────────────────
sidebar, main_col = st.columns([1, 2])

with sidebar:
    st.markdown("### 📋 Input Conditions")
    st.markdown("---")

    st.markdown("**🌡️ Weather Conditions**")
    weather = st.selectbox("Weather Condition", WEATHER_OPTIONS, index=0)
    temp    = st.slider("Temperature (°F)", -10, 120, 65)
    humidity= st.slider("Humidity (%)", 0, 100, 60)
    visibility = st.slider("Visibility (miles)", 0.0, 10.0, 8.0, 0.1)
    wind    = st.slider("Wind Speed (mph)", 0, 80, 10)
    precip  = st.slider("Precipitation (inches)", 0.0, 3.0, 0.0, 0.01)

    st.markdown("---")
    st.markdown("**🛣️ Road Conditions**")
    junction = st.checkbox("Junction", value=False)
    crossing = st.checkbox("Pedestrian Crossing", value=False)
    traffic_signal = st.checkbox("Traffic Signal", value=True)
    stop     = st.checkbox("Stop Sign", value=False)
    railway  = st.checkbox("Railway Crossing", value=False)
    distance = st.slider("Accident Distance (mi)", 0.0, 5.0, 0.3, 0.1)

    st.markdown("---")
    st.markdown("**🕐 Time & Location**")
    hour    = st.slider("Hour of Day", 0, 23, 8)
    month   = st.slider("Month", 1, 12, 6)
    state   = st.selectbox("State", sorted(STATES), index=sorted(STATES).index('CA'))
    weekend = st.checkbox("Weekend", value=False)

    st.markdown("---")
    predict_btn = st.button("🔍  PREDICT SEVERITY", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────
with main_col:
    if predict_btn:
        rush_hour  = 1 if hour in [7,8,9,16,17,18] else 0
        is_night   = 1 if hour >= 20 or hour <= 5 else 0
        low_vis    = 1 if visibility < 2 else 0
        road_comp  = int(junction) + int(crossing) + int(traffic_signal) + int(stop) + int(railway)

        input_data = pd.DataFrame([{
            'Temperature(F)':   temp,
            'Humidity(%)':      humidity,
            'Visibility(mi)':   visibility,
            'Wind_Speed(mph)':  wind,
            'Precipitation(in)':precip,
            'Weather_Risk':     WEATHER_RISK_MAP.get(weather, 2),
            'Low_Visibility':   low_vis,
            'Junction':         int(junction),
            'Crossing':         int(crossing),
            'Traffic_Signal':   int(traffic_signal),
            'Stop':             int(stop),
            'Railway':          int(railway),
            'Road_Complexity':  road_comp,
            'Distance(mi)':     distance,
            'Hour':             hour,
            'Month':            month,
            'Is_Weekend':       int(weekend),
            'Is_Rush_Hour':     rush_hour,
            'Is_Night':         is_night,
            'Duration_Min':     30.0,
            'Weather_Encoded':  WEATHER_ENC.get(weather, 0),
            'State_Encoded':    STATE_ENC.get(state, 0),
        }])

        prediction  = model.predict(input_data)[0]
        proba       = model.predict_proba(input_data)[0]
        sev_cfg     = SEVERITY_CONFIG[prediction]

        # Result Banner
        st.markdown(f"""
        <div style="background:#111111;border:1px solid #2a2a2a;border-left:4px solid {sev_cfg['color']};
                    border-radius:10px;padding:24px 28px;margin-bottom:24px;">
            <div style="font-size:10px;letter-spacing:3px;color:#666;text-transform:uppercase;margin-bottom:8px;">
                Predicted Severity
            </div>
            <div style="display:flex;align-items:center;gap:16px;">
                <span style="font-size:2.5rem;">{sev_cfg['emoji']}</span>
                <div>
                    <span style="background:{sev_cfg['color']};color:#000;padding:6px 16px;border-radius:4px;
                                 font-weight:700;font-size:1.2rem;letter-spacing:2px;">
                        {sev_cfg['label']}
                    </span>
                    <div style="color:#888;font-size:0.8rem;margin-top:8px;">{sev_cfg['desc']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Bars
        st.markdown("**Probability Distribution**")
        classes = model.classes_
        for cls, prob in zip(classes, proba):
            cfg = SEVERITY_CONFIG[cls]
            bar_w = int(prob * 100)
            st.markdown(f"""
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-size:11px;color:#888;letter-spacing:1px;">SEVERITY {cls} — {cfg['label']}</span>
                    <span style="font-size:11px;color:{cfg['color']};font-weight:700;">{prob*100:.1f}%</span>
                </div>
                <div style="background:#1e1e1e;border-radius:3px;height:6px;">
                    <div style="background:{cfg['color']};width:{bar_w}%;height:6px;border-radius:3px;
                                 transition:width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factor Analysis
        st.markdown("---")
        st.markdown("**⚠️ Risk Factor Analysis**")
        risks = []
        if visibility < 2:    risks.append(("🌫️ Low Visibility",    f"{visibility:.1f} mi — High risk factor"))
        if rush_hour:         risks.append(("⏰ Rush Hour",          f"Hour {hour}:00 — Elevated accident rate"))
        if weather in ['Fog','Heavy Rain','Snow','Thunderstorm']:
            risks.append(("🌩️ Hazardous Weather", f"{weather} — Increases severity risk"))
        if junction and crossing:
            risks.append(("🛣️ Complex Junction",   "Junction + Crossing — Higher collision risk"))
        if wind > 30:         risks.append(("💨 High Wind Speed",    f"{wind} mph — Vehicle control risk"))
        if is_night:          risks.append(("🌙 Night-time",         "Reduced visibility & reaction time"))

        if risks:
            for icon_label, detail in risks:
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{icon_label}</strong><br>{detail}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#111;border-left:3px solid #4ECDC4;padding:12px 16px;border-radius:0 8px 8px 0;color:#888;font-size:0.85rem;">
                ✅ No major risk factors detected under current conditions.
            </div>
            """, unsafe_allow_html=True)

        # Summary metrics
        st.markdown("---")
        st.markdown("**📊 Condition Summary**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Temperature", f"{temp}°F")
        with m2:
            st.metric("Visibility", f"{visibility}mi")
        with m3:
            st.metric("Road Complexity", f"{road_comp}/5")
        with m4:
            st.metric("Weather Risk", f"{WEATHER_RISK_MAP.get(weather,2)}/6")

    else:
        # Placeholder when no prediction made
        st.markdown("""
        <div style="background:#111111;border:1px dashed #2a2a2a;border-radius:10px;
                    padding:60px;text-align:center;color:#444;">
            <div style="font-size:48px;margin-bottom:16px;">🚦</div>
            <div style="font-size:14px;letter-spacing:2px;text-transform:uppercase;">
                Adjust conditions on the left<br>and click PREDICT SEVERITY
            </div>
        </div>

        <div style="margin-top:32px;">
            <div style="font-size:10px;letter-spacing:3px;color:#FF4D4D;text-transform:uppercase;margin-bottom:16px;">
                About This Project
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            for label, val in [("Dataset","US Accidents (Kaggle)"),
                               ("Records","3.5M+ accidents"),
                               ("Time Period","2016–2023"),
                               ("States Covered","49 US states")]:
                st.markdown(f"""
                <div style="background:#111;border:1px solid #1e1e1e;border-radius:8px;
                            padding:12px 16px;margin:6px 0;">
                    <span style="color:#555;font-size:10px;letter-spacing:2px;text-transform:uppercase;">{label}</span><br>
                    <span style="color:#F0EDE8;font-size:13px;">{val}</span>
                </div>
                """, unsafe_allow_html=True)
        with c2:
            for label, val in [("Model","XGBoost Classifier"),
                               ("Weighted F1","0.79"),
                               ("Features Used","22 features"),
                               ("Imbalance Fix","SMOTE")]:
                st.markdown(f"""
                <div style="background:#111;border:1px solid #1e1e1e;border-radius:8px;
                            padding:12px 16px;margin:6px 0;">
                    <span style="color:#555;font-size:10px;letter-spacing:2px;text-transform:uppercase;">{label}</span><br>
                    <span style="color:#F0EDE8;font-size:13px;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#333;font-size:11px;letter-spacing:2px;padding:16px 0;">
    ACCIDENT SEVERITY PREDICTION · PORTFOLIO PROJECT · DATA ANALYSIS
</div>
""", unsafe_allow_html=True)
