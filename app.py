import streamlit as st
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page config – MUST be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 Crop Suitability Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Full custom CSS – dark‑green premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* ── Background – animated gradient ── */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 40%, #0f3d1e 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(46,125,50,0.25) 0%, rgba(27,94,32,0.15) 100%);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #69f0ae, #b9f6ca, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.hero p {
    color: rgba(255,255,255,0.65);
    font-size: 1.1rem;
    font-weight: 300;
    margin: 0;
}

/* ── Stats badge row ── */
.badge-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(76,175,80,0.15);
    border: 1px solid rgba(76,175,80,0.35);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    color: #69f0ae;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Section cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(6px);
}
.card-title {
    color: #69f0ae;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Input labels ── */
label, .stNumberInput label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Number input boxes ── */
.stNumberInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(76,175,80,0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: border 0.25s ease;
}
.stNumberInput input:focus {
    border: 1px solid #69f0ae !important;
    box-shadow: 0 0 0 3px rgba(105,240,174,0.15) !important;
}
.stNumberInput button {
    background: rgba(76,175,80,0.2) !important;
    border: none !important;
    color: #69f0ae !important;
    border-radius: 8px !important;
}
.stNumberInput button:hover {
    background: rgba(76,175,80,0.4) !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #2e7d32, #69f0ae) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%) !important;
    color: white !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(46,125,50,0.5);
    text-transform: uppercase;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(46,125,50,0.7) !important;
    background: linear-gradient(135deg, #388e3c 0%, #4caf50 50%, #81c784 100%) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(46,125,50,0.3) 0%, rgba(27,94,32,0.2) 100%);
    border: 1px solid rgba(105,240,174,0.5);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeInUp 0.5s ease;
}
.result-crop {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #69f0ae, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: capitalize;
    margin: 0.5rem 0;
}
.result-label {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.result-conf {
    color: #69f0ae;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 50px;
    height: 8px;
    margin-top: 0.75rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 50px;
    background: linear-gradient(90deg, #2e7d32, #69f0ae);
    transition: width 1s ease;
}

/* ── Tips box ── */
.tip-box {
    background: rgba(255,193,7,0.07);
    border: 1px solid rgba(255,193,7,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: rgba(255,255,255,0.65);
    font-size: 0.83rem;
    line-height: 1.6;
    margin-top: 1.2rem;
}
.tip-box b { color: #ffd54f; }

/* ── Metrics row ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #69f0ae;
}
.metric-lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.2rem;
}

/* ── How it works steps ── */
.step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.step-num {
    background: linear-gradient(135deg, #2e7d32, #69f0ae);
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-text {
    color: rgba(255,255,255,0.7);
    font-size: 0.88rem;
    line-height: 1.6;
    padding-top: 3px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.footer a { color: #69f0ae; text-decoration: none; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.6s ease both; }

/* ── Crop emoji map ── */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Crop → emoji mapping
# ─────────────────────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍑", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕",
}

def get_emoji(crop):
    return CROP_EMOJI.get(crop.lower(), "🌱")

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero fade-in">
    <h1>🌿 Crop Suitability Predictor</h1>
    <p>AI‑powered crop recommendation based on soil nutrients & environmental conditions</p>
    <div class="badge-row">
        <span class="badge">🧪 7 Soil Parameters</span>
        <span class="badge">🤖 K‑Nearest Neighbours Model</span>
        <span class="badge">📊 97.95% Accuracy</span>
        <span class="badge">🌍 22 Crop Classes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT  –  left column (form)  |  right column (info + result)
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── LEFT – Input form ──────────────────────────────────────────────────────
with left:
    # Soil nutrients card
    st.markdown("""
    <div class="card">
        <div class="card-title">🧪 Soil Nutrients</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0,
                            value=90.0, step=1.0, help="Nitrogen content in soil (kg/ha)")
    with c2:
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0,
                            value=42.0, step=1.0, help="Phosphorus content in soil (kg/ha)")
    with c3:
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0,
                            value=43.0, step=1.0, help="Potassium content in soil (kg/ha)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Environmental conditions card
    st.markdown("""
    <div class="card">
        <div class="card-title">🌡️ Environmental Conditions</div>
    </div>
    """, unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=55.0,
                               value=25.5, step=0.1, help="Average temperature in °C")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0,
                                   value=80.0, step=0.1, help="Relative humidity in %")
    with e2:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0,
                             value=6.5, step=0.01, help="Soil acidity/alkalinity")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0,
                                   value=200.0, step=1.0, help="Annual rainfall in mm")

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    predict = st.button("🌾  Predict the Best Crop", type="primary", use_container_width=True)

    # Tip box
    st.markdown("""
    <div class="tip-box">
        <b>💡 Tip:</b> Use values from a recent soil test report for the most accurate recommendation.
        N, P, K are typically measured in kg/ha. pH of 6–7 suits most crops.
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT – Info + Result ──────────────────────────────────────────────────
with right:
    # Model accuracy metrics
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Model Performance</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-val">97.95%</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-val">97.93%</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    m3, m4 = st.columns(2)
    with m3:
        st.markdown('<div class="metric-card"><div class="metric-val">1500+</div><div class="metric-lbl">Training Rows</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><div class="metric-val">22</div><div class="metric-lbl">Crop Types</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="card">
        <div class="card-title">⚙️ How It Works</div>
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-text">Enter your soil nutrient values (N, P, K) and environmental readings.</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-text">Values are scaled using StandardScaler to normalise the ranges.</div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-text">The K‑NN model finds the 5 most similar historical records in the dataset.</div>
        </div>
        <div class="step">
            <div class="step-num">4</div>
            <div class="step-text">The majority crop among those neighbours is returned as the recommendation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction result ──
    if predict:
        input_df = pd.DataFrame({
            "N": [n], "P": [p], "K": [k],
            "temperature": [temp], "humidity": [humidity],
            "ph": [ph], "rainfall": [rainfall],
        })

        with st.spinner("Analysing soil conditions…"):
            crop = model.predict(input_df)[0]
            conf = model.predict_proba(input_df).max() * 100

        emoji = get_emoji(crop)
        bar_w = int(conf)

        st.markdown(f"""
        <div class="result-box fade-in">
            <div class="result-label">✅ Recommended Crop</div>
            <div class="result-crop">{emoji}  {crop.title()}</div>
            <div class="result-conf">Confidence: {conf:.1f}%</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{bar_w}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini explanation
        st.markdown(f"""
        <div class="tip-box" style="margin-top:1rem; border-color:rgba(105,240,174,0.2);">
            <b>🌱 {crop.title()}</b> is the most suitable crop for the given conditions.
            The model is <b>{conf:.1f}% confident</b> based on {5} nearest neighbours
            in the training dataset.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by
    <a href="https://github.com/KishoreG2006" target="_blank">KishoreG2006</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/KishoreG2006/Crop-Suitability-Classification" target="_blank">
        View Source on GitHub
    </a>
    &nbsp;·&nbsp; Dataset: Kaggle Crop Recommendation &nbsp;·&nbsp; © 2026
</div>
""", unsafe_allow_html=True)
