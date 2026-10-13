import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os

st.set_page_config(
    page_title='Villain Predictor | Marvel & DC',
    page_icon='🦹',
    layout='centered',
    initial_sidebar_state='collapsed',
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0d2e 50%, #0d1a2e 100%);
    min-height: 100vh;
}
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ff4444, #ff8800, #ffdd00, #44aaff, #aa44ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub { text-align: center; color: #aabbcc; font-size: 1.05rem; margin-bottom: 2rem; }
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.section-label {
    color: #88aaff; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; font-size: 1.1rem !important;
    font-weight: 700 !important; padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
.result-villain {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(153,27,27,0.3));
    border: 1px solid rgba(239,68,68,0.5); border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.result-hero {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.3));
    border: 1px solid rgba(52,211,153,0.5); border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.result-emoji { font-size: 4rem; display: block; margin-bottom: 0.5rem; }
.result-label { font-size: 1.6rem; font-weight: 900; color: white; }
.result-prob { font-size: 1rem; color: #ccddee; margin-top: 0.3rem; }
label { color: #bbccee !important; }
</style>
''', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class VillainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(26, 8)
        self.output = nn.Linear(8, 1)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

@st.cache_resource
def load_artifacts():
    model = VillainNet()
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'villain_net.pt'),
        map_location='cpu', weights_only=True
    ))
    model.eval()
    scaler       = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.joblib'))
    tidy_cats    = joblib.load(os.path.join(MODELS_DIR, 'tidy_categories.joblib'))
    median_year  = joblib.load(os.path.join(MODELS_DIR, 'median_year.joblib'))
    return model, scaler, feature_cols, tidy_cats, median_year

model, scaler, feature_cols, tidy_cats, median_year = load_artifacts()

def build_features(publisher, sex, identity, alive, eye_color, hair_color, appearances, year):
    def tidy_val(val, top_vals):
        return val if val in top_vals else 'Other'
    row = {
        'publisher': publisher,
        'sex':       tidy_val(sex, tidy_cats['SEX']),
        'identity':  tidy_val(identity, tidy_cats['ID']),
        'alive':     tidy_val(alive, tidy_cats['ALIVE']),
        'eye':       tidy_val(eye_color, tidy_cats['EYE']),
        'hair':      tidy_val(hair_color, tidy_cats['HAIR']),
    }
    df_row = pd.get_dummies(pd.DataFrame([row]), dtype=float)
    dummy_cols = [c for c in feature_cols if c not in ('appearances', 'year')]
    for col in dummy_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
    df_row = df_row[dummy_cols]
    df_row['appearances'] = np.log1p(appearances)
    df_row['year']        = float(year)
    df_row = df_row[feature_cols]
    scaled = scaler.transform(df_row.values)
    return torch.tensor(scaled, dtype=torch.float32)

st.markdown('<h1 class="hero-title">🦹 Villain Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Neural Network · Marvel & DC Comic Characters</p>', unsafe_allow_html=True)

st.markdown('<div class="glass-card"><div class="section-label">📋 Character Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    publisher  = st.selectbox('Publisher',   ['DC', 'Marvel'])
    sex        = st.selectbox('Sex',         ['Male', 'Female', 'Other'])
    identity   = st.selectbox('Identity',    ['Secret', 'Public', 'Unknown', 'Other'])
with col2:
    alive      = st.selectbox('Status',      ['Living', 'Deceased', 'Other'])
    eye_color  = st.selectbox('Eye Color',   ['Unknown', 'Blue', 'Brown', 'Black', 'Green', 'Other'])
    hair_color = st.selectbox('Hair Color',  ['Unknown', 'Black', 'Brown', 'Blond', 'White', 'Other'])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card"><div class="section-label">📊 Stats</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    appearances = st.slider('Comic Appearances', 1, 4043, 10)
with col4:
    year = st.slider('First Appearance Year', 1939, 2013, int(median_year))
st.markdown('</div>', unsafe_allow_html=True)

if st.button('⚡ Predict Alignment'):
    x_tensor = build_features(publisher, sex, identity, alive, eye_color, hair_color, appearances, year)
    with torch.no_grad():
        prob = model(x_tensor).item()
    is_villain = prob >= 0.5
    pct = round(prob * 100, 1)
    if is_villain:
        st.markdown(f'''
        <div class="result-villain">
            <span class="result-emoji">😈</span>
            <div class="result-label">VILLAIN</div>
            <div class="result-prob">Villain confidence: <strong>{pct}%</strong></div>
        </div>''', unsafe_allow_html=True)
    else:
        good_pct = round((1 - prob) * 100, 1)
        st.markdown(f'''
        <div class="result-hero">
            <span class="result-emoji">🦸</span>
            <div class="result-label">HERO</div>
            <div class="result-prob">Hero confidence: <strong>{good_pct}%</strong></div>
        </div>''', unsafe_allow_html=True)
    st.markdown('---')
    c5, c6 = st.columns(2)
    c5.metric('🦸 Hero',    f'{round((1-prob)*100,1)}%')
    c6.metric('😈 Villain', f'{pct}%')

with st.expander('ℹ️ About this model'):
    st.markdown('''
**Dataset:** FiveThirtyEight Marvel & DC Comic Characters

**Architecture (VillainNet):**
- Input: 26 features (publisher, sex, identity, alive, eye/hair color, appearances, year)
- Hidden: 26 → 8 neurons (ReLU)
- Output: 8 → 1 neuron (Sigmoid)

**Training:** 100 epochs · Adam lr=0.01 · MSE loss · ~67% test accuracy

**Bugs fixed in notebook:**
- Cell 8: Missing comma })  dtype → }), dtype, typo loglp → log1p, APPEARANCE → APPEARANCES
- Cell 9: X.shape / X.columns → x.shape / x.columns
- Cell 17: self.out(x) → self.output(x)
    ''')

st.markdown('<p style="text-align:center;color:#556677;font-size:0.8rem;margin-top:2rem;">Built with PyTorch · Streamlit · Marvel & DC Data</p>', unsafe_allow_html=True)
