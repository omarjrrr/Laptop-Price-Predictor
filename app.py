import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from forex_python.converter import CurrencyRates  

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_resources():
    pipe_path = BASE_DIR / 'pipe.pkl'
    df_path = BASE_DIR / 'df.pkl'

    if not pipe_path.exists():
        raise FileNotFoundError(f"Missing model file: {pipe_path}")
    if not df_path.exists():
        raise FileNotFoundError(f"Missing data file: {df_path}")

    with open(pipe_path, 'rb') as f:
        pipe_local = pickle.load(f)
    with open(df_path, 'rb') as f:
        df_local = pickle.load(f)

    return pipe_local, df_local


try:
    pipe, df = load_resources()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("💻 Laptop Price Predictor (in USD)")

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

cpu_col = find_col(df, ['Cpu Brand', 'Cpu brand', 'Cpu', 'CpuName'])
gpu_col = find_col(df, ['Gpu Brand', 'Gpu brand', 'Gpu'])
ram_col = find_col(df, ['Ram', 'RAM'])
os_col = find_col(df, ['Os', 'os', 'OpSys'])
type_col = find_col(df, ['TypeName', 'Type', 'Type name'])

company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df[type_col].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160',
      '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)
cpu = st.selectbox('CPU', df[cpu_col].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df[gpu_col].unique())
os_name = st.selectbox('Operating System', df[os_col].unique())

if st.button('💰 Predict Price in USD'):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = pd.DataFrame([{
        'Company': company,
        type_col: type_name,
        ram_col: int(ram),
        'Weight': float(weight),
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': float(ppi),
        cpu_col: cpu,
        'HDD': int(hdd),
        'SSD': int(ssd),
        gpu_col: gpu,
        os_col: os_name
    }])
    predicted_price_inr = int(np.exp(pipe.predict(query)[0]))

    try:
        c = CurrencyRates()
        rate = c.get_rate('USD', 'INR')  
        predicted_price_usd = predicted_price_inr / rate
    except Exception:
        rate = 83 
        predicted_price_usd = predicted_price_inr / rate

    st.subheader("✅ Predicted Laptop Price:")
    st.success(f"🇺🇸 ${predicted_price_usd:,.2f} USD (1 USD = {rate:.2f} INR)")
