import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = 'https://rakhi5604.pythonanywhere.com'

st.set_page_config(
    page_title="Microgrid ML Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCTIONS TO FETCH DATA
# ============================================================

@st.cache_data(ttl=5)
def fetch_sensor_data():
    """Fetch real-time sensor data from API"""
    try:
        response = requests.get(f'{API_URL}/api/sensor-data', timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching sensor data: {e}")
        return None

@st.cache_data(ttl=5)
def fetch_model_info():
    """Check if ML model is loaded"""
    try:
        response = requests.get(f'{API_URL}/api/model-info', timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(sensor_data):
    """Make ML prediction based on current sensor data"""
    try:
        # Prepare payload for prediction
        payload = {
            'Month': datetime.now().month,
            'Day': datetime.now().day,
            'DayOfWeek': datetime.now().weekday(),
            'DayOfYear': datetime.now().timetuple().tm_yday,
            'Irradiance_W_m2': 800,  # You can update this with real sensor
            'Temperature_C': 25       # You can update this with real sensor
        }
        
        response = requests.post(
            f'{API_URL}/api/predict',
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# ============================================================
# MAIN DASHBOARD
# ============================================================

# Title and Header
st.title("⚡ Microgrid AI/ML Monitoring Dashboard")
st.markdown("Real-time monitoring with machine learning predictions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Refresh interval
    refresh_interval = st.slider("Auto-refresh (seconds)", 5, 60, 10)
    
    st.markdown("---")
    
    # Model status
    model_info = fetch_model_info()
    if model_info and model_info.get('model_loaded'):
        st.success("🤖 ML Model: Active")
        st.info(f"Model Type: {model_info.get('model_type', 'Unknown')}")
    else:
        st.warning("🤖 ML Model: Not Loaded")
    
    st.markdown("---")
    
    # System status
    st.subheader("🔌 System Status")
    sensor_data = fetch_sensor_data()
    if sensor_data:
        st.success("✓ API Connected")
    else:
        st.error("✗ API Disconnected")

# ============================================================
# MAIN CONTENT
# ============================================================

# Fetch data
data = fetch_sensor_data()

if data:
    timestamp = data.get('timestamp', datetime.now().isoformat())
    st.caption(f"Last updated: {timestamp}")
    
    # ========== ROW 1: KEY METRICS ==========
    st.subheader("📊 Real-Time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        solar_power = data['solar']['power']
        st.metric(
            label="☀️ Solar Power",
            value=f"{solar_power:.2f} kW",
            delta=f"{data['solar']['efficiency']}% efficiency"
        )
    
    with col2:
        battery_level = data['battery']['level']
        st.metric(
            label="🔋 Battery Level",
            value=f"{battery_level}%",
            delta=f"{data['battery']['health']}% health"
        )
    
    with col3:
        grid_voltage = data['grid']['voltage']
        st.metric(
            label="⚡ Grid Voltage",
            value=f"{grid_voltage} V",
            delta=f"PF: {data['grid']['power_factor']}"
        )
    
    with col4:
        load = data['load']
        st.metric(
            label="🏠 Load",
            value=f"{load:.2f} kW",
            delta="Active"
        )
    
    st.markdown("---")
    
    # ========== ROW 2: DETAILED DATA ==========
    st.subheader("🔍 Detailed System Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ☀️ Solar Panel")
        st.write(f"**Voltage:** {data['solar']['voltage']:.1f} V")
        st.write(f"**Current:** {data['solar']['current']:.1f} A")
        st.write(f"**Power:** {data['solar']['power']:.2f} kW")
        st.write(f"**Temperature:** {data['solar']['temperature']}°C")
        
        # Progress bar for efficiency
        st.progress(data['solar']['efficiency'] / 100)
        st.caption(f"Efficiency: {data['solar']['efficiency']}%")
    
    with col2:
        st.markdown("### 🔋 Battery System")
        st.write(f"**Voltage:** {data['battery']['voltage']:.1f} V")
        st.write(f"**Current:** {data['battery']['current']:.1f} A")
        st.write(f"**Power:** {data['battery']['power']:.2f} kW")
        st.write(f"**Temperature:** {data['battery']['temperature']}°C")
        
        # Progress bar for battery level
        st.progress(data['battery']['level'] / 100)
        st.caption(f"Charge: {data['battery']['level']}%")
    
    with col3:
        st.markdown("### ⚡ Grid Connection")
        st.write(f"**Voltage:** {data['grid']['voltage']} V")
        st.write(f"**Current:** {data['grid']['current']:.1f} A")
        st.write(f"**Frequency:** {data['grid']['frequency']} Hz")
        st.write(f"**Power Factor:** {data['grid']['power_factor']}")
        
        # Status indicator
        if data['grid']['voltage'] > 220 and data['grid']['voltage'] < 240:
            st.success("✓ Grid Normal")
        else:
            st.warning("⚠ Grid Abnormal")
    
    st.markdown("---")
    
    # ========== ROW 3: ML PREDICTIONS ==========
    st.subheader("🤖 AI/ML Predictions")
    
    prediction = make_prediction(data)
    
    if prediction and prediction.get('status') == 'success':
        col1, col2 = st.columns(2)
        
        with col1:
            pred_value = prediction.get('prediction', 0)
            st.metric(
                label="🔮 Predicted Solar Output (Next Hour)",
                value=f"{pred_value:.2f} kW",
                delta="ML Forecast"
            )
            
            # Create a simple gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_value,
                title={'text': "Predicted Power (kW)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Prediction Details")
            st.write(f"**Model Type:** RandomForestRegressor")
            st.write(f"**Features Used:** {len(prediction.get('features_used', []))}")
            st.write(f"**Prediction Time:** {prediction.get('timestamp', 'N/A')}")
            
            st.info("💡 This prediction is based on historical patterns and current conditions.")
    else:
        st.warning("⚠️ ML predictions unavailable. Check model status in sidebar.")
    
    st.markdown("---")
    
    # ========== ROW 4: ENERGY BALANCE ==========
    st.subheader("⚖️ Energy Balance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create energy flow visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Solar', 'Battery', 'Grid'],
            y=[data['solar']['power'], data['battery']['power'], 0],
            name='Generation',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=['Load'],
            y=[data['load']],
            name='Consumption',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Energy Flow (kW)",
            xaxis_title="Source/Load",
            yaxis_title="Power (kW)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Energy Summary")
        
        total_generation = data['solar']['power'] + data['battery']['power']
        total_consumption = data['load']
        surplus = total_generation - total_consumption
        
        st.write(f"**Total Generation:** {total_generation:.2f} kW")
        st.write(f"**Total Consumption:** {total_consumption:.2f} kW")
        
        if surplus > 0:
            st.success(f"**✓ Surplus:** {surplus:.2f} kW")
        elif surplus < 0:
            st.error(f"**✗ Deficit:** {abs(surplus):.2f} kW")
        else:
            st.info("**Balance:** 0 kW")
    
    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()

else:
    st.error("❌ Unable to fetch data from API. Please check your connection.")
    st.info(f"API URL: {API_URL}")
    
    if st.button("🔄 Retry Connection"):
        st.rerun()
