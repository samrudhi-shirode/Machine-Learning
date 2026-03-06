import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model and pipeline
# -----------------------------
model = joblib.load("model.joblib")
pre = joblib.load("pre.joblib")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------
# Custom CSS Design
# -----------------------------
st.markdown("""
<style>

.main {
background-color:#f5f7fb;
}

.title{
text-align:center;
font-size:40px;
font-weight:bold;
color:#2c3e50;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:40px;
}

.card{
background-color:white;
padding:30px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

.result-success{
color:green;
font-size:30px;
font-weight:bold;
text-align:center;
}

.result-danger{
color:red;
font-size:30px;
font-weight:bold;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<p class="title">⚙️ Machine Failure Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI based predictive maintenance model</p>', unsafe_allow_html=True)

# -----------------------------
# Input Section
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    type_machine = st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=320.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=330.0,
        value=310.0
    )

with col2:

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=50
    )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction Button
# -----------------------------
st.write("")

if st.button("🔍 Predict Machine Failure"):

    input_data = pd.DataFrame({
        "Type":[type_machine],
        "Air temperature [K]":[air_temp],
        "Process temperature [K]":[process_temp],
        "Rotational speed [rpm]":[rotational_speed],
        "Torque [Nm]":[torque],
        "Tool wear [min]":[tool_wear]
    })

    # preprocessing
    input_pre = pre.transform(input_data)

    # prediction
    prediction = model.predict(input_pre)

    st.write("")

    if prediction[0] == 1:

        st.markdown(
        '<p class="result-danger">⚠️ Machine Failure Likely</p>',
        unsafe_allow_html=True)

    else:

        st.markdown(
        '<p class="result-success">✅ Machine Working Normally</p>',
        unsafe_allow_html=True)
