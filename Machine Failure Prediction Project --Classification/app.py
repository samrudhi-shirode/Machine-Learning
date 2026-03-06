import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- CSS Styling ---------------- #

st.markdown("""
<style>

.main{
background: linear-gradient(120deg,#eef2f7,#f8fbff);
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#1f4e79;
}

.subtitle{
text-align:center;
font-size:20px;
color:gray;
margin-bottom:30px;
}

div.stButton > button:first-child {
background: linear-gradient(90deg,#ff7e5f,#feb47b);
color:white;
font-size:18px;
font-weight:bold;
border-radius:12px;
padding:12px 35px;
border:none;
}

div.stButton > button:hover {
background: linear-gradient(90deg,#ff6a4a,#ff9966);
color:white;
}

.result-success{
background:#d4edda;
padding:18px;
border-radius:10px;
font-size:22px;
color:#155724;
text-align:center;
}

.result-danger{
background:#f8d7da;
padding:18px;
border-radius:10px;
font-size:22px;
color:#721c24;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ---------------- #

st.markdown('<p class="title">⚙️ Machine Failure Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predictive Maintenance using Machine Learning</p>', unsafe_allow_html=True)

st.write("### 👩‍💻 Developed By: **Samrudhi Shirode**")

# ---------------- Load Model ---------------- #

pre = joblib.load("pre.joblib")
model = joblib.load("model.joblib")

# ---------------- Input Layout ---------------- #

col1, col2 = st.columns(2)

with col1:
    
    Machine_Type = st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )

    Air_Temp = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=320.0,
        value=300.0
    )

    Process_Temp = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=330.0,
        value=310.0
    )

with col2:

    Rotational_Speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    Torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    Tool_Wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=50
    )

st.write("")

# ---------------- Prediction Button ---------------- #

submit = st.button("🔍 Predict Machine Failure")

if submit:

    data = {
        "Type":[Machine_Type],
        "Air temperature [K]":[Air_Temp],
        "Process temperature [K]":[Process_Temp],
        "Rotational speed [rpm]":[Rotational_Speed],
        "Torque [Nm]":[Torque],
        "Tool wear [min]":[Tool_Wear]
    }

    xnew = pd.DataFrame(data)

    xnew_pre = pre.transform(xnew)

    preds = model.predict(xnew_pre)

    st.write("")

    if preds[0] == 1:
        st.markdown(
            '<p class="result-danger">⚠️ Machine Failure Likely</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="result-success">✅ Machine Working Normally</p>',
            unsafe_allow_html=True
        )