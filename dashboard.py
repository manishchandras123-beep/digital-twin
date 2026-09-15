import numpy as np
import pandas as pd
import streamlit as st

from disease_engine import BASELINE, DISEASE_PROFILES
from patient_simulator import PatientVitalsStream
from risk_scoring import calculate_news2
from ml_predictor import train_model, predict_risk
from database import init_db, insert_reading, get_all_readings, clear_patient

st.set_page_config(page_title="Patient Digital Twin", layout="centered")

init_db()

st.title("🩺 Patient Digital Twin")
st.caption("Simulated patient — educational project, not a real medical tool. All predictions are trained on synthetic data.")

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

name_input = st.text_input("Patient name", value=st.session_state.patient_name, placeholder="e.g. John Doe")

disease_key = st.selectbox(
    "Condition to simulate",
    options=list(DISEASE_PROFILES.keys()),
    format_func=lambda k: DISEASE_PROFILES[k]["label"],
)

restart_needed = (
    name_input != st.session_state.patient_name
    or disease_key != st.session_state.get("disease_key")
)

if restart_needed:
    st.session_state.patient_name = name_input
    st.session_state.disease_key = disease_key
    st.session_state.pop("stream", None)

if not st.session_state.patient_name:
    st.info("Enter a patient name above to start.")
    st.stop()

if "stream" not in st.session_state:
    st.session_state.stream = PatientVitalsStream(disease=disease_key, condition_span_readings=60, seed=None)

model_cache_key = f"model_{disease_key}"
if model_cache_key not in st.session_state:
    with st.spinner("Training AI risk model for this condition..."):
        st.session_state[model_cache_key] = train_model(disease=disease_key)
ml_model = st.session_state[model_cache_key]

st.divider()

mode = st.radio("Mode", ["Live Monitor", "What-If Simulator"], horizontal=True)

if mode == "Live Monitor":

    c1, c2 = st.columns(2)
    with c1:
        next_clicked = st.button("▶ Next Reading", use_container_width=True)
    with c2:
        reset_clicked = st.button("🔄 Restart", use_container_width=True)

    if reset_clicked:
        clear_patient(st.session_state.patient_name)
        st.session_state.stream = PatientVitalsStream(disease=disease_key, condition_span_readings=60, seed=None)
        st.rerun()

    if next_clicked:
        reading = st.session_state.stream.next_reading()
        risk = calculate_news2(reading)
        reading["patient_name"] = st.session_state.patient_name
        reading["disease"] = disease_key
        reading["news2_score"] = risk["news2_score"]
        reading["risk_label"] = risk["risk_label"]
        insert_reading(reading)

    data = get_all_readings(patient_name=st.session_state.patient_name)

    if not data:
        st.info("Click 'Next Reading' to start monitoring this patient.")
    else:
        df = pd.DataFrame(data)
        latest = df.iloc[-1]

        risk_color = {
            "Normal": "🟢", "Low Risk": "🟡", "Medium Risk": "🟠", "High Risk": "🔴"
        }.get(latest["risk_label"], "⚪")

        st.markdown(f"## {risk_color} {st.session_state.patient_name} — {latest['risk_label']}")
        st.caption(f"Rule-based NEWS2 score: {latest['news2_score']} / 15")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Heart Rate", f"{latest['heart_rate_bpm']:.0f}")
        m2.metric("Temp (°C)", f"{latest['temperature_C']:.1f}")
        m3.metric("Resp Rate", f"{latest['resp_rate_bpm']:.0f}")
        m4.metric("BP (sys)", f"{latest['systolic_bp_mmHg']:.0f}")
        m5.metric("SpO2 %", f"{latest['spo2_percent']:.0f}")

        st.subheader("🤖 AI prediction")
        ai_label, confidence, proba = predict_risk(ml_model, latest.to_dict())
        agree = "✅ agrees with rule-based score" if ai_label == latest["risk_label"] else "⚠️ differs from rule-based score"
        st.markdown(f"**{ai_label}** ({confidence*100:.1f}% confidence) — {agree}")
        proba_df = pd.DataFrame([proba]).T.rename(columns={0: "probability"})
        st.bar_chart(proba_df)

        st.subheader("Current vitals vs normal baseline")
        compare_df = pd.DataFrame({
            "current": [latest["heart_rate_bpm"], latest["temperature_C"], latest["resp_rate_bpm"],
                        latest["systolic_bp_mmHg"], latest["spo2_percent"]],
            "normal": [BASELINE["heart_rate_bpm"], BASELINE["temperature_C"], BASELINE["resp_rate_bpm"],
                       BASELINE["systolic_bp_mmHg"], BASELINE["spo2_percent"]],
        }, index=["Heart Rate", "Temp", "Resp Rate", "Systolic BP", "SpO2"])
        st.bar_chart(compare_df)

        with st.expander("Show trend and forecast"):
            df_indexed = df.set_index("timestamp")
            st.line_chart(df_indexed[["heart_rate_bpm", "temperature_C", "resp_rate_bpm",
                                       "systolic_bp_mmHg", "spo2_percent"]])

            st.write("**NEWS2 score trend + forecast (next 5 readings)**")
            if len(df) >= 3:
                y = df["news2_score"].values.astype(float)
                x = np.arange(len(y))
                recent = min(len(y), 10)
                coeffs = np.polyfit(x[-recent:], y[-recent:], 1)
                future_x = np.arange(len(y), len(y) + 5)
                future_y = np.clip(np.polyval(coeffs, future_x), 0, 15)

                forecast_df = pd.DataFrame({
                    "actual": list(y) + [None] * 5,
                    "forecast": [None] * (len(y) - 1) + [y[-1]] + list(future_y),
                })
                st.line_chart(forecast_df)
                st.caption(
                    f"Forecast is a simple linear trend projection based on the last {recent} readings, "
                    "not a trained model — treat it as a rough directional estimate only."
                )
            else:
                st.caption("Need at least 3 readings to show a forecast.")

else:
    st.caption("Try out different vitals for a hypothetical patient and see both the rule-based and AI risk assessment.")

    preset = st.selectbox("Preset", ["Healthy", "Early warning signs", "Severe"])
    presets = {
        "Healthy": dict(hr=75, temp=37.0, rr=16, sbp=118, spo2=98),
        "Early warning signs": dict(hr=105, temp=38.3, rr=21, sbp=105, spo2=94),
        "Severe": dict(hr=132, temp=39.6, rr=28, sbp=85, spo2=88),
    }
    p = presets[preset]

    hr = st.slider("Heart Rate (bpm)", 30, 220, p["hr"])
    temp = st.slider("Temperature (°C)", 33.0, 42.0, float(p["temp"]), step=0.1)
    rr = st.slider("Respiratory Rate", 4, 45, p["rr"])
    sbp = st.slider("Systolic BP (mmHg)", 50, 250, p["sbp"])
    spo2 = st.slider("SpO2 (%)", 70, 100, p["spo2"])

    vitals = {
        "heart_rate_bpm": hr, "temperature_C": temp, "resp_rate_bpm": rr,
        "systolic_bp_mmHg": sbp, "spo2_percent": spo2,
    }
    result = calculate_news2(vitals)
    ai_label, confidence, proba = predict_risk(ml_model, vitals)

    risk_color = {
        "Normal": "🟢", "Low Risk": "🟡", "Medium Risk": "🟠", "High Risk": "🔴"
    }.get(result["risk_label"], "⚪")

    st.divider()
    col_rule, col_ai = st.columns(2)
    with col_rule:
        st.markdown("**Rule-based (NEWS2)**")
        st.markdown(f"## {risk_color} {result['risk_label']}")
        st.caption(f"Score: {result['news2_score']} / 15")
        st.progress(min(result["news2_score"] / 15, 1.0))
    with col_ai:
        st.markdown("**AI prediction**")
        ai_color = {
            "Normal": "🟢", "Low Risk": "🟡", "Medium Risk": "🟠", "High Risk": "🔴"
        }.get(ai_label, "⚪")
        st.markdown(f"## {ai_color} {ai_label}")
        st.caption(f"Confidence: {confidence*100:.1f}%")
        st.progress(confidence)