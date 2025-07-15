import streamlit as st
import joblib
import numpy as np

# Load the model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# App title
st.set_page_config(page_title="💼 Employer Salary Predictor", layout="centered")
st.title("💼 Employer Salary Prediction")
st.markdown("Predict whether a person's salary is greater than 50K or not.")

# Form for input
with st.form("income"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("📅 Age", min_value=17, max_value=100, step=1)
        education = st.selectbox("🎓 Education", encoders['education'].classes_)
        occupation = st.selectbox("💼 Occupation", encoders['occupation'].classes_)
        hours = st.slider("⏱ Hours per Week", 1, 99, 40)

    with col2:
        gender = st.radio("👤 Gender", encoders['gender'].classes_)
        marital_status = st.selectbox("💍 Marital Status", encoders['marital-status'].classes_)
        capital_gain = st.number_input("📈 Capital Gain", 0)
        capital_loss = st.number_input("📉 Capital Loss", 0)

    # Predict button
    submit = st.form_submit_button("🔮 Predict Salary Class")

# On submit
if submit:
    try:
        X_input = np.array([[
            age,
            encoders['education'].transform([education])[0],
            encoders['occupation'].transform([occupation])[0],
            hours,
            encoders['gender'].transform([gender])[0],
            encoders['marital-status'].transform([marital_status])[0],
            capital_gain,
            capital_loss
        ]])

        prediction = model.predict(X_input)[0]
        result = ">50K" if prediction == 1 else "≤50K"
        st.success(f"💰 Predicted Salary Class: **{result}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("<center>🚀 Created by <strong>Yuvaraja</strong></center>", unsafe_allow_html=True)
