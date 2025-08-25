import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model & encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# ----------------- UI Design -----------------
st.set_page_config(page_title="💼 Employer Salary Prediction", page_icon="💰", layout="centered")

st.title("💼 Employer Salary Prediction App")
st.markdown("### Project by **Yuvaraja P** (Final Year CSE - IoT, Paavai Engineering College)")
st.write("Fill in the details below and click **Predict Salary** to see the result.")

st.markdown("---")

# ----------------- Input Form -----------------
with st.form("prediction_form"):
    st.subheader("📋 Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 17, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["10th", "12th", "Bachelors", "Masters", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        occupation = st.selectbox("Occupation", ["Tech", "Management", "Clerical", "Sales", "Other"])
        workclass = st.selectbox("Workclass", ["Private", "Self-Employed", "Government", "Other"])
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        country = st.selectbox("Country", ["United States", "India", "Other"])

    # Submit Button
    submit = st.form_submit_button("🔮 Predict Salary")

# ----------------- Prediction Section -----------------
if submit:
    try:
        # Encode categorical inputs
        gender_enc = encoders["gender"].transform([gender])[0]
        education_enc = encoders["education"].transform([education])[0]
        marital_enc = encoders["marital_status"].transform([marital_status])[0]
        occupation_enc = encoders["occupation"].transform([occupation])[0]
        workclass_enc = encoders["workclass"].transform([workclass])[0]
        country_enc = encoders["country"].transform([country])[0]

        # Arrange input
        X_input = np.array([[age, hours_per_week, gender_enc, education_enc,
                             marital_enc, occupation_enc, workclass_enc, country_enc]])

        # Prediction
        pred = model.predict(X_input)[0]
        st.success(f"💰 Estimated Salary Category: **{pred}**")

        # ----------------- Visualization 1: Feature Values -----------------
        st.subheader("📊 Feature Contribution Overview")

        feature_names = ["Age", "Hours/Week", "Gender", "Education",
                         "Marital Status", "Occupation", "Workclass", "Country"]
        input_values = [age, hours_per_week, gender_enc, education_enc,
                        marital_enc, occupation_enc, workclass_enc, country_enc]

        df_features = pd.DataFrame({"Feature": feature_names, "Value": input_values})

        fig, ax = plt.subplots()
        ax.barh(df_features["Feature"], df_features["Value"], color="skyblue")
        ax.set_xlabel("Encoded Value / Numeric Input")
        ax.set_title("Input Features Overview")
        st.pyplot(fig)

        # ----------------- Visualization 2: Prediction Probability -----------------
        if hasattr(model, "predict_proba"):
            st.subheader("📈 Prediction Probability")

            probs = model.predict_proba(X_input)[0]
            df_probs = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probs
            })

            fig2, ax2 = plt.subplots()
            ax2.bar(df_probs["Category"], df_probs["Probability"], color=["#4CAF50", "#2196F3"])
            ax2.set_ylabel("Probability")
            ax2.set_title("Confidence in Prediction")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

st.markdown("---")
st.markdown("👨‍💻 Developed by **Yuvaraja P** | Final Year CSE (IoT) | Paavai Engineering College")

