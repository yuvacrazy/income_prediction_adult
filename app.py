import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("salary.pkl")

# Page config
st.set_page_config(page_title="💼 Employer Salary Prediction", layout="wide")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Fill the details to predict your salary range.")
st.sidebar.markdown("---")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: #fafafa; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Title
# -------------------------------
st.title("💼 Employer Salary Prediction System")
st.markdown("###### Project by **Yuvaraja P | Final Year CSE - IoT**")
st.write("Predict your salary category using Machine Learning 🚀")

# -------------------------------
# Input Form
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Enter your details")

    age = st.slider("Age", 18, 70, 25)
    education = st.selectbox("Education Level", ["10th", "12th", "Diploma", "Bachelors", "Masters", "PhD"])
    occupation = st.text_input("Occupation (e.g., Engineer, Teacher)")
    hours = st.slider("Hours worked per week", 1, 100, 40)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

    # Input vector (⚠️ update encoding as per preprocessing used in training)
    X_input = np.array([[age, hours, len(education), len(occupation),
                         1 if gender == "Male" else 0,
                         1 if marital == "Married" else 0,
                         1 if marital == "Divorced" else 0,
                         1 if marital == "Single" else 0]])

with col2:
    st.subheader("📊 Salary Distribution")
    sample_data = np.random.normal(50000, 15000, 200)
    fig, ax = plt.subplots()
    sns.histplot(sample_data, kde=True, ax=ax, color="skyblue")
    ax.set_title("Sample Salary Distribution")
    st.pyplot(fig)

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("---")
if st.button("🔮 Predict Salary"):
    prediction = model.predict(X_input)[0]

    # Result Card
    st.markdown(
        f"""
        <div style="background-color:#f9f9f9; padding:20px; border-radius:15px; 
        border:2px solid #ddd; text-align:center;">
        <h2>🎯 Prediction Result</h2>
        <p style="font-size:22px;">Based on your details, the predicted salary category is:</p>
        <h1 style="color:#2e86de;">{prediction}</h1>
        </div>
        """, unsafe_allow_html=True
    )

    # -------------------------------
    # Feature Importance Plot
    # -------------------------------
    st.subheader("📈 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = ["Age", "Hours", "Edu_len", "Occ_len", "Gender_Male", "Married", "Divorced", "Single"]

        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features, palette="viridis", ax=ax)
        ax.set_title("Model Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💻 Developed by **Yuvaraja P | Paavai Engineering College**")
