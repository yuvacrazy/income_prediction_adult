import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("salary_model.pkl")

# Page config
st.set_page_config(page_title="💼 Employer Salary Prediction", layout="wide")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")
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
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔮 Predict", "📊 Visualize"])

# -------------------------------
# Home Tab
# -------------------------------
with tab1:
    st.title("💼 Employer Salary Prediction System")
    st.markdown("###### Project by **Yuvaraja P | Final Year CSE - IoT**")
    st.write(
        """
        Welcome to the **Employer Salary Prediction System** 🚀  
        This project uses **Machine Learning** to predict whether a person’s salary falls into a certain category based on their profile details.  

        ### Features:
        - Clean UI with modern dashboard design  
        - Interactive input form for salary prediction  
        - Visual insights with charts & feature importance  

        Navigate to **Predict Tab** to try it out!
        """
    )
    st.success("Tip: Use sidebar to change theme 🌙")

# -------------------------------
# Predict Tab
# -------------------------------
with tab2:
    st.header("🔮 Salary Prediction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📌 Enter your details")

        age = st.slider("Age", 18, 70, 25)
        education = st.selectbox("Education Level", ["10th", "12th", "Diploma", "Bachelors", "Masters", "PhD"])
        occupation = st.text_input("Occupation (e.g., Engineer, Teacher)")
        hours = st.slider("Hours worked per week", 1, 100, 40)
        g
