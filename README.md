# Employer Salary Prediction App

This is a Streamlit app that predicts whether a person earns >50K or ≤50K using a trained machine learning model.

## Features

- Inputs: Age, Education, Occupation, Hours-per-week, Gender, Marital Status
- Backend: Random Forest model saved with joblib
- Frontend: Built with Streamlit

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Options

- [Render.com](https://render.com)
- [Streamlit Cloud](https://streamlit.io/cloud)

## Files

- `app.py` — Main app interface
- `requirements.txt` — Python dependencies
- `salary_model.pkl` — Your trained ML model
- `encoders.pkl` — Label encoders used in training