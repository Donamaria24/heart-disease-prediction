import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('dataset.csv')

# -----------------------------
# Train Model
# -----------------------------
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# -----------------------------
# UI Design
# -----------------------------
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("❤️ Heart Disease Prediction Dashboard")

# Sidebar
st.sidebar.header("Patient Input Features")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])
    restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope", [1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'chest pain type': cp, # Changed from 'cp'
        'resting bp s': trestbps, # Changed from 'trestbps'
        'cholesterol': chol, # Changed from 'chol'
        'fasting blood sugar': fbs, # Changed from 'fbs'
        'resting ecg': restecg, # Changed from 'restecg'
        'max heart rate': thalach, # Changed from 'thalach'
        'exercise angina': exang, # Changed from 'exang'
        'oldpeak': oldpeak,
        'ST slope': slope # Changed from 'slope'
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# -----------------------------
# Output
# -----------------------------
st.subheader("🧾 Patient Input")
st.write(input_df)

st.subheader("🔍 Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk (Healthy)")

st.subheader("📊 Prediction Probability")
st.write(f"Healthy: {probability[0][0]*100:.2f}%")
st.write(f"Heart Disease: {probability[0][1]*100:.2f}%")

# -----------------------------
# Data Visualization
# -----------------------------
st.subheader("📈 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Target Distribution")
    st.bar_chart(df['target'].value_counts())

with col2:
    st.write("Age Distribution")
    st.line_chart(df['age'])

st.subheader("📊 Correlation Heatmap")
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)