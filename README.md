# ❤️ Heart Disease Prediction Dashboard

## 📌 Overview

This project is a Machine Learning-based web application that predicts whether a patient is at risk of heart disease based on medical parameters. The application is built using **Streamlit** for an interactive user interface and **Logistic Regression** for prediction.

---

## 🚀 Features

* ✅ Predict heart disease risk in real-time
* 📊 Interactive dashboard using Streamlit
* 🧠 Machine Learning model (Logistic Regression)
* 📈 Data visualization (charts & heatmap)
* 🧾 User-friendly input form

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```
heart_disease/
│
├── app.py                # Streamlit dashboard
├── train_model.py        # Model training script
├── dataset.csv           # Dataset
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ How It Works

1. Load dataset containing patient health details
2. Preprocess data (scaling, feature selection)
3. Train a Logistic Regression model
4. Save model and scaler using pickle
5. Streamlit app loads model and predicts results based on user input

---

## ▶️ How to Run the Project

### 🔹 Step 1: Install dependencies

```
pip install -r requirements.txt
```

### 🔹 Step 2: Train the model

```
python train_model.py
```

### 🔹 Step 3: Run the app

```
streamlit run app.py
```

---

## 📊 Dataset Description

The dataset includes the following features:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Rest ECG
* Maximum heart rate
* Exercise-induced angina
* Oldpeak
* ST slope

**Target:**

* `0` → Normal
* `1` → Heart Disease

---

## 🎯 Future Improvements

* 🔥 Use advanced models (Random Forest, XGBoost)
* 🌐 Deploy the app online
* 📱 Improve UI/UX design
* 📊 Add more detailed analytics

---

## 👩‍💻 Author

**Dona Maria Siju**

---

## ⭐ Acknowledgment

This project is developed for learning and academic purposes to understand Machine Learning and deployment using Streamlit.
