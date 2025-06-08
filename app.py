import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Students Social Media Addiction.csv")
    return df

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    label_cols = ['Gender', 'Academic_Level', 'Country',
                  'Most_Used_Platform', 'Affects_Academic_Performance',
                  'Relationship_Status']

    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['Student_ID', 'Addicted_Score'])
    y = df['Addicted_Score']
    return X, y, encoders

# Load data and train model
df = load_data()
X, y, encoders = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Prediksi Skor Kecanduan Sosial Media (1–10)")

# Input form
st.sidebar.header("Masukkan Informasi Mahasiswa:")
gender = st.sidebar.selectbox("Jenis Kelamin", encoders['Gender'].classes_)
age = st.sidebar.slider("Umur", 15, 30, 20)
academic_level = st.sidebar.selectbox("Tingkat Akademik", encoders['Academic_Level'].classes_)
country = st.sidebar.selectbox("Negara", encoders['Country'].classes_)
platform = st.sidebar.selectbox("Platform Paling Sering Digunakan", encoders['Most_Used_Platform'].classes_)
affect = st.sidebar.selectbox("Mempengaruhi Akademik?", encoders['Affects_Academic_Performance'].classes_)
sleep = st.sidebar.slider("Jam Tidur per Malam", 3.0, 10.0, 7.0)
mental_score = st.sidebar.slider("Skor Kesehatan Mental (1–10)", 1, 10, 5)
relationship = st.sidebar.selectbox("Status Hubungan", encoders['Relationship_Status'].classes_)
conflicts = st.sidebar.slider("Jumlah Konflik Karena Sosial Media", 0, 10, 2)
usage = st.sidebar.slider("Rata-rata Penggunaan Harian (jam)", 0.0, 12.0, 4.0)

# Prepare input
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': encoders['Gender'].transform([gender])[0],
    'Academic_Level': encoders['Academic_Level'].transform([academic_level])[0],
    'Country': encoders['Country'].transform([country])[0],
    'Avg_Daily_Usage_Hours': usage,
    'Most_Used_Platform': encoders['Most_Used_Platform'].transform([platform])[0],
    'Affects_Academic_Performance': encoders['Affects_Academic_Performance'].transform([affect])[0],
    'Sleep_Hours_Per_Night': sleep,
    'Mental_Health_Score': mental_score,
    'Relationship_Status': encoders['Relationship_Status'].transform([relationship])[0],
    'Conflicts_Over_Social_Media': conflicts,
}])

# Prediction
if st.button("Prediksi Skor Kecanduan"):
    pred = model.predict(input_data)[0]
    st.subheader(f"🎯 Prediksi Skor Kecanduan Anda: {pred:.2f} dari 10")
