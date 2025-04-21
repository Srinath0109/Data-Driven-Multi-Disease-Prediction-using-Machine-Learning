
import streamlit as st
import numpy as np
import pickle

# Load models and scalers
model_d = pickle.load(open('model_diabetes.pkl', 'rb'))
scaler_d = pickle.load(open('scaler_diabetes.pkl', 'rb'))

model_h = pickle.load(open('model_heart.pkl', 'rb'))
scaler_h = pickle.load(open('scaler_heart.pkl', 'rb'))

model_p = pickle.load(open('model_parkinsons.pkl', 'rb'))
scaler_p = pickle.load(open('scaler_parkinsons.pkl', 'rb'))

st.title("🧠 Multi-Disease Prediction System")
st.markdown("Select a disease and input values to check the risk prediction.")

disease = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinson's"])

if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 0, 120)

    if st.button("Predict Diabetes"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler_d.transform(data)
        result = model_d.predict(scaled)
        st.success("🟢 Not likely to have Diabetes." if result[0] == 0 else "🔴 Likely to have Diabetes.")

elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 0, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    rbp = st.number_input("Resting Blood Pressure", 0, 200)
    chol = st.number_input("Cholesterol", 0, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 0, 250)
    ex_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Encode manually as done during training
    sex_val = 1 if sex == "Male" else 0
    chest_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    restecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    angina_val = 1 if ex_angina == "Yes" else 0
    slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    fbs_val = 1 if fbs == "Yes" else 0

    heart_data = np.array([[age, sex_val, chest_map[chest], rbp, chol, fbs_val,
                            restecg_map[rest_ecg], max_hr, angina_val, oldpeak, slope_map[st_slope]]])
    scaled = scaler_h.transform(heart_data)
    result = model_h.predict(scaled)
    st.success("🟢 No Heart Disease detected." if result[0] == 0 else "🔴 Likely to have Heart Disease.")

else:
    st.header("🧠 Parkinson's Prediction")

    st.write("Enter the following voice-related biomedical parameters:")

    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
                'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    values = [st.number_input(f"{f}") for f in features]
    parkinson_data = np.array([values])
    scaled = scaler_p.transform(parkinson_data)
    result = model_p.predict(scaled)

    st.success("🟢 No Parkinson's detected." if result[0] == 0 else "🔴 Likely to have Parkinson's.")
