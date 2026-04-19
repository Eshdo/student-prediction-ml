import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Student Predictor", page_icon="🎓", layout="centered")

# Title
st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will **Pass or Fail** based on performance.")

# Load dataset
data = pd.read_csv("student_data_balanced.csv")

# Clean target column
data['final_result'] = data['final_result'].str.strip().str.lower()
data['final_result'] = data['final_result'].map({'fail': 0, 'pass': 1})

# Features & target
X = data.drop('final_result', axis=1)
y = data['final_result']

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X, y)

# --- INPUT SECTION ---
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("📘 Hours Studied", 1, 10, 5)
    attendance = st.slider("📅 Attendance (%)", 40, 100, 70)

with col2:
    assignment = st.slider("📝 Assignment Score", 30, 100, 60)
    previous = st.slider("📊 Previous Marks", 30, 100, 65)

# --- PREDICTION ---
if st.button("🔍 Predict Result"):

    input_data = [[hours, attendance, assignment, previous]]
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
        st.metric("Confidence", f"{probability[0][1]*100:.2f}%")
    else:
        st.error("❌ Student is likely to FAIL")
        st.metric("Confidence", f"{probability[0][0]*100:.2f}%")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")
