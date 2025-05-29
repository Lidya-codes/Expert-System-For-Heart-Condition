# app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup ---
st.set_page_config(page_title="Heart Condition Detector", layout="centered")
st.title("💓 Expert System for Heart Condition Detection")
st.write("Detect abnormal heart conditions using R-R intervals and machine learning.")

# --- Step 1: Generate and Train ---
@st.cache_resource
def train_model():
    def generate_rr_intervals(num_samples=500, normal_rate=70, abnormal_rate=120):
        normal_rr_interval = 60 / normal_rate
        abnormal_rr_interval = 60 / abnormal_rate
        rr_intervals = []
        labels = []

        for _ in range(num_samples):
            is_abnormal = np.random.rand() > 0.7  # 30% abnormal
            if is_abnormal:
                rr = abnormal_rr_interval + np.random.uniform(-0.2, 0.2)
                labels.append(1)
            else:
                rr = normal_rr_interval + np.random.uniform(-0.1, 0.1)
                labels.append(0)
            rr_intervals.append(rr)

        df = pd.DataFrame({
            'R-R Interval': rr_intervals,
            'Label': labels
        })
        return df

    df = generate_rr_intervals()
    X = df[['R-R Interval']]
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    return model, scaler, df

model, scaler, df = train_model()

# --- Step 2: Visualization ---
with st.expander("📊 See Sample Data and Distribution"):
    st.write(df.head())
    fig, ax = plt.subplots()
    sns.histplot(df, x='R-R Interval', hue='Label', kde=True, bins=30, ax=ax)
    st.pyplot(fig)

# --- Step 3: User Input ---
st.subheader("🧪 Try a Prediction")

rr_input = st.number_input("Enter an R-R Interval (in seconds)", min_value=0.3, max_value=1.5, step=0.01, value=0.85)

if st.button("Predict"):
    input_scaled = scaler.transform([[rr_input]])
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Abnormal Heart Condition Detected")
    else:
        st.success("✅ Heart Condition is Normal")

    st.caption("This is based on your provided R-R interval.")


