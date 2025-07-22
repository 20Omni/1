import streamlit as st
import pandas as pd
import joblib
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_model = joblib.load("priority_xgboost.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

# Initialize session state for workload
if "user_workload" not in st.session_state:
    st.session_state.user_workload = {}

# Title
st.title("📋 AI Task Assignment Dashboard")

# Sample list of users
users = ["Alice", "Bob", "Charlie", "David"]

# Task input
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign Task")

if submitted:
    # Vectorize text
    task_vector = vectorizer.transform([task_desc])
    
    # Predict priority
    pred_encoded = priority_model.predict(task_vector)[0]
    pred_priority = label_encoder.inverse_transform([pred_encoded])[0]

    # Calculate deadline urgency score
    today = datetime.date.today()
    days_left = (deadline - today).days
    deadline_score = max(0, 10 - days_left)  # Closer deadline → higher score

    # Assign to user with minimum (load + urgency penalty)
    user_scores = []
    for user in users:
        load = st.session_state.user_workload.get(user, 0)
        score = load + deadline_score  # Higher deadline_score increases urgency
        user_scores.append((user, score))

    # Assign to user with lowest score
    assigned_user = sorted(user_scores, key=lambda x: x[1])[0][0]

    # Update workload
    st.session_state.user_workload[assigned_user] = st.session_state.user_workload.get(assigned_user, 0) + 1

    # Display result
    st.success(f"✅ Task assigned to **{assigned_user}** with predicted priority: **{pred_priority}**")
    st.info(f"🕐 Days until deadline: {days_left}")
    st.write("📊 Current Workload:")
    st.write(pd.DataFrame.from_dict(st.session_state.user_workload, orient="index", columns=["Tasks Assigned"]))

# Reset button
if st.button("🔁 Reset Workload"):
    st.session_state.user_workload = {}
    st.success("Workload reset!")