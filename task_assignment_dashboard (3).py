
import streamlit as st
import pandas as pd
import joblib
from collections import defaultdict
import io

# Load models
rf_model = joblib.load("priority_random_forest.pkl")
xgb_model = joblib.load("priority_xgboost.pkl")
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

st.set_page_config(page_title="AI Task Manager", layout="centered")

st.title("📌 AI Task Assignment Dashboard")

st.sidebar.title("⚙️ Model & Users")
model_choice = st.sidebar.selectbox("Choose your trained model:", ["Random Forest", "XGBoost"])

user_input = st.sidebar.text_input("Enter user names (comma-separated)", "Alice,Bob,Charlie")
user_list = [u.strip() for u in user_input.split(",") if u.strip()]

uploaded_file = st.file_uploader("📤 Upload your task CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "Task" not in df.columns:
            st.error("❗Column 'Task' is required in your CSV.")
        else:
            st.subheader("✅ Uploaded Task Data")
            st.dataframe(df)

            # Vectorize task text
            X = vectorizer.transform(df["Task"].astype(str))

            # Predict
            if model_choice == "Random Forest":
                preds = rf_model.predict(X)
            else:
                preds = xgb_model.predict(X)

            df["Predicted Priority"] = label_encoder.inverse_transform(preds)

            # Task Assignment Logic
            task_counts = defaultdict(int)
            assigned_users = []

            # Sort tasks: High → Medium → Low
            priority_order = ["High", "Medium", "Low"]
            df["Priority Level"] = df["Predicted Priority"].apply(lambda x: priority_order.index(x))
            df = df.sort_values("Priority Level")

            for _, row in df.iterrows():
                min_user = min(user_list, key=lambda u: task_counts[u])
                assigned_users.append(min_user)
                task_counts[min_user] += 1

            df["Assigned User"] = assigned_users
            df.drop(columns="Priority Level", inplace=True)

            st.success("🎯 Tasks assigned successfully!")

            st.subheader("📊 Final Task Assignment")
            st.dataframe(df)

            # Download
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Assigned Tasks CSV", data=csv_data, file_name="assigned_tasks.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
