import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model
from utils import load_data, preprocess_data

st.title("🧠 Churn Prediction Dashboard")

uploaded = st.file_uploader("Upload your CSV", type="csv")
if uploaded:
    df = load_data(uploaded)
    df_proc = preprocess_data(df)

    if 'churn' in df_proc.columns:
        model = train_model(df_proc)
        df_proc['churn_prob'] = model.predict_proba(df_proc.drop('churn', axis=1))[:, 1]
    else:
        st.error("No 'churn' column found. Upload training data.")

    st.subheader("📈 Churn Probability Distribution")
    sns.histplot(df_proc['churn_prob'], kde=True)
    st.pyplot()

    st.subheader("🧮 Churn vs Retain Pie Chart")
    churned = (df_proc['churn_prob'] > 0.5).sum()
    retained = len(df_proc) - churned
    plt.pie([churned, retained], labels=["Churn", "Retain"], autopct="%1.1f%%", colors=["red", "green"])
    st.pyplot()

    st.subheader("⚠️ Top 10 At-Risk Users")
    st.dataframe(df_proc.sort_values("churn_prob", ascending=False).head(10))

    st.subheader("⬇️ Download Predictions")
    st.download_button("Download CSV", df_proc.to_csv(index=False), "predictions.csv", "text/csv")