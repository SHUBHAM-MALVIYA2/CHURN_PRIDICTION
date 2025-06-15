import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Assume 'Churn' is the target (binary: Yes/No or 1/0)
    target = 'Churn'
    if df[target].dtype == object:
        df[target] = df[target].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    # Identify features
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, numeric_features, categorical_features

# Train model and compute predictions
def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build pipeline with XGBoost
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC-ROC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return model, X_test, y_test, y_pred_proba, auc

# SHAP explainability
def compute_shap_values(model, X_test):
    preprocessor = model.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_transformed)
    return shap_values, X_test_transformed

# Streamlit Dashboard
def main():
    st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
    
    st.title("📊 Churn Prediction Dashboard")
    st.write("Upload a CSV file to predict customer churn and visualize results.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file (e.g., telco_train.csv)", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess data
        X, y, preprocessor, numeric_features, categorical_features = load_and_preprocess_data(uploaded_file)
        
        # Train model
        model, X_test, y_test, y_pred_proba, auc = train_model(X, y, preprocessor)
        
        st.subheader(f"Model Performance: AUC-ROC = {auc:.3f}")
        
        # Churn probability distribution
        st.subheader("Churn Probability Distribution")
        fig_dist = px.histogram(x=y_pred_proba, nbins=50, title="Distribution of Churn Probabilities",
                               labels={'x': 'Churn Probability'}, template="plotly_dark")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Churn vs Retain Pie Chart
        st.subheader("Churn vs Retain")
        churn_counts = pd.Series(y_pred_proba > 0.5).value_counts()
        fig_pie = go.Figure(data=[
            go.Pie(labels=['Retain', 'Churn'], values=churn_counts, hole=0.4)
        ])
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top-10 Risk Table
        st.subheader("Top 10 At-Risk Customers")
        risk_df = pd.DataFrame({
            'Customer Index': X_test.index,
            'Churn Probability': y_pred_proba
        }).sort_values(by='Churn Probability', ascending=False).head(10)
        st.dataframe(risk_df, use_container_width=True)
        
        # Download predictions
        st.subheader("Download Predictions")
        predictions_df = pd.DataFrame({
            'Customer Index': X_test.index,
            'Churn Probability': y_pred_proba
        })
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
        
        # SHAP Explainability
        st.subheader("Model Explainability (SHAP)")
        shap_values, X_test_transformed = compute_shap_values(model, X_test)
        st_shap(shap.summary_plot(shap_values, X_test_transformed, show=False), height=400)
        
if __name__ == "__main__":
    main()