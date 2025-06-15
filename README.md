📄 README.md — Churn Prediction & Dashboard
# 🧠 DS-2: Stop the Churn — Churn Prediction Tool

This project is built for a hackathon focused on **predicting customer churn** in the telecom/fintech domain. It uses **XGBoost** for classification and features an interactive **dashboard** for visualizing churn risk.

---

## 🚀 Objective

Every lost user costs revenue. This project:
- Predicts 30-day churn risk from customer behavior data
- Visualizes risk through a clear, interactive dashboard

---

## 🧰 Features

### ✅ Core Functionality
- Accepts user data CSV (with or without churn label)
- Handles missing values and encodes features
- Trains an XGBoost model (AUC-ROC optimized)
- Predicts churn probability for each user

### 📊 Visual Output
- AUC-ROC Curve
- Churn Probability Distribution (Histogram)
- Churn vs Retain Pie Chart
- Top-10 High-Risk Users Table
- Downloadable CSV with predictions

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/churn-predictor.git
cd churn-predictor
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run Locally (Streamlit)
bash
Copy
Edit
streamlit run app.py
📁 File Structure
bash
Copy
Edit
churn-predictor/
│
├── app.py                # Streamlit dashboard
├── model.py              # Model training and AUC evaluation
├── utils.py              # Data loading and preprocessing
├── requirements.txt      # All necessary libraries
└── README.md             # Project documentation
📊 Model Details
Classifier: XGBoost

Metric: AUC-ROC

Handling: Label encoding + median imputation for missing values

Frameworks: pandas, scikit-learn, xgboost, streamlit, seaborn

📤 Output Example
Each row in the output CSV contains:

customerID

churn_probability

🧠 Nice-to-Haves (Optional Enhancements)
SHAP feature explainability

REST API for real-time prediction

Mobile-friendly UI / Dark mode

External behavioral data support

🏁 Results
Achieved strong AUC-ROC performance on validation set

Dashboard highlights actionable churn risk segments

👨‍💻 Author
Made with ❤️ during a hackathon by SHUBHAM



[Here is the Explanation vidio of the code](https://drive.google.com/file/d/193_lhNW-_gpSGF-Cah6fT4m6JVTfzsoy/view?usp=sharing)


