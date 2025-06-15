from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_model(df):
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.4f}")
    return model