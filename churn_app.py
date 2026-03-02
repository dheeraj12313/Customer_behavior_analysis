import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Retention Prediction", layout="wide")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Employee Retention Prediction")
st.markdown("Predict whether a data scientist is likely to seek a job change.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/aug_train.csv")

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
)

use_smote = st.sidebar.checkbox("Apply SMOTE (Handle Class Imbalance)", value=True)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
df_proc = df.copy()

target = "target"
X = df_proc.drop(columns=[target])
y = df_proc[target]

# Encode categorical variables
categorical_cols = X.select_dtypes(include="object").columns
le = LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Apply SMOTE
if use_smote:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_name == "XGBoost":
    model = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
else:
    model = LGBMClassifier(random_state=42)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
st.subheader("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1-Score", f"{f1:.2f}")
col5.metric("ROC-AUC", f"{roc:.2f}")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
if model_name != "Logistic Regression":
    st.subheader("⭐ Feature Importance")

    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# INDIVIDUAL PREDICTION
# --------------------------------------------------
st.subheader("🧑‍💼 Individual Employee Prediction")

with st.form("prediction_form"):
    input_data = {}

    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data[col] = val

    submit = st.form_submit_button("Predict Job Change Risk")

if submit:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 🔮 Prediction Result")
    st.metric("Job Change Probability", f"{prob:.2%}")

    if prob > 0.7:
        st.error("⚠️ High Risk of Job Change")
    elif prob > 0.4:
        st.warning("⚠️ Medium Risk of Job Change")
    else:
        st.success("✅ Low Risk of Job Change")