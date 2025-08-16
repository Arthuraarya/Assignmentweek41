import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("bank_churn_data.csv")
    df['churn'] = (df['attrition_flag'].str.lower().str.contains('attrited')).astype(int)
    return df

df = load_data()

# =====================
# SIDEBAR MENU
# =====================
st.sidebar.title("📊 Bank Churn Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["EDA", "Modeling", "Interpretasi"])

# =====================
# EDA PAGE
# =====================
if page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    # Pilih variable untuk ditampilkan
    col_choice = st.selectbox("Pilih variabel untuk histogram:", df.select_dtypes(include=np.number).columns)

    fig = px.histogram(df, x=col_choice, color="churn", barmode="overlay",
                       nbins=20, title=f"Distribusi {col_choice} by Churn")
    st.plotly_chart(fig, use_container_width=True)

    # Churn rate berdasarkan kategori (opsional)
    cat_choice = st.selectbox("Pilih variabel kategorikal:", df.select_dtypes(include=['object']).columns)
    churn_rate = df.groupby(cat_choice)['churn'].mean().reset_index()
    fig = px.bar(churn_rate, x=cat_choice, y="churn",
                 title=f"Churn Rate berdasarkan {cat_choice}")
    st.plotly_chart(fig, use_container_width=True)

# =====================
# MODELING PAGE
# =====================
elif page == "Modeling":
    st.title("🤖 Modeling Churn Prediction")

    # Split data
    X = df.drop(columns=['user_id','attrition_flag','churn','age_bin'], errors="ignore")
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Preprocessor
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # Pilih model
    model_choice = st.sidebar.selectbox("Pilih Model:", ["Logistic Regression", "Random Forest"])
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    clf = Pipeline([('pre', preprocessor), ('model', model)])
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:,1]

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_score),
        "PR AUC": average_precision_score(y_test, y_score)
    }
    st.subheader("📊 Evaluation Metrics")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0:"Score"}))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={metrics['ROC AUC']:.2f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Chance"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR AUC={metrics['PR AUC']:.2f}"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

# =====================
# INTERPRETATION PAGE
# =====================
elif page == "Interpretasi":
    st.title("🔍 Model Interpretability")

    # Train ulang untuk interpretasi
    X = df.drop(columns=['user_id','attrition_flag','churn','age_bin'], errors="ignore")
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    clf = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000, class_weight="balanced"))])
    clf.fit(X_train, y_train)

    # Feature Importance
    perm = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm.importances_mean
    }).sort_values('importance', ascending=False)

    n_feats = st.slider("Pilih jumlah fitur teratas:", 5, 20, 10)
    st.write(f"Top {n_feats} Feature Importance:")
    fig = px.bar(perm_df.head(n_feats), x="importance", y="feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(perm_df.head(n_feats))
