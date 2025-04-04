import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Prediction App", layout="wide")
st.title("🧠 Employee Attrition & Performance Prediction")
st.markdown("Upload your HR dataset to predict attrition and performance ratings.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    # Drop irrelevant columns
    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Label Encoding for categorical features
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col])

    # Separate Targets
    y_class = df["Attrition"]
    y_reg = df["PerformanceRating"]
    X = df.drop(columns=["Attrition", "PerformanceRating"])

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    # Train Classifier
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    classifier.fit(X_train_c, y_train_c)
    y_pred_c = classifier.predict(X_test_c)

    # Train Regressor
    regressor = XGBRegressor()
    regressor.fit(X_train_r, y_train_r)
    y_pred_r = regressor.predict(X_test_r)

    # Evaluation: Classification
    st.subheader("📈 Attrition Prediction Report")
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c, zero_division=1))

    cm = confusion_matrix(y_test_c, y_pred_c)
    st.text("Confusion Matrix:")
    st.text(cm)

    # Evaluation: Regression
    st.subheader("📊 Performance Prediction Report")
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Feature Importances
    st.subheader("🔥 Top 10 Features Affecting Attrition")
    feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
    top_feat = feat_importances.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top_feat.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance (Attrition)")
    st.pyplot(fig)

    st.success("✅ Model training and evaluation completed!")
else:
    st.info("Upload a CSV file to get started.")
