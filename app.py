import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import re
from io import StringIO
import docx2txt
import PyPDF2

st.set_page_config(page_title="Employee Prediction App", layout="wide")
st.title("🧠 Employee Attrition & Performance Prediction")
st.markdown("Upload your HR dataset to predict attrition and performance ratings.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col])

    y_class = df["Attrition"]
    y_reg = df["PerformanceRating"]
    X = df.drop(columns=["Attrition", "PerformanceRating"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    classifier.fit(X_train_c, y_train_c)
    y_pred_c = classifier.predict(X_test_c)

    regressor = XGBRegressor()
    regressor.fit(X_train_r, y_train_r)
    y_pred_r = regressor.predict(X_test_r)

    st.subheader("📈 Attrition Prediction Report")
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c, zero_division=1))

    cm = confusion_matrix(y_test_c, y_pred_c)
    st.text("Confusion Matrix:")
    st.text(cm)

    st.subheader("📊 Performance Prediction Report")
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    st.subheader("🔥 Top 10 Features Affecting Attrition")
    feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
    top_feat = feat_importances.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    top_feat.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance (Attrition)")
    st.pyplot(fig)

    st.success("✅ Model training and evaluation completed!")

    # 🚀 Manual Prediction Section
    st.header("📌 Manual Prediction (Custom Input)")
    with st.form("manual_form"):
        input_data = {}
        for col in X.columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            val = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=float(df[col].mean()))
            input_data[col] = val
        submitted = st.form_submit_button("Predict Now")
        if submitted:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_attrition = classifier.predict(input_scaled)[0]
            pred_performance = regressor.predict(input_scaled)[0]
            st.write(f"🧾 Predicted Attrition: {'Yes' if pred_attrition == 1 else 'No'}")
            st.write(f"⭐ Predicted Performance Rating: {pred_performance:.2f}")

    # 📄 Resume Upload & Analysis Section
    st.header("📎 Resume Upload for Role & Skill Match")
    resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TEX)", type=["pdf", "docx", "tex"])

    def extract_text(file):
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        elif file.name.endswith(".tex"):
            return StringIO(file.getvalue().decode("utf-8")).read()
        return ""

    if resume_file is not None:
        resume_text = extract_text(resume_file)
        if resume_text:
            st.subheader("📝 Extracted Resume Text")
            st.text_area("Resume Preview", resume_text[:3000], height=200)

            # Keyword analysis
            keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
            feature_names = set(X.columns.str.lower())
            matched_features = keywords & feature_names

            st.write("🔍 Matched Features from Resume:")
            st.write(", ".join(matched_features) if matched_features else "No features matched.")

            matched_ratio = len(matched_features) / len(feature_names)
            if matched_ratio >= 0.5:
                st.success("✅ Resume aligns well with dataset features and roles.")
            elif 0.2 <= matched_ratio < 0.5:
                st.warning("⚠️ Partial match found. Consider adding more relevant experience or skills.")
            else:
                st.error("❌ Resume does not align well with dataset roles. Please revise.")

else:
    st.info("Upload a CSV file to get started.")
