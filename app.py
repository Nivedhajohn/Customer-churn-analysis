# app/app.py
"""
Streamlit app for Customer Churn:
- Batch CSV prediction
- Single record prediction
- Data Insights (graphs & charts)
Run:
    streamlit run app/app.py
"""

import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Settings
# -------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
data_path = os.path.join(project_root, 'data', 'customer churn data.csv')

# -------------------------
# Load Model
# -------------------------
if not os.path.exists(model_path):
    st.error("Model not found. Run: python src/train_model.py")
    st.stop()

pipe = joblib.load(model_path)
st.success("Loaded model successfully.")

# -------------------------
# Sidebar Menu
# -------------------------
menu = ["Prediction", "Data Insights"]
choice = st.sidebar.selectbox("Menu", menu)

# -------------------------
# PAGE 1: Prediction
# -------------------------
if choice == "Prediction":
    st.title("📊 Customer Churn Prediction")
    st.write("Upload CSV or fill the form below for single predictions.")

    # Sidebar CSV upload
    st.sidebar.header("Upload CSV for batch prediction")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        if st.sidebar.button("Run batch prediction"):
            try:
                proba = pipe.predict_proba(df)[:, 1]
                preds = pipe.predict(df)
                df_out = df.copy()
                df_out['churn_probability'] = proba
                df_out['churn_label'] = ['Churn' if p == 1 else 'Not Churn' for p in preds]
                st.success("Prediction complete — preview below:")
                st.dataframe(df_out.head())
                st.download_button("Download predictions CSV", df_out.to_csv(index=False),
                                   file_name="predictions.csv")
            except Exception as e:
                st.error(f"Prediction failed. Make sure columns match training set. Error: {e}")

    st.markdown("---")
    st.header("Single record prediction (manual input)")

    # Detect expected columns from preprocessor
    preprocessor = pipe.named_steps.get('preprocessor', None)
    expected_cols = []
    if preprocessor is not None:
        for trans in preprocessor.transformers_:
            if len(trans) >= 3 and isinstance(trans[2], (list, tuple)):
                expected_cols.extend(list(trans[2]))

    if expected_cols:
        st.write(f"Detected columns: {expected_cols}")
        with st.form("single_form"):
            input_data = {}
            for c in expected_cols:
                if any(k in c.lower() for k in ['num', 'amount', 'charges', 'monthly', 'tenure', 'count', 'age', 'balance']):
                    val = st.number_input(c, value=0.0, format="%.4f")
                else:
                    val = st.text_input(c, value="")
                input_data[c] = val
            submitted = st.form_submit_button("Predict single")
            if submitted:
                df_single = pd.DataFrame([input_data])
                for col in df_single.columns:
                    try:
                        df_single[col] = pd.to_numeric(df_single[col], errors='ignore')
                    except Exception:
                        pass
                try:
                    prob = pipe.predict_proba(df_single)[:, 1][0]
                    pred = pipe.predict(df_single)[0]
                    st.write("Churn probability:", float(prob))
                    st.write("Prediction:", "⚠️ Churn" if pred == 1 else "✅ Not Churn")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("App couldn't detect columns from the preprocessor. Use CSV upload instead.")

# -------------------------
# PAGE 2: Data Insights
# -------------------------
elif choice == "Data Insights":
    st.title("📈 Customer Churn Data Insights")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Churn Distribution Pie
        st.markdown("### Churn Distribution (Pie Chart)")
        fig1, ax1 = plt.subplots()
        df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"], ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        # Churn Distribution Bar
        st.markdown("### Churn Distribution (Bar Chart)")
        fig1b, ax1b = plt.subplots()
        sns.countplot(x="Churn", data=df, palette="Set2", ax=ax1b)
        st.pyplot(fig1b)

        # Contract vs Churn
        if "Contract" in df.columns:
            st.markdown("### Contract Type vs Churn")
            fig2, ax2 = plt.subplots()
            sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax2)
            st.pyplot(fig2)

        # Monthly Charges vs Churn
        if "MonthlyCharges" in df.columns:
            st.markdown("### Monthly Charges vs Churn")
            fig3, ax3 = plt.subplots()
            sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3", ax=ax3)
            st.pyplot(fig3)

        # Tenure Distribution
        if "tenure" in df.columns:
            st.markdown("### Tenure Distribution by Churn")
            fig4, ax4 = plt.subplots()
            sns.histplot(df, x="tenure", hue="Churn", bins=30, kde=True, palette="husl", ax=ax4)
            st.pyplot(fig4)

        # Correlation Heatmap
        st.markdown("### Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax5)
        st.pyplot(fig5)
    else:
        st.error("Dataset not found for visualizations. Place your CSV in the 'data' folder.")
