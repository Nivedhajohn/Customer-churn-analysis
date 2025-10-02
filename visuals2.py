# app/app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------
# Load Model
# -------------------------
model_path = os.path.join("models", "churn_model.pkl")
model = joblib.load(model_path)

# -------------------------
# Load Data
# -------------------------
data_path = os.path.join("data", "churn_dataset.csv")
df = pd.read_csv(data_path)

# -------------------------
# Streamlit App Layout
# -------------------------
st.title("📊 Customer Churn Prediction & Analysis")

menu = ["Prediction", "Data Insights"]
choice = st.sidebar.selectbox("Menu", menu)

# -------------------------
# Page 1: Prediction
# -------------------------
if choice == "Prediction":
    st.subheader("🔮 Predict Customer Churn")

    # Example input fields
    gender = st.selectbox("Gender", df["gender"].unique())
    tenure = st.slider("Tenure (Months)", 0, int(df["tenure"].max()))
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)

    # Convert input to DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges]
    })

    # One-hot encoding or same preprocessing as training
    # For simplicity, assume preprocessing is applied the same way here

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        churn_label = "⚠️ High Risk of Churn" if prediction == 1 else "✅ Low Risk of Churn"
        st.success(churn_label)

# -------------------------
# Page 2: Data Insights (Graphs)
# -------------------------
elif choice == "Data Insights":
    st.subheader("📈 Data Insights & Visualizations")

    # Graph 1: Churn Distribution Pie Chart
    st.markdown("### Churn Distribution")
    fig1, ax1 = plt.subplots()
    df["Churn"].value_counts().plot.pie(
        autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"], ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # Graph 2: Contract vs Churn
    if "Contract" in df.columns:
        st.markdown("### Contract Type vs Churn")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax2)
        st.pyplot(fig2)

    # Graph 3: Monthly Charges Boxplot
    if "MonthlyCharges" in df.columns:
        st.markdown("### Monthly Charges vs Churn")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3", ax=ax3)
        st.pyplot(fig3)

    # Graph 4: Tenure Histogram
    if "tenure" in df.columns:
        st.markdown("### Tenure Distribution by Churn")
        fig4, ax4 = plt.subplots()
        sns.histplot(df, x="tenure", hue="Churn", bins=30, kde=True, palette="husl", ax=ax4)
        st.pyplot(fig4)

    # Graph 5: Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax5)
    st.pyplot(fig5)
