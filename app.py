
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="OGW Prediction App", layout="wide")

# ------------------------
# LOAD MODELS AND SCALERS
# ------------------------
@st.cache_resource
def load_models():
    models = {
        "Random Forest": pickle.load(open("rfr_model.pkl", "rb")),
        #"Linear Regression": pickle.load(open("lr_model.pkl", "rb")),
        #"Decision Tree": pickle.load(open("dt_model.pkl", "rb"))
    }
    x_scaler = pickle.load(open("x_scaler.pkl", "rb"))
    y_scaler = pickle.load(open("y_scaler.pkl", "rb"))
    return models, x_scaler, y_scaler

# ------------------------
# INPUT FORM
# ------------------------
def get_sidebar_input():
    st.sidebar.markdown("## 🧮 Real-Time Prediction Inputs")

    well_code = st.sidebar.text_input("N_WELL_BORE_CODE")
    well_name = st.sidebar.selectbox("Well Name", ["Well-1", "Well-2", "Well-3", "Well-4", "Well-5"])
    well_type = st.sidebar.selectbox("Well Type", ["WI", "OP"])
    dp = st.sidebar.number_input("Downhole Pressure (PSI)", min_value=0.0)
    dt = st.sidebar.number_input("Downhole Temperature (Kelvin)", min_value=200.0)
    avg_tp = st.sidebar.number_input("Average Tubing Pressure", min_value=0.0)
    annulus_p = st.sidebar.number_input("Annulus Pressure (PSI)", min_value=0.0)
    whp = st.sidebar.number_input("AVG WHP (PSI)", min_value=0.0)
    choke = st.sidebar.number_input("Choke Size", min_value=0.0)

    submitted = st.sidebar.button("🚀 Predict")

    if submitted:
        input_df = pd.DataFrame([[well_code, well_name, well_type, dp, dt, avg_tp, annulus_p, whp, choke]],
                                columns=["N_WELL_BORE_CODE", "Well Name", "Well Type", "Downhole Pressure (PSI)",
                                         "Downhole Temperature (Kelvin)", "Average Tubing Pressure",
                                         "Annulus Pressure (PSI)", "AVG WHP (PSI)", "Choke Size"])
        return input_df
    return None

# ------------------------
# LOAD DATA
# ------------------------
def load_data():
    uploaded_file = st.sidebar.file_uploader("📂 Upload Production CSV File", type=["csv"], key="file_uploader")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'PRODUCTION DATE' in df.columns:
            df['PRODUCTION DATE'] = pd.to_datetime(df['PRODUCTION DATE'], errors='coerce')
            df.dropna(subset=['PRODUCTION DATE'], inplace=True)
            df['year'] = df['PRODUCTION DATE'].dt.year
            df['month'] = df['PRODUCTION DATE'].dt.month
            df['day'] = df['PRODUCTION DATE'].dt.day
        return df
    return None

# ------------------------
# FEATURE VISUALIZATION
# ------------------------
def visualize_data(df):
    st.subheader("📊 Feature-wise Visualizations")
    plot_type = st.selectbox("Choose plot type", ["Histogram", "Line Plot", "Scatter", "Correlation Heatmap"])

    if plot_type in ["Histogram", "Line Plot", "Scatter"]:
        col_x = st.selectbox("X-axis", df.columns)
        col_y = None
        if plot_type in ["Line Plot", "Scatter"]:
            col_y = st.selectbox("Y-axis", df.columns)

        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_x, nbins=30, title=f"Distribution of {col_x}")
        elif plot_type == "Line Plot":
            fig = px.line(df, x=col_x, y=col_y, title=f"{col_y} over {col_x}")
        elif plot_type == "Scatter":
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ------------------------
# MAKE PREDICTION
# ------------------------
def predict_production(model, x_scaler, y_scaler, input_df):
    input_df_scaled = x_scaler.transform(input_df.select_dtypes(include=[np.number]))
    prediction_scaled = model.predict(input_df_scaled)
    prediction = y_scaler.inverse_transform([prediction_scaled[0]])
    return prediction[0]


# ------------------------
# PREDICT CSV AND PLOT
# ------------------------
def predict_csv_data(df, model, x_scaler, y_scaler):
    if df is None or model is None:
        st.warning("CSV file or model not available.")
        return

    # One-hot encode
    df_encoded = pd.get_dummies(df.copy())
    missing_cols = set(x_scaler.feature_names_in_) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[x_scaler.feature_names_in_]

    # Predict
    scaled_features = x_scaler.transform(df_encoded)
    preds_scaled = model.predict(scaled_features)
    preds = y_scaler.inverse_transform(preds_scaled)

    # Display dataframe
    pred_df = df.copy()
    pred_df['Predicted Oil'] = preds[:, 0]
    pred_df['Predicted Gas'] = preds[:, 1]
    pred_df['Predicted Water'] = preds[:, 2]

    st.subheader("📈 Predicted Values from CSV")
    st.dataframe(pred_df.head())

    if 'PRODUCTION DATE' in pred_df.columns:
        fig_oil = px.line(pred_df, x='PRODUCTION DATE', y='Predicted Oil', title="Predicted Oil Over Time")
        fig_gas = px.line(pred_df, x='PRODUCTION DATE', y='Predicted Gas', title="Predicted Gas Over Time")
        fig_water = px.line(pred_df, x='PRODUCTION DATE', y='Predicted Water', title="Predicted Water Over Time")

        st.plotly_chart(fig_oil, use_container_width=True)
        st.plotly_chart(fig_gas, use_container_width=True)
        st.plotly_chart(fig_water, use_container_width=True)


# ------------------------
# MAIN UI
# ------------------------
def main():
    st.title("🛢️ OGW Production Prediction Dashboard")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Visualize Data", "Predict"])

    df = load_data()

    if page == "Visualize Data":
        if df is not None:
            visualize_data(df)
        else:
            st.warning("Upload a CSV file to visualize data.")

    # KEEP EVERYTHING ELSE ABOVE UNCHANGED...

    elif page == "Predict":
        models, x_scaler, y_scaler = load_models()
        model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
        model = models[model_name]

        # Predict for CSV
        predict_csv_data(df, model, x_scaler, y_scaler)

        # Predict for Real-Time Input
        input_df = get_sidebar_input()
        if input_df is not None:
            input_encoded = pd.get_dummies(input_df.copy())
            missing_cols = set(x_scaler.feature_names_in_) - set(input_encoded.columns)
            for col in missing_cols:
                input_encoded[col] = 0
            input_encoded = input_encoded[x_scaler.feature_names_in_]

            input_scaled = x_scaler.transform(input_encoded)
            prediction_scaled = model.predict(input_scaled)
            prediction = y_scaler.inverse_transform([prediction_scaled[0]])[0]

            # SHAP
            explainer = shap.Explainer(model, input_encoded)
            shap_values = explainer(input_encoded)

            st.subheader("📌 SHAP Explanation")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(bbox_inches='tight', dpi=300)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Oil Production (stb/day)", f"{prediction[0]:.2f}")
            col2.metric("Gas Volume (scf/day)", f"{prediction[1]:.2f}")
            col3.metric("Water Production (stb/day)", f"{prediction[2]:.2f}")


if __name__ == "__main__":
    main()
