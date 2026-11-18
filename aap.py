"""
OGW Production Prediction Dashboard
=====================================
Multi-target ensemble ML for simultaneous oil, gas, and water production forecasting.
Trains directly from the SPE DSEATS 2024 training dataset — no pre-saved .pkl files needed.

Models: Extra Trees | Random Forest | XGBoost | HistGradientBoosting | Linear Regression
Features: Real-time prediction, SHAP interpretability, validation forecasting, model comparison

Author : Group 4 — UENR Petroleum Engineering (Rebuilt for Journal Submission)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
TRAINING_DATA = "dseats_2024_training_dataset.csv"

FEATURE_COLS = [
    "N_WELL_BORE_CODE", "WELL_TYPE_encoded",
    "Downhole Pressure (PSI)", "Downhole Temperature (Kelvin)",
    "Average Tubing Pressure", "Annulus Pressure (PSI)",
    "AVG WHP (PSI)", "Choke Size",
    "year", "month", "day",
]

TARGET_COLS = [
    "Oil Production (stb/day)",
    "Gas Volume (scf/day)",
    "Water Production (stb/day)",
]

TARGET_UNITS = {"Oil": "stb/day", "Gas": "scf/day", "Water": "stb/day"}
TARGET_COLORS = {"Oil": "#D4760A", "Gas": "#E85D26", "Water": "#1C7ED6"}

st.set_page_config(
    page_title="OGW Prediction Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main container */
    .block-container { padding-top: 1.5rem; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #a8b2d1 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6f1ff !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Success/info boxes */
    .stAlert { border-radius: 10px; }
    
    /* Tables */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA & MODEL PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def clip_outliers_features(df, feature_cols):
    """IQR-based outlier clipping on input features only (never targets)."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


@st.cache_resource(show_spinner="Training models on DSEATS dataset...")
def train_pipeline(csv_path):
    """
    Full training pipeline: load → clean → engineer → scale → train all models.
    Cached so it only runs once per session.
    """
    # ── Load ──
    raw = pd.read_csv(csv_path)

    # ── Clean ──
    data = raw.dropna(subset=["Choke Size"]).copy()

    # ── Temporal features ──
    data["PRODUCTION DATE"] = pd.to_datetime(data["PRODUCTION DATE"], format="mixed")
    data["year"] = data["PRODUCTION DATE"].dt.year
    data["month"] = data["PRODUCTION DATE"].dt.month
    data["day"] = data["PRODUCTION DATE"].dt.day

    # ── Encode categoricals ──
    le = LabelEncoder()
    data["WELL_TYPE_encoded"] = le.fit_transform(data["WELL_TYPE"])

    # ── Separate X / Y ──
    X = data[FEATURE_COLS].copy()
    Y = data[TARGET_COLS].copy()

    # ── Clip outliers on features ONLY ──
    clip_cols = [
        "Downhole Pressure (PSI)", "Downhole Temperature (Kelvin)",
        "Average Tubing Pressure", "Annulus Pressure (PSI)",
        "AVG WHP (PSI)", "Choke Size",
    ]
    X = clip_outliers_features(X, clip_cols)

    # ── Scale ──
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)
    Y_scaled = y_scaler.fit_transform(Y)

    # ── Train/test split ──
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_scaled, Y_scaled, test_size=0.2, random_state=RANDOM_STATE
    )

    # ── Define models ──
    model_defs = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, max_depth=50, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=50, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=500, max_depth=10, learning_rate=0.05,
                min_samples_leaf=20, random_state=RANDOM_STATE,
            )
        ),
        "Linear Regression": LinearRegression(),
    }
    if HAS_XGB:
        model_defs["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            )
        )

    # ── Train & evaluate each model ──
    trained = {}
    metrics = {}
    for name, model in model_defs.items():
        t0 = time.time()
        model.fit(X_tr, Y_tr)
        train_time = time.time() - t0

        Y_pred = model.predict(X_te)

        # Aggregated metrics (on scaled data)
        agg = {
            "MAE": mean_absolute_error(Y_te, Y_pred),
            "MSE": mean_squared_error(Y_te, Y_pred),
            "RMSE": np.sqrt(mean_squared_error(Y_te, Y_pred)),
            "R2": r2_score(Y_te, Y_pred),
            "Train Time (s)": round(train_time, 2),
        }

        # Per-target metrics (original units)
        Y_te_orig = y_scaler.inverse_transform(Y_te)
        Y_pred_orig = y_scaler.inverse_transform(Y_pred)
        per_target = {}
        for j, tgt in enumerate(TARGET_COLS):
            short = tgt.split("(")[0].strip()
            per_target[short] = {
                "MAE": mean_absolute_error(Y_te_orig[:, j], Y_pred_orig[:, j]),
                "RMSE": np.sqrt(mean_squared_error(Y_te_orig[:, j], Y_pred_orig[:, j])),
                "R2": r2_score(Y_te_orig[:, j], Y_pred_orig[:, j]),
            }
        agg["per_target"] = per_target

        trained[name] = model
        metrics[name] = agg

    # ── Store test data for scatter plots ──
    test_data = {
        "X_te": X_te, "Y_te": Y_te,
        "Y_te_orig": y_scaler.inverse_transform(Y_te),
    }

    # ── Build predictions dict for each model on test set ──
    preds_cache = {}
    for name, model in trained.items():
        yp = model.predict(X_te)
        preds_cache[name] = y_scaler.inverse_transform(yp)

    return {
        "models": trained,
        "metrics": metrics,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "label_encoder": le,
        "training_data": data,
        "test_data": test_data,
        "preds_cache": preds_cache,
        "X_scaled_full": X_scaled,
        "Y_scaled_full": Y_scaled,
    }


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def format_number(val, unit=""):
    """Smart number formatting."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:,.2f}M {unit}"
    elif abs(val) >= 1_000:
        return f"{val/1_000:,.1f}K {unit}"
    else:
        return f"{val:,.1f} {unit}"


def preprocess_upload(df_raw, le):
    """Preprocess an uploaded CSV for prediction (mirrors training pipeline)."""
    df = df_raw.copy()

    # Parse date
    if "PRODUCTION DATE" in df.columns:
        df["PRODUCTION DATE"] = pd.to_datetime(df["PRODUCTION DATE"], errors="coerce")
        df["year"] = df["PRODUCTION DATE"].dt.year
        df["month"] = df["PRODUCTION DATE"].dt.month
        df["day"] = df["PRODUCTION DATE"].dt.day

    # Encode WELL_TYPE
    if "WELL_TYPE" in df.columns:
        known_classes = set(le.classes_)
        df["WELL_TYPE_encoded"] = df["WELL_TYPE"].apply(
            lambda x: le.transform([x])[0] if x in known_classes else 0
        )

    # Fill missing Choke Size with 0 for uploaded data (can't drop rows in live use)
    if "Choke Size" in df.columns:
        df["Choke Size"] = df["Choke Size"].fillna(0)

    # Select features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, missing

    X_input = df[FEATURE_COLS].copy()
    clip_cols = [
        "Downhole Pressure (PSI)", "Downhole Temperature (Kelvin)",
        "Average Tubing Pressure", "Annulus Pressure (PSI)",
        "AVG WHP (PSI)", "Choke Size",
    ]
    X_input = clip_outliers_features(X_input, clip_cols)
    return X_input, []


def make_scatter_plot(actual, predicted, target_name, color):
    """Create a clean actual vs predicted scatter plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers",
        marker=dict(color=color, size=4, opacity=0.4),
        name="Predictions",
    ))
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="white", dash="dash", width=1),
        name="Perfect",
    ))
    r2 = r2_score(actual, predicted)
    fig.update_layout(
        title=dict(text=f"{target_name}  ·  R² = {r2:.4f}", font=dict(size=14)),
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_dark",
        height=350,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE
# ═══════════════════════════════════════════════════════════════════════
# Resolve training data path
data_path = None
for candidate in [TRAINING_DATA, f"/mnt/user-data/uploads/{TRAINING_DATA}",
                  os.path.join(os.path.dirname(__file__), TRAINING_DATA) if "__file__" in dir() else ""]:
    if candidate and os.path.exists(candidate):
        data_path = candidate
        break

if data_path is None:
    st.error(f"Training dataset `{TRAINING_DATA}` not found. "
             f"Place it in the same directory as this script.")
    st.stop()

pipe = train_pipeline(data_path)
models = pipe["models"]
metrics = pipe["metrics"]
x_scaler = pipe["x_scaler"]
y_scaler = pipe["y_scaler"]
le = pipe["label_encoder"]
train_data = pipe["training_data"]
test_data = pipe["test_data"]
preds_cache = pipe["preds_cache"]


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
# 🛢️ OGW Production Prediction Dashboard
**Multi-target ensemble ML for oil, gas & water production forecasting**  
*Trained on SPE DSEATS 2024 Africa Regional Datathon dataset*
""")

# Model ranking bar
best_name = max(metrics, key=lambda k: metrics[k]["R2"])
st.success(
    f"🏆 **Best model: {best_name}** — R² = {metrics[best_name]['R2']:.4f} · "
    f"RMSE = {metrics[best_name]['RMSE']:.4f} · "
    f"Trained in {metrics[best_name]['Train Time (s)']}s"
)


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    model_choice = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=list(models.keys()).index(best_name),
        help="Choose which trained model to use for predictions",
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV for prediction",
        type=["csv"],
        help="Upload a CSV with the same feature columns as the training data",
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.caption(f"Training samples: **{len(train_data):,}**")
    st.caption(f"Features: **{len(FEATURE_COLS)}**")
    st.caption(f"Targets: **Oil · Gas · Water**")
    st.caption(f"Date range: **2008 — 2015**")
    st.caption(f"Models trained: **{len(models)}**")


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🔮 Predict from CSV",
    "⌨️ Manual Prediction",
    "🧠 SHAP Analysis",
    "📈 Data Explorer",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Model Comparison")

    # ── Metrics table ──
    rows = []
    for name in models:
        m = metrics[name]
        rows.append({
            "Model": name,
            "R²": f"{m['R2']:.4f}",
            "MAE": f"{m['MAE']:.4f}",
            "RMSE": f"{m['RMSE']:.4f}",
            "MSE": f"{m['MSE']:.6f}",
            "Train Time": f"{m['Train Time (s)']}s",
        })
    df_metrics = pd.DataFrame(rows).sort_values("R²", ascending=False)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── R² Bar Chart ──
    fig_bar = go.Figure()
    sorted_models = sorted(metrics.keys(), key=lambda k: metrics[k]["R2"], reverse=True)
    r2_vals = [metrics[m]["R2"] for m in sorted_models]
    bar_colors = ["#E8443A" if m == best_name else "#2E86AB" for m in sorted_models]

    fig_bar.add_trace(go.Bar(
        x=sorted_models, y=r2_vals,
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in r2_vals],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="R² Score Comparison (Test Set)",
        yaxis_title="R² Score",
        yaxis_range=[0, 1.08],
        template="plotly_dark",
        height=400,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Scatter plots for selected model ──
    st.markdown(f"### Actual vs Predicted — {model_choice}")
    Y_te_orig = test_data["Y_te_orig"]
    Y_pred_orig = preds_cache[model_choice]

    cols = st.columns(3)
    short_names = ["Oil Production", "Gas Volume", "Water Production"]
    colors_list = list(TARGET_COLORS.values())

    for j, (col, sn, clr) in enumerate(zip(cols, short_names, colors_list)):
        with col:
            fig = make_scatter_plot(Y_te_orig[:, j], Y_pred_orig[:, j], sn, clr)
            st.plotly_chart(fig, use_container_width=True)

    # ── Per-Target Metrics ──
    st.markdown(f"### Per-Target Metrics — {model_choice}")
    pt = metrics[model_choice]["per_target"]
    pt_rows = []
    for tgt_name, tgt_metrics in pt.items():
        pt_rows.append({
            "Target": tgt_name,
            "R²": f"{tgt_metrics['R2']:.4f}",
            "MAE": f"{tgt_metrics['MAE']:,.2f}",
            "RMSE": f"{tgt_metrics['RMSE']:,.2f}",
        })
    st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2: PREDICT FROM CSV
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Predict from Uploaded CSV")

    if uploaded_file is not None:
        try:
            content = uploaded_file.read()
            df_upload = pd.read_csv(io.BytesIO(content))
            st.info(f"Loaded **{len(df_upload):,}** rows × **{len(df_upload.columns)}** columns")

            X_input, missing_cols = preprocess_upload(df_upload, le)

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Scale and predict
                X_scaled_input = x_scaler.transform(X_input)
                model = models[model_choice]
                Y_pred_scaled = model.predict(X_scaled_input)
                Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

                # Attach predictions
                df_result = df_upload.copy()
                df_result["Predicted Oil (stb/day)"] = Y_pred[:, 0]
                df_result["Predicted Gas (scf/day)"] = Y_pred[:, 1]
                df_result["Predicted Water (stb/day)"] = Y_pred[:, 2]

                # ── Summary metrics ──
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Oil Prediction", format_number(Y_pred[:, 0].mean(), "stb/day"))
                c2.metric("Avg Gas Prediction", format_number(Y_pred[:, 1].mean(), "scf/day"))
                c3.metric("Avg Water Prediction", format_number(Y_pred[:, 2].mean(), "stb/day"))

                # ── Production trend chart ──
                if "PRODUCTION DATE" in df_result.columns:
                    df_result["PRODUCTION DATE"] = pd.to_datetime(
                        df_result["PRODUCTION DATE"], errors="coerce"
                    )
                    df_plot = df_result.dropna(subset=["PRODUCTION DATE"]).sort_values(
                        "PRODUCTION DATE"
                    )

                    fig_trend = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Oil Production", "Gas Production", "Water Production"],
                        vertical_spacing=0.06,
                    )

                    pred_cols = [
                        ("Predicted Oil (stb/day)", "#D4760A"),
                        ("Predicted Gas (scf/day)", "#E85D26"),
                        ("Predicted Water (stb/day)", "#1C7ED6"),
                    ]
                    actual_cols = [
                        ("Oil Production (stb/day)", "#D4760A"),
                        ("Gas Volume (scf/day)", "#E85D26"),
                        ("Water Production (stb/day)", "#1C7ED6"),
                    ]

                    for i, ((pred_c, clr), (act_c, _)) in enumerate(
                        zip(pred_cols, actual_cols), 1
                    ):
                        # Plot actuals if they exist and aren't all NaN
                        if act_c in df_plot.columns and df_plot[act_c].notna().any():
                            fig_trend.add_trace(go.Scatter(
                                x=df_plot["PRODUCTION DATE"],
                                y=df_plot[act_c],
                                mode="lines", name=f"Actual {act_c.split('(')[0].strip()}",
                                line=dict(color=clr, width=2),
                                opacity=0.6,
                            ), row=i, col=1)

                        fig_trend.add_trace(go.Scatter(
                            x=df_plot["PRODUCTION DATE"],
                            y=df_plot[pred_c],
                            mode="lines",
                            name=f"Predicted {pred_c.split('(')[0].strip().replace('Predicted ', '')}",
                            line=dict(color=clr, width=2, dash="dot"),
                        ), row=i, col=1)

                    fig_trend.update_layout(
                        height=700, template="plotly_dark",
                        title="Predicted Production Trends",
                        showlegend=True,
                        margin=dict(t=60, b=40),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                # ── Data table ──
                st.markdown("### Prediction Results")
                display_cols = ["PRODUCTION DATE"] if "PRODUCTION DATE" in df_result.columns else []
                display_cols += [
                    "Predicted Oil (stb/day)",
                    "Predicted Gas (scf/day)",
                    "Predicted Water (stb/day)",
                ]
                st.dataframe(
                    df_result[display_cols].head(200),
                    use_container_width=True, hide_index=True,
                )

                # ── Download ──
                csv_out = df_result.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Full Predictions (CSV)",
                    csv_out, "ogw_predictions.csv", "text/csv",
                )

        except Exception as e:
            st.error(f"Error processing upload: {e}")
    else:
        st.info(
            "Upload a CSV file in the sidebar to generate predictions. "
            "The file should contain the same feature columns as the training data."
        )

        st.markdown("**Required columns:**")
        req_cols = [
            "N_WELL_BORE_CODE", "WELL_TYPE", "Downhole Pressure (PSI)",
            "Downhole Temperature (Kelvin)", "Average Tubing Pressure",
            "Annulus Pressure (PSI)", "AVG WHP (PSI)", "Choke Size",
            "PRODUCTION DATE",
        ]
        st.code(", ".join(req_cols))


# ─────────────────────────────────────────────────────────────────────
# TAB 3: MANUAL PREDICTION
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Manual Prediction")
    st.caption("Enter operational parameters to get real-time oil, gas, and water production forecasts.")

    with st.form("manual_input"):
        col1, col2, col3 = st.columns(3)

        with col1:
            dp = st.number_input(
                "Downhole Pressure (PSI)", min_value=0.0,
                max_value=6000.0, value=3378.0, step=50.0,
            )
            dt = st.number_input(
                "Downhole Temperature (K)", min_value=250.0,
                max_value=450.0, value=377.0, step=1.0,
            )
            atp = st.number_input(
                "Avg Tubing Pressure (PSI)", min_value=0.0,
                max_value=6000.0, value=2472.0, step=50.0,
            )
            well_type = st.selectbox("Well Type", ["OP", "WI"])

        with col2:
            ap = st.number_input(
                "Annulus Pressure (PSI)", min_value=0.0,
                max_value=600.0, value=213.0, step=10.0,
            )
            whp = st.number_input(
                "Avg Wellhead Pressure (PSI)", min_value=0.0,
                max_value=3000.0, value=620.0, step=25.0,
            )
            choke = st.number_input(
                "Choke Size", min_value=0.0,
                max_value=150.0, value=14.3, step=1.0,
            )

        with col3:
            wellbore = st.selectbox(
                "Wellbore",
                options=[105, 106, 107, 108, 109],
                format_func=lambda x: f"Well {x}",
            )
            year = st.number_input("Year", min_value=2008, max_value=2030, value=2015)
            month = st.number_input("Month", min_value=1, max_value=12, value=7)
            day = st.number_input("Day", min_value=1, max_value=31, value=15)

        submitted = st.form_submit_button("🚀 Predict Production", use_container_width=True)

    if submitted:
        wt_encoded = le.transform([well_type])[0] if well_type in le.classes_ else 0

        input_df = pd.DataFrame([[
            wellbore, wt_encoded, dp, dt, atp, ap, whp, choke, year, month, day
        ]], columns=FEATURE_COLS)
        input_scaled = x_scaler.transform(input_df)

        model = models[model_choice]
        pred_scaled = model.predict(input_scaled)
        pred = y_scaler.inverse_transform(pred_scaled)[0]

        st.markdown("---")
        st.markdown("### Prediction Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("🛢️ Oil Production", f"{max(0, pred[0]):,.1f} stb/day")
        c2.metric("🔥 Gas Production", format_number(max(0, pred[1]), "scf/day"))
        c3.metric("💧 Water Production", f"{max(0, pred[2]):,.1f} stb/day")

        # Context: where does this prediction sit in training distribution?
        st.markdown("### Prediction Context")
        fig_ctx = make_subplots(rows=1, cols=3, subplot_titles=[
            "Oil", "Gas", "Water"
        ])

        for j, (tgt, clr) in enumerate(zip(TARGET_COLS, TARGET_COLORS.values())):
            hist_vals = train_data[tgt].values
            fig_ctx.add_trace(go.Histogram(
                x=hist_vals, nbinsx=50, marker_color=clr,
                opacity=0.5, name=tgt.split("(")[0].strip(),
                showlegend=False,
            ), row=1, col=j+1)

            fig_ctx.add_vline(
                x=max(0, pred[j]), line_dash="dash",
                line_color="white", line_width=2,
                annotation_text="Your prediction",
                annotation_font_color="white",
                row=1, col=j+1,
            )

        fig_ctx.update_layout(
            height=300, template="plotly_dark",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_ctx, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4: SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## SHAP Feature Importance")

    if not HAS_SHAP:
        st.warning("SHAP not installed. Run `pip install shap` to enable this tab.")
    else:
        st.caption(
            f"Analyzing how each feature contributes to predictions for **{model_choice}**"
        )

        # Determine which model/estimators to explain
        current_model = models[model_choice]
        X_full = pipe["X_scaled_full"]
        Y_full = pipe["Y_scaled_full"]

        n_shap = min(300, X_full.shape[0])
        X_sample = X_full[:n_shap]

        short_targets = ["Oil Production", "Gas Volume", "Water Production"]

        for j, (target, short) in enumerate(zip(TARGET_COLS, short_targets)):
            st.markdown(f"### {short}")

            with st.spinner(f"Computing SHAP values for {short}..."):
                try:
                    # Train a single-target model for clean SHAP
                    single_model = ExtraTreesRegressor(
                        n_estimators=500, max_depth=50, random_state=RANDOM_STATE, n_jobs=-1,
                    )
                    single_model.fit(X_full, Y_full[:, j])

                    explainer = shap.TreeExplainer(single_model)
                    shap_values = explainer.shap_values(X_sample)

                    # Summary plot
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                    shap.summary_plot(
                        shap_values, X_sample,
                        feature_names=FEATURE_COLS,
                        show=False,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)

                    # Top features with operational recommendations
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    top_idx = np.argsort(mean_abs)[::-1][:3]

                    recs = []
                    for idx in top_idx:
                        feat = FEATURE_COLS[idx]
                        impact = mean_abs[idx]
                        if "Pressure" in feat:
                            recs.append(
                                f"**{feat}** (impact: {impact:.4f}) — "
                                f"Monitor and optimize pressure to stabilize {short.lower()}."
                            )
                        elif "Choke" in feat:
                            recs.append(
                                f"**{feat}** (impact: {impact:.4f}) — "
                                f"Adjusting choke size strongly influences {short.lower()}."
                            )
                        elif "Temperature" in feat:
                            recs.append(
                                f"**{feat}** (impact: {impact:.4f}) — "
                                f"Temperature variations affect {short.lower()} predictions."
                            )
                        elif "year" in feat.lower():
                            recs.append(
                                f"**{feat}** (impact: {impact:.4f}) — "
                                f"Temporal trend: {short.lower()} changes with reservoir maturity."
                            )
                        else:
                            recs.append(
                                f"**{feat}** (impact: {impact:.4f}) — "
                                f"Review operational control as it impacts {short.lower()}."
                            )

                    with st.expander("💡 Operational Insights", expanded=True):
                        for rec in recs:
                            st.markdown(f"- {rec}")

                except Exception as e:
                    st.error(f"SHAP computation failed for {short}: {e}")


# ─────────────────────────────────────────────────────────────────────
# TAB 5: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## Training Data Explorer")

    viz_type = st.selectbox(
        "Visualization Type",
        ["Production Trends", "Feature Distributions", "Correlation Heatmap",
         "Scatter Analysis", "Production by Wellbore"],
    )

    if viz_type == "Production Trends":
        df_trends = train_data.copy()
        df_trends = df_trends.sort_values("PRODUCTION DATE")

        # Aggregate by month
        df_trends["YearMonth"] = df_trends["PRODUCTION DATE"].dt.to_period("M")
        monthly = df_trends.groupby("YearMonth")[TARGET_COLS].mean().reset_index()
        monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Oil Production", "Gas Volume", "Water Production"],
            vertical_spacing=0.06,
        )
        for i, (tgt, clr) in enumerate(zip(TARGET_COLS, TARGET_COLORS.values()), 1):
            fig.add_trace(go.Scatter(
                x=monthly["Date"], y=monthly[tgt],
                mode="lines+markers", line=dict(color=clr, width=2),
                marker=dict(size=4),
                name=tgt.split("(")[0].strip(),
            ), row=i, col=1)

        fig.update_layout(
            height=650, template="plotly_dark",
            title="Monthly Average Production Trends (2008–2015)",
            margin=dict(t=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Feature Distributions":
        feat_choice = st.selectbox("Select Feature", [
            "Downhole Pressure (PSI)", "Downhole Temperature (Kelvin)",
            "Average Tubing Pressure", "Annulus Pressure (PSI)",
            "AVG WHP (PSI)", "Choke Size",
        ])
        fig = px.histogram(
            train_data, x=feat_choice, nbins=50,
            template="plotly_dark",
            title=f"Distribution of {feat_choice}",
            color_discrete_sequence=["#2E86AB"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        numeric = train_data.select_dtypes(include=[np.number])
        corr = numeric.corr()

        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, template="plotly_dark",
            title="Pearson Correlation Matrix",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Analysis":
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X-axis", train_data.select_dtypes(include=[np.number]).columns)
        y_col = c2.selectbox(
            "Y-axis",
            train_data.select_dtypes(include=[np.number]).columns,
            index=min(3, len(train_data.select_dtypes(include=[np.number]).columns) - 1),
        )
        fig = px.scatter(
            train_data, x=x_col, y=y_col,
            opacity=0.3, template="plotly_dark",
            title=f"{y_col} vs {x_col}",
            color_discrete_sequence=["#E8443A"],
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Production by Wellbore":
        well_prod = train_data.groupby("WellBore Name")[TARGET_COLS].mean().reset_index()

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Oil", "Gas", "Water"],
        )
        for j, (tgt, clr) in enumerate(zip(TARGET_COLS, TARGET_COLORS.values()), 1):
            fig.add_trace(go.Bar(
                x=well_prod["WellBore Name"], y=well_prod[tgt],
                marker_color=clr, name=tgt.split("(")[0].strip(),
                showlegend=False,
            ), row=1, col=j)

        fig.update_layout(
            height=400, template="plotly_dark",
            title="Mean Production by Wellbore",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>"
    "OGW Prediction Dashboard · Group 4 — UENR Petroleum & Natural Gas Engineering · "
    "Supervised by Dr. Shaibu Mohammed · 2025"
    "</div>",
    unsafe_allow_html=True,
)