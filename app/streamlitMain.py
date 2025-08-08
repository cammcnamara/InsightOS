import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftTreeClassifier

from model import train_model
from preprocessing import clean_and_encode

# --- PAGE CONFIG ---
st.set_page_config(page_title="InsightOS", layout="wide")
st.markdown(
    """
    <style>
        html, body, .main {
            background-color: #f3f4f6;
            color: #111827;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1f2937;
            color: white;
            font-weight: 600;
            border-radius: 0.375rem;
            padding: 0.5rem 1.25rem;
            border: none;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #111827;
        }
        h1, h2, h3, .stMarkdown h4, .stMarkdown p {
            color: #111827;
            font-weight: 600;
        }
        .block-container {
            padding: 2rem;
            background-color: #aaaaaa;
            border-radius: 0.75rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.image("logo.png", width=500)
st.title("InsightOS: Root Cause Analysis Platform")
st.markdown(
    "An advanced ML-powered tool for identifying key business drivers and enabling actionable decision-making through data transparency."
)

# ========== Sidebar Settings ==========
st.sidebar.markdown("### Select Insights to Include")
show_exec_summary = st.sidebar.checkbox("Executive Summary", value=True)
show_global_drivers = st.sidebar.checkbox("Global Drivers", value=False)
show_segment_analysis = st.sidebar.checkbox("Segment Analysis", value=False)
show_causal_analysis = st.sidebar.checkbox("Causal Analysis", value=False)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# ========== If file uploaded ==========
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.sidebar:
        st.header("Dataset Metadata")
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        st.markdown("**Features:**")
        for col in df.columns:
            st.markdown(f"- {col}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Persistent session state
    if "target" not in st.session_state:
        st.session_state.target = df.columns[0]
    if "segment" not in st.session_state:
        st.session_state.segment = df.columns[1]

    st.session_state.target = st.selectbox(
        "Select your target column",
        df.columns,
        index=df.columns.get_loc(st.session_state.target),
    )
    st.session_state.segment = st.selectbox(
        "Select segment column",
        [col for col in df.columns if col != st.session_state.target],
        index=0,
    )

    target = st.session_state.target
    segment = st.session_state.segment

    run_insights = st.button("Generate Insights")

    if run_insights:
        with st.spinner("Cleaning data, training model, calculating SHAP..."):
            df_cleaned, encoders, original_values = clean_and_encode(df)
            model, shap_values, X_test = train_model(df_cleaned, target)

        if show_exec_summary:
            st.markdown("---")
            st.subheader("Executive Summary")
            st.markdown(
                f"""
                - **Target Analyzed:** `{target}`
                - **Model Type:** `{"Classifier" if df[target].nunique() <= 10 else "Regressor"}`
                - **Global Top Driver:** `{X_test.columns[shap_values.abs.mean(0).values.argmax()]}`
                - **Top Segment Analyzed:** `{df_cleaned[segment].value_counts().index[0]}`
            """
            )

        if show_global_drivers:
            st.markdown("---")
            st.subheader("Top Global Feature Drivers")
            shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
            mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### Global Feature Importance")
                st.bar_chart(mean_abs_shap)
            with col2:
                st.markdown("#### SHAP Sample Table")
                st.dataframe(shap_df.head(), use_container_width=True)

        if show_segment_analysis:
            st.markdown("---")
            st.subheader("Segment-Level Insights")
            top_segments = df_cleaned[segment].value_counts().index[:3]

            for seg in top_segments:
                seg_label = original_values[segment].get(seg, seg)
                with st.expander(f"Segment: {seg_label}"):
                    seg_df = df_cleaned[df_cleaned[segment] == seg]
                    if len(seg_df) < 10:
                        st.warning("Segment too small to analyze")
                        continue

                    _, seg_shap_vals, seg_X_test = train_model(seg_df, target)
                    seg_shap_df = pd.DataFrame(
                        seg_shap_vals.values, columns=seg_X_test.columns
                    )
                    seg_mean_shap = (
                        seg_shap_df.abs().mean().sort_values(ascending=False)
                    )

                    col3, col4 = st.columns([2, 1])
                    with col3:
                        st.markdown("##### Segment Feature Importance")
                        st.bar_chart(seg_mean_shap)
                    with col4:
                        top_driver = seg_mean_shap.idxmax()
                        top_value = seg_mean_shap.max()
                        st.metric("Top Driver", top_driver)
                        st.metric("Impact (SHAP)", f"{top_value:.2f}")

        if show_causal_analysis and run_insights:
            st.markdown("---")
            st.subheader("Causal Uplift Modeling")

            if "treatment_col" not in st.session_state:
                st.session_state.treatment_col = df.columns[0]
            if "outcome_col" not in st.session_state:
                st.session_state.outcome_col = df.columns[1]

            st.session_state.treatment_col = st.selectbox(
                "Select binary treatment column",
                df.columns,
                index=df.columns.get_loc(st.session_state.treatment_col),
            )

            st.session_state.outcome_col = st.selectbox(
                "Select outcome column",
                [col for col in df.columns if col != st.session_state.treatment_col],
                index=0,
            )

            if st.button("Run Uplift Analysis"):
                with st.spinner("Fitting causal model..."):
                    df_causal = df[
                        [st.session_state.treatment_col, st.session_state.outcome_col]
                    ].dropna()
                    model = UpliftTreeClassifier(max_depth=4, min_samples_leaf=50)
                    model.fit(
                        X=df_causal[[st.session_state.treatment_col]],
                        treatment=df_causal[st.session_state.treatment_col],
                        y=df_causal[st.session_state.outcome_col],
                    )
                    uplift_scores = model.predict(
                        df_causal[[st.session_state.treatment_col]]
                    )
                    st.success("Causal model run successfully.")

                    df_causal["uplift_score"] = uplift_scores
                    st.dataframe(df_causal.head())
