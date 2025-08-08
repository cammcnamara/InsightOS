import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import train_model
from preprocessing import clean_and_encode


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
        h1, h2, h3, .stMarkdown h4 {
            color: #111827;
            font-weight: 700;
        }
        .block-container {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 0.75rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.image("logo.png", width=400)
st.title("InsightOS: Root Cause Analysis Platform")
st.markdown(
    "An advanced ML-powered tool for identifying key business drivers and enabling actionable decision-making through data transparency."
)

page = st.sidebar.radio(
    "Navigation", ["Executive Summary", "Global Drivers", "Segment Analysis"]
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.sidebar:
        st.header("Dataset Metadata")
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        st.markdown("**Features:**")
        for col in df.columns:
            st.markdown(f"- {col}")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    target = st.selectbox("Select your target column", df.columns)
    group_column = st.selectbox(
        "Optional: Select a segment column to analyze",
        [col for col in df.columns if col != target],
    )

    if st.button("Generate Insights"):
        with st.spinner(
            "Analyzing dataset, training model, and calculating SHAP values..."
        ):
            df_cleaned, encoders, original_values = clean_and_encode(df)
            model, shap_values, X_test = train_model(df_cleaned, target)

        if page == "Executive Summary":
            st.markdown("---")
            st.subheader("Executive Summary")
            st.markdown(
                """
            - **Target Analyzed:** `{}`
            - **Model Type:** `{}`
            - **Global Top Driver:** `{}`
            - **Top Segment Analyzed:** `{}`
            """.format(
                    target,
                    "Classifier" if df[target].nunique() <= 10 else "Regressor",
                    X_test.columns[shap_values.abs.mean(0).values.argmax()],
                    df_cleaned[group_column].value_counts().index[0],
                )
            )

        elif page == "Global Drivers":
            st.markdown("---")
            st.subheader("Top Global Feature Drivers")
            shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
            mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("#### Global Feature Importance")
                    st.bar_chart(mean_abs_shap)
                with col2:
                    st.markdown("#### Raw SHAP Values")
                    st.dataframe(shap_df.head(), use_container_width=True)

        elif page == "Segment Analysis":
            st.markdown("---")
            st.subheader("Segment-Level Insights")
            top_segments = df_cleaned[group_column].value_counts().index[:3]

            for segment in top_segments:
                segment_label = original_values[group_column].get(segment, segment)
                with st.expander(f"Segment: {segment_label}"):
                    segment_df = df_cleaned[df_cleaned[group_column] == segment]
                    if len(segment_df) < 10:
                        st.warning("(Segment too small to analyze reliably)")
                        continue

                    _, seg_shap_values, seg_X_test = train_model(segment_df, target)
                    seg_shap_df = pd.DataFrame(
                        seg_shap_values.values, columns=seg_X_test.columns
                    )
                    seg_mean_abs_shap = (
                        seg_shap_df.abs().mean().sort_values(ascending=False)
                    )

                    col3, col4 = st.columns([2, 1])
                    with col3:
                        st.markdown("##### Segment Feature Importance")
                        st.bar_chart(seg_mean_abs_shap)
                    with col4:
                        top_driver = seg_mean_abs_shap.idxmax()
                        top_value = seg_mean_abs_shap.max()
                        st.metric(label="Top Driver", value=top_driver)
                        st.metric(label="Impact (SHAP units)", value=f"{top_value:.2f}")
                        st.markdown(
                            f"The top driver in segment **{segment_label}** is **{top_driver}**, contributing significantly to variation in the target."
                        )
