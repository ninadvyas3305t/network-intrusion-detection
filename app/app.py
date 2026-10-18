"""AI-Based Network Intrusion Detection System
Streamlit Web Application Dashboard
Portfolio & College Project Defense Interface
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import IntrusionPredictor
from src.explainability import IntrusionExplainer, FEATURE_DESCRIPTIONS

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean Styling for Professional Academic & Portfolio Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .alert-banner {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .status-badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-attack {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Caching Data & Model Artifacts
@st.cache_resource
def get_predictor() -> IntrusionPredictor:
    """Load and cache the inference engine."""
    return IntrusionPredictor(models_dir="models")

@st.cache_resource
def get_explainer() -> IntrusionExplainer:
    """Load and cache the SHAP explainer."""
    return IntrusionExplainer(model_path="models/best_model_binary.joblib", metadata_path="models/feature_metadata.json")

@st.cache_data
def get_training_results() -> Dict[str, Any]:
    """Load precomputed training experiment results."""
    with open("models/training_results.json", "r") as f:
        return json.load(f)

@st.cache_data
def get_threshold_results() -> Dict[str, Any]:
    """Load precomputed threshold analysis."""
    with open("models/threshold_analysis.json", "r") as f:
        return json.load(f)

@st.cache_data
def get_eda_summary() -> Dict[str, Any]:
    """Load precomputed EDA summary."""
    with open("data/processed/eda_summary.json", "r") as f:
        return json.load(f)

@st.cache_data
def get_shap_summary() -> Dict[str, Any]:
    """Load precomputed global SHAP summary."""
    with open("models/shap_summary.json", "r") as f:
        return json.load(f)

# Sidebar Navigation & Global Controls
def render_sidebar():
    st.sidebar.markdown("## 🛡️ NIDS Portal")
    st.sidebar.caption("AI-Based Network Intrusion Detection System")

    nav_options = [
        "Executive Dashboard",
        "Dataset & EDA",
        "Intrusion Detection",
        "Model Performance",
        "Explainability (SHAP)",
    ]
    selected_page = st.sidebar.radio("Navigation", nav_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection Threshold")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Custom cut-off probability for classifying a network flow as Attack. Lower values increase sensitivity to stealth attacks; higher values reduce false alarms.",
    )

    if threshold == 0.50:
        st.sidebar.info("🎯 **Default (0.50)**: Mathematically balanced production operating point.")
    elif threshold < 0.50:
        st.sidebar.warning(f"🚨 **High Sensitivity ({threshold:.2f})**: Paranoid security mode (minimizes missed attacks, higher false alarms).")
    else:
        st.sidebar.success(f"⚖️ **High Precision ({threshold:.2f})**: Conservative alert mode (minimizes false alarms, strict confirmation).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Project Metadata")
    st.sidebar.markdown("""
    - **Dataset**: UNSW-NB15 Benchmark
    - **Total Records**: 257,673
    - **Primary Model**: XGBoost
    - **Classes**: 10 (Binary + Multiclass)
    - **College Viva Version**: 1.0
    """)

    return selected_page, threshold

# Page 1: Executive Dashboard
def render_dashboard(training_res: Dict[str, Any], eda_res: Dict[str, Any]):
    st.markdown('<div class="main-title">🛡️ AI-Based Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time network traffic anomaly detection and flow classification powered by Supervised Machine Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-banner">
        <strong>📌 Official Benchmark Evaluation Notice:</strong> The performance figures displayed below represent 
        rigorous empirical measurements evaluated against the <strong>official UNSW-NB15 test partition (82,332 test samples)</strong>. 
        Zero data leakage was permitted; training and testing distributions are strictly isolated.
    </div>
    """, unsafe_allow_html=True)

    xgb_metrics = training_res["binary_experiments"]["XGBoost"]["metrics"]
    xgb_timing = training_res["binary_experiments"]["XGBoost"]["training_time_seconds"]
    xgb_latency = training_res["binary_experiments"]["XGBoost"]["inference_latency"]["per_sample_latency_ms"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">257,673</div>
            <div class="metric-label">Total Flow Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{xgb_metrics['accuracy']*100:.1f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{xgb_metrics['recall']*100:.1f}%</div>
            <div class="metric-label">Attack Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{xgb_metrics['f1']:.4f}</div>
            <div class="metric-label">Binary F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{xgb_metrics['roc_auc']:.4f}</div>
            <div class="metric-label">ROC-AUC Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1.0])

    with left_col:
        st.subheader("🎯 Primary & Secondary Classification Tasks")
        st.markdown("""
        The system implements a dual-stage supervised learning architecture:
        - **Primary Task (Binary Classification)**: Discerning **Normal (Benign)** traffic from **Malicious Attack** traffic with a focus on minimizing False Negatives (missed intrusions).
        - **Secondary Task (Multiclass Classification)**: Granular categorization into **10 distinct threat vectors** (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms).
        - **Production Speed**: Full XGBoost training completed in **0.93 seconds** with an inference latency of **0.00022 ms per flow**.
        """)

        # Donut Chart of Test Set Distribution
        test_binary = eda_res["test_binary_distribution"]
        donut_data = pd.DataFrame({
            "Traffic Type": ["Normal (Benign)", "Malicious Attack"],
            "Count": [test_binary["0"], test_binary["1"]],
        })
        fig = px.pie(
            donut_data,
            names="Traffic Type",
            values="Count",
            hole=0.55,
            color="Traffic Type",
            color_discrete_map={"Normal (Benign)": "#2B5C8F", "Malicious Attack": "#D95F02"},
            title="Official Benchmark Test Partition Breakdown (N=82,332)",
        )
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("🏗️ System Architecture Pipeline")
        st.markdown("""
        ```mermaid
        flowchart TD
            A[Raw Network Packet Flows] --> B[42 Extracted Flow Features]
            B --> C[Data Preprocessing Pipeline]
            C --> D[StandardScaler: 39 Flow Metrics]
            C --> E[OneHotEncoder: proto, service, state]
            D --> F[194 Transformed Feature Vector]
            E --> F
            F --> G[XGBoost Detection Engine]
            G --> H[Binary Decision: Normal vs Attack]
            G --> I[10-Class Threat Taxonomy]
            G --> J[SHAP TreeExplainer Attribution]
        ```
        """)
        st.caption("End-to-end reproducible pipeline fitted strictly on training partition.")

# Page 2: Dataset & EDA
def render_dataset_eda(eda_res: Dict[str, Any]):
    st.markdown('<div class="main-title">📈 Dataset & Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical investigation, class distributions, and domain correlation mechanics of UNSW-NB15</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Class & Attack Taxonomy", "🔗 Correlation & Collinearity Study", "📦 Feature Distributions"])

    with tab1:
        st.subheader("Distribution Across 10 Attack Categories")
        st.markdown("""
        The UNSW-NB15 dataset includes 9 modern CVE-based attack categories alongside legitimate normal traffic.
        Notice the realistic long-tailed distribution: dominant classes (**Normal, Generic, Exploits**) make up over 76% of traffic,
        while extreme rare threats (**Worms: 174 total, Shellcode: 1,511 total**) represent less than 1%.
        """)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("data/processed/eda_figures/binary_class_distribution.png"):
                st.image("data/processed/eda_figures/binary_class_distribution.png", caption="Train vs Test Binary Distribution", use_container_width=True)
        with col_img2:
            if os.path.exists("data/processed/eda_figures/attack_categories_distribution.png"):
                st.image("data/processed/eda_figures/attack_categories_distribution.png", caption="10 Attack Categories Breakdown", use_container_width=True)

        st.markdown("#### Exact Category Frequencies (Benchmark Partition)")
        train_cats = eda_res["train_multiclass_distribution"]
        test_cats = eda_res["test_multiclass_distribution"]

        cat_rows = []
        for cat in train_cats.keys():
            tr = train_cats.get(cat, 0)
            te = test_cats.get(cat, 0)
            tot = tr + te
            pct = (tot / eda_res["total_records"]) * 100
            cat_rows.append({
                "Category": cat,
                "Train Samples": f"{tr:,}",
                "Test Samples": f"{te:,}",
                "Total Samples": f"{tot:,}",
                "Proportion (%)": f"{pct:.2f}%",
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

    with tab2:
        st.subheader("🔗 Collinearity & Physical Networking Dependencies")
        st.markdown("""
        In network flow accounting, high correlations between features are **physical consequences of network protocol design**,
        not redundant artifacts. We preserved these features for tree-based models (Random Forest, XGBoost) because decision trees
        evaluate orthogonal threshold splits without suffering variance inflation.
        """)

        if os.path.exists("data/processed/eda_figures/correlation_heatmap.png"):
            st.image("data/processed/eda_figures/correlation_heatmap.png", caption="Correlation Heatmap of Key Flow Metrics", use_container_width=True)

        st.markdown("#### High-Correlation Pairs (|r| ≥ 0.90) and Domain Rationale")
        high_corr_df = pd.DataFrame(eda_res["high_correlations"])
        high_corr_df.columns = ["Feature 1", "Feature 2", "Pearson r", "Networking Domain Explanation"]
        st.dataframe(high_corr_df, use_container_width=True)

    with tab3:
        st.subheader("📦 Discriminative Feature Profiles")
        st.markdown("""
        Key features such as **Source Time-To-Live (`sttl`)** and **Transaction Rate (`rate`)** exhibit 
        pronounced divergence between Normal and Attack sessions, serving as high-importance split candidates for XGBoost.
        """)
        if os.path.exists("data/processed/eda_figures/feature_distributions_by_class.png"):
            st.image("data/processed/eda_figures/feature_distributions_by_class.png", caption="Boxplots of Key Discriminative Flow Metrics by Class", use_container_width=True)

# Page 3: Intrusion Detection (Live Inference)
def render_intrusion_detection(predictor: IntrusionPredictor, threshold: float):
    st.markdown('<div class="main-title">🛡️ Intrusion Detection Live Inference Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Execute real-time network traffic classification on uploaded batch CSVs or simulate single network flows</div>', unsafe_allow_html=True)

    mode = st.radio("Select Inference Mode", ["📂 Batch CSV Traffic Analysis", "🔍 Single Network Flow Simulator"], horizontal=True)

    if mode == "📂 Batch CSV Traffic Analysis":
        st.subheader("Batch CSV Inspection")
        st.markdown("""
        Upload a CSV containing captured network flow features. The pipeline validates all required columns, 
        applies the saved `StandardScaler` and `OneHotEncoder` transformers, and classifies each flow.
        """)

        # Sample CSV download button
        if os.path.exists("data/processed/sample_test_batch.csv"):
            with open("data/processed/sample_test_batch.csv", "rb") as f:
                st.download_button(
                    label="📥 Download Sample Test Batch (50 Mixed Flows)",
                    data=f,
                    file_name="unsw_nb15_sample_test_batch.csv",
                    mime="text/csv",
                    help="Click to download an authentic 50-sample CSV from the test partition to test the uploader.",
                )

        uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded file contains **{len(raw_df)}** records and **{len(raw_df.columns)}** columns.")

                # Validate inputs
                is_valid, missing_cols = predictor.validate_inputs(raw_df)

                if not is_valid:
                    st.error(f"❌ Input Validation Failed: Missing {len(missing_cols)} required features.")
                    st.write("Missing columns:", missing_cols[:10], "... (truncated)" if len(missing_cols) > 10 else "")
                else:
                    st.success("✅ Input Validation Passed: All required flow features are present.")

                    with st.spinner("Executing inference pipeline across flows..."):
                        pred_df = predictor.predict_batch(raw_df, threshold=threshold)

                    # Summary Metrics
                    total_flows = len(pred_df)
                    attack_count = int((pred_df["prediction"] == "Attack").sum())
                    normal_count = int((pred_df["prediction"] == "Normal").sum())
                    attack_pct = (attack_count / total_flows) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Flows Analyzed", f"{total_flows:,}")
                    m2.metric("Normal Flows", f"{normal_count:,}")
                    m3.metric("Attack Flows Detected", f"{attack_count:,}")
                    m4.metric("Attack Proportion", f"{attack_pct:.1f}%")

                    # Attack Categories Breakdown
                    if attack_count > 0:
                        attacks_only = pred_df[pred_df["prediction"] == "Attack"]
                        cat_counts = attacks_only["predicted_category"].value_counts().reset_index()
                        cat_counts.columns = ["Attack Category", "Count"]

                        fig_cat = px.bar(
                            cat_counts,
                            x="Attack Category",
                            y="Count",
                            color="Attack Category",
                            title="Threat Category Breakdown Among Detected Attacks",
                        )
                        fig_cat.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0))
                        st.plotly_chart(fig_cat, use_container_width=True)

                    st.markdown("#### Inspection Table")
                    display_cols = [c for c in ["prediction", "attack_probability", "predicted_category", "category_confidence", "proto", "service", "state", "dur", "sbytes", "dbytes", "sttl", "rate"] if c in pred_df.columns]
                    st.dataframe(pred_df[display_cols].head(100), use_container_width=True)

                    # Download Results
                    csv_export = pred_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Full Predictions as CSV",
                        data=csv_export,
                        file_name="intrusion_detection_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

    else:
        st.subheader("🔍 Single Network Flow Simulator & Explainability")
        st.markdown("Simulate an individual network flow or load a realistic scenario to inspect instant model reasoning:")

        # Presets
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        preset_type = None
        if p_col1.button("🟢 Normal Web Flow"):
            preset_type = "normal"
        if p_col2.button("🔴 Reconnaissance / Port Scan"):
            preset_type = "recon"
        if p_col3.button("🔴 DoS Flooding Attack"):
            preset_type = "dos"
        if p_col4.button("🔴 Web Exploit"):
            preset_type = "exploit"

        # Defaults based on preset
        if preset_type == "normal":
            d_dur, d_proto, d_service, d_state = 0.52, "tcp", "http", "fin"
            d_spkts, d_dpkts, d_sbytes, d_dbytes = 10, 8, 800, 2400
            d_rate, d_sttl, d_dttl = 32.5, 31, 29
            d_ct_srv_src, d_ct_srv_dst = 2, 2
        elif preset_type == "recon":
            d_dur, d_proto, d_service, d_state = 0.001, "tcp", "none", "int"
            d_spkts, d_dpkts, d_sbytes, d_dbytes = 2, 0, 120, 0
            d_rate, d_sttl, d_dttl = 1500.0, 254, 0
            d_ct_srv_src, d_ct_srv_dst = 45, 45
        elif preset_type == "dos":
            d_dur, d_proto, d_service, d_state = 0.00001, "udp", "dns", "int"
            d_spkts, d_dpkts, d_sbytes, d_dbytes = 200, 0, 14000, 0
            d_rate, d_sttl, d_dttl = 25000.0, 254, 0
            d_ct_srv_src, d_ct_srv_dst = 60, 60
        elif preset_type == "exploit":
            d_dur, d_proto, d_service, d_state = 1.45, "tcp", "http", "fin"
            d_spkts, d_dpkts, d_sbytes, d_dbytes = 35, 28, 4800, 12000
            d_rate, d_sttl, d_dttl = 43.4, 62, 252
            d_ct_srv_src, d_ct_srv_dst = 5, 4
        else:
            d_dur, d_proto, d_service, d_state = 0.12, "tcp", "http", "fin"
            d_spkts, d_dpkts, d_sbytes, d_dbytes = 14, 12, 1200, 3600
            d_rate, d_sttl, d_dttl = 180.0, 64, 60
            d_ct_srv_src, d_ct_srv_dst = 3, 3

        with st.form("single_flow_form"):
            c1, c2, c3, c4 = st.columns(4)
            proto_val = c1.selectbox("Protocol (proto)", ["tcp", "udp", "arp", "ospf", "icmp"], index=["tcp", "udp", "arp", "ospf", "icmp"].index(d_proto))
            service_val = c2.selectbox("Service (service)", ["http", "dns", "smtp", "ftp", "ssh", "none"], index=["http", "dns", "smtp", "ftp", "ssh", "none"].index(d_service))
            state_val = c3.selectbox("State (state)", ["fin", "con", "int", "req", "rst"], index=["fin", "con", "int", "req", "rst"].index(d_state))
            dur_val = c4.number_input("Duration (dur, s)", value=float(d_dur), format="%.5f")

            c5, c6, c7, c8 = st.columns(4)
            spkts_val = c5.number_input("Source Packets (spkts)", value=int(d_spkts))
            dpkts_val = c6.number_input("Dest Packets (dpkts)", value=int(d_dpkts))
            sbytes_val = c7.number_input("Source Bytes (sbytes)", value=int(d_sbytes))
            dbytes_val = c8.number_input("Dest Bytes (dbytes)", value=int(d_dbytes))

            c9, c10, c11, c12 = st.columns(4)
            rate_val = c9.number_input("Packet Rate (rate)", value=float(d_rate))
            sttl_val = c10.number_input("Source TTL (sttl)", value=int(d_sttl))
            dttl_val = c11.number_input("Dest TTL (dttl)", value=int(d_dttl))
            ct_srv_src_val = c12.number_input("Service Connections (ct_srv_src)", value=int(d_ct_srv_src))

            submit_btn = st.form_submit_button("🚀 Analyze Network Flow")

        if submit_btn:
            # Build flow dictionary
            flow_dict = {
                "dur": dur_val, "proto": proto_val, "service": service_val, "state": state_val,
                "spkts": spkts_val, "dpkts": dpkts_val, "sbytes": sbytes_val, "dbytes": dbytes_val,
                "rate": rate_val, "sttl": sttl_val, "dttl": dttl_val, "ct_srv_src": ct_srv_src_val,
                "ct_srv_dst": ct_srv_src_val, "ct_dst_ltm": ct_srv_src_val,
            }

            res = predictor.predict_single(flow_dict, threshold=threshold, with_explanation=True)

            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                st.markdown("### Decision Result")
                if res["prediction"] == "Attack":
                    st.markdown("""
                    <div style="background-color:#FEE2E2; border-left:6px solid #EF4444; padding:16px; border-radius:8px;">
                        <h3 style="color:#991B1B; margin:0;">⚠️ INTRUSION DETECTED</h3>
                        <p style="margin:4px 0 0 0; color:#7F1D1D; font-size:1.1rem;">
                            Classified as <strong>Attack</strong> with probability <strong>{:.1f}%</strong>
                        </p>
                    </div>
                    """.format(res["attack_probability"] * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color:#DCFCE7; border-left:6px solid #10B981; padding:16px; border-radius:8px;">
                        <h3 style="color:#166534; margin:0;">🛡️ NORMAL TRAFFIC</h3>
                        <p style="margin:4px 0 0 0; color:#14532D; font-size:1.1rem;">
                            Classified as <strong>Normal</strong> (Attack probability: {:.1f}%)
                        </p>
                    </div>
                    """.format(res["attack_probability"] * 100), unsafe_allow_html=True)

                st.markdown(f"""
                - **Active Decision Threshold**: `{threshold:.2f}`
                - **Attack Probability**: `{res['attack_probability']:.4f}`
                - **Normal Probability**: `{res['normal_probability']:.4f}`
                - **Predicted Category**: `{res['predicted_category']}` ({res['category_confidence']*100:.1f}% confidence)
                """)

            with res_col2:
                st.markdown("### 🔍 SHAP Local Attribution")
                expl = res["explanation"]
                factors = expl["top_contributing_factors"]

                chart_df = pd.DataFrame({
                    "Feature": [f["feature"] for f in factors][::-1],
                    "Impact": [f["shap_value"] for f in factors][::-1],
                    "Direction": ["Pushes Attack" if f["shap_value"] > 0 else "Pushes Normal" for f in factors][::-1]
                })

                fig_local = px.bar(
                    chart_df,
                    x="Impact",
                    y="Feature",
                    color="Direction",
                    orientation="h",
                    color_discrete_map={"Pushes Attack": "#EF4444", "Pushes Normal": "#10B981"},
                    title="Key Factors Driving This Specific Prediction",
                )
                fig_local.update_layout(height=280, margin=dict(t=35, b=0, l=0, r=0))
                st.plotly_chart(fig_local, use_container_width=True)

            st.markdown("#### Plain-English Factor Breakdown")
            for f in factors:
                direction_txt = "🔴 **Pushed towards Attack**" if f["shap_value"] > 0 else "🟢 **Mitigated towards Normal**"
                st.write(f"- {direction_txt}: `{f['feature']}` (SHAP = {f['shap_value']:+.3f}) — *{f['description']}*")

# Page 4: Model Performance
def render_model_performance(training_res: Dict[str, Any], threshold_res: Dict[str, Any]):
    st.markdown('<div class="main-title">⚖️ Model Performance & Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative evaluation across Baseline, Random Forest, and XGBoost on official benchmark partition</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Binary Models Benchmark", "🎯 Multiclass 10-Class Benchmark", "🎚️ Threshold Trade-Off Study"])

    with tab1:
        st.subheader("Binary Classification Benchmark (Normal vs. Attack)")
        st.markdown("""
        Evaluated on the full 82,332 benchmark test samples. All execution timings and memory usages were 
        measured directly on host hardware:
        """)

        b_models = training_res["binary_experiments"]
        rows = []
        for name, data in b_models.items():
            m = data["metrics"]
            rows.append({
                "Model": name,
                "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
                "Precision (%)": f"{m['precision']*100:.2f}%",
                "Recall (Attack) (%)": f"{m['recall']*100:.2f}%",
                "F1-Score": f"{m['f1']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Training Time (s)": f"{data['training_time_seconds']:.2f} s",
                "Peak RAM (MB)": f"{data['peak_memory_mb']:.2f} MB",
                "Per-Sample Latency (ms)": f"{data['inference_latency']['per_sample_latency_ms']:.5f} ms",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Model Selection Defense: Why XGBoost over Random Forest?")
        st.markdown("""
        > **College Viva Defense Point**:
        > - **Random Forest** achieved slightly higher attack recall at default threshold (**98.94%** vs **98.65%**, missing 482 attacks vs 611).
        > - However, **XGBoost** achieved superior **Precision (82.00% vs 80.92%)**, higher overall **F1-Score (0.8956 vs 0.8903)**, and higher **ROC-AUC (0.9851 vs 0.9824)**.
        > - Furthermore, XGBoost trained in **0.93 seconds** using **0.67 MB RAM**, whereas Random Forest required **2.22 seconds** and **101.49 MB RAM**.
        > - As demonstrated in the Threshold Study, if parity on attack recall is desired, setting XGBoost's threshold to **0.40** yields **99.52% recall** (only 219 missed attacks), easily surpassing Random Forest!
        """)

        st.subheader("Confusion Matrices (Test Partition)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if os.path.exists("data/processed/model_figures/cm_binary_logistic_regression_baseline.png"):
                st.image("data/processed/model_figures/cm_binary_logistic_regression_baseline.png", caption="Logistic Regression Confusion Matrix", use_container_width=True)
        with c2:
            if os.path.exists("data/processed/model_figures/cm_binary_random_forest.png"):
                st.image("data/processed/model_figures/cm_binary_random_forest.png", caption="Random Forest Confusion Matrix", use_container_width=True)
        with c3:
            if os.path.exists("data/processed/model_figures/cm_binary_xgboost.png"):
                st.image("data/processed/model_figures/cm_binary_xgboost.png", caption="XGBoost Confusion Matrix", use_container_width=True)

    with tab2:
        st.subheader("Multiclass 10-Category Benchmark")
        st.markdown("""
        Secondary task evaluating fine-grained attack taxonomy. Notice the crucial distinction between 
        **Weighted F1** and **Macro F1**:
        """)

        m_models = training_res["multiclass_experiments"]
        m_rows = []
        for name, data in m_models.items():
            m = data["metrics"]
            m_rows.append({
                "Model": name,
                "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
                "Weighted F1": f"{m['f1_weighted']:.4f}",
                "Macro F1": f"{m['f1_macro']:.4f}",
                "Macro ROC-AUC": f"{m['roc_auc_macro']:.4f}",
                "Training Time (s)": f"{data['training_time_seconds']:.2f} s",
                "Peak RAM (MB)": f"{data['peak_memory_mb']:.2f} MB",
            })
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

        st.markdown("#### Performance on Extreme Rare Classes (XGBoost Multiclass)")
        st.markdown("""
        In imbalanced cybersecurity data, standard models frequently ignore rare classes. 
        Here is how XGBoost performed on rare attack vectors:
        - **Worms (44 test samples)**: Precision = **60.00%**, Recall = **47.73%**, F1 = **0.5316** (detected 21 out of 44 extreme-rare worms).
        - **Shellcode (378 test samples)**: Precision = **32.32%**, Recall = **78.84%**, F1 = **0.4585** (caught 298 out of 378 shellcodes).
        - **Generic Attacks (18,871 test samples)**: Precision = **99.77%**, Recall = **96.98%**, F1 = **0.9836**.
        """)

        if os.path.exists("data/processed/model_figures/cm_multi_xgboost_multiclass.png"):
            st.image("data/processed/model_figures/cm_multi_xgboost_multiclass.png", caption="XGBoost 10-Class Confusion Matrix", use_container_width=True)

    with tab3:
        st.subheader("🎚️ Decision Threshold Trade-Off Analysis")
        st.markdown("""
        Decision thresholding governs the operational sensitivity of the Intrusion Detection System. 
        Lowering the threshold catches stealthier attacks at the expense of alert fatigue:
        """)

        if os.path.exists("data/processed/model_figures/threshold_tradeoff.png"):
            st.image("data/processed/model_figures/threshold_tradeoff.png", caption="Precision-Recall-F1 & False Positives vs False Negatives Curve", use_container_width=True)

        st.markdown("#### Measured Test Set Results Across Thresholds")
        st.dataframe(pd.DataFrame(threshold_res["test_threshold_curve"]), use_container_width=True)

# Page 5: Explainability
def render_explainability(shap_summary: Dict[str, Any]):
    st.markdown('<div class="main-title">🔍 Model Explainability (SHAP TreeExplainer)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interpretable AI: Explaining model decision mechanics through game-theoretic Shapley values</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-banner">
        <strong>⚠️ Academic Viva Distinction (Model Attribution vs. Causal Proof):</strong><br>
        SHAP (SHapley Additive exPlanations) quantifies <em>feature attribution</em>—how much each feature adjusted the 
        model's log-odds output away from the base value within the learned manifold. 
        It does <strong>not</strong> prove physical causation in a forensic sense, but rather makes the machine learning 
        reasoning transparent and audit-ready for security analysts.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.subheader("Global Feature Importance (Top Predictors)")
        if os.path.exists("data/processed/model_figures/shap_summary_bar.png"):
            st.image("data/processed/model_figures/shap_summary_bar.png", caption="Top 15 Most Influential Features (Global SHAP)", use_container_width=True)

    with col2:
        st.subheader("Top Predictive Features & Physical Meanings")
        top_f_df = pd.DataFrame(shap_summary["top_features"])
        top_f_df = top_f_df[["rank", "feature", "mean_abs_shap", "description"]]
        top_f_df.columns = ["Rank", "Feature Name", "Mean |SHAP|", "Domain Interpretation"]
        st.dataframe(top_f_df.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 How Local TreeSHAP Works")
    st.markdown("""
    For any individual connection flow $x$:
    $$\\hat{f}(x) = \\phi_0 + \\sum_{i=1}^{M} \\phi_i(x)$$
    Where:
    - $\\phi_0$ is the **base value** (average model output across the background population).
    - $\\phi_i(x)$ is the **Shapley contribution** of feature $i$.
    - A positive value $\\phi_i > 0$ pushes the prediction toward **Malicious Attack**.
    - A negative value $\\phi_i < 0$ mitigates the score toward **Normal Traffic**.
    """)

# Main Entry Point
def main():
    selected_page, threshold = render_sidebar()

    # Load resources
    predictor = get_predictor()
    training_res = get_training_results()
    threshold_res = get_threshold_results()
    eda_res = get_eda_summary()
    shap_summary = get_shap_summary()

    if selected_page == "Executive Dashboard":
        render_dashboard(training_res, eda_res)
    elif selected_page == "Dataset & EDA":
        render_dataset_eda(eda_res)
    elif selected_page == "Intrusion Detection":
        render_intrusion_detection(predictor, threshold)
    elif selected_page == "Model Performance":
        render_model_performance(training_res, threshold_res)
    elif selected_page == "Explainability (SHAP)":
        render_explainability(shap_summary)

if __name__ == "__main__":
    main()
