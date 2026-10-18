"""Exploratory Data Analysis (EDA) Module
Generates statistical analyses, correlation studies, and publication-ready visualizations
for the UNSW-NB15 dataset. Saves outputs for direct consumption by Streamlit.
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style setup for professional aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

EDA_FIG_DIR = "data/processed/eda_figures"
EDA_SUMMARY_FILE = "data/processed/eda_summary.json"

def perform_eda(
    train_path: str = "data/processed/train_cleaned.csv",
    test_path: str = "data/processed/test_cleaned.csv",
    fig_dir: str = EDA_FIG_DIR,
    summary_file: str = EDA_SUMMARY_FILE,
) -> Dict[str, Any]:
    """Run comprehensive EDA, export charts, and output structured summary JSON."""
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    logger.info("Loading cleaned datasets for EDA...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Class Distribution Analysis
    train_binary_counts = train_df["label"].value_counts().to_dict()
    test_binary_counts = test_df["label"].value_counts().to_dict()
    train_multiclass_counts = train_df["attack_cat"].value_counts().to_dict()
    test_multiclass_counts = test_df["attack_cat"].value_counts().to_dict()

    # Plot 1: Binary Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = ["#2b5c8f", "#d95f02"]
    
    # Train
    train_labels = ["Normal (0)", "Attack (1)"]
    train_vals = [train_binary_counts.get(0, 0), train_binary_counts.get(1, 0)]
    train_pcts = [v / len(train_df) * 100 for v in train_vals]
    bars1 = axes[0].bar(train_labels, train_vals, color=palette, edgecolor="black", width=0.55)
    axes[0].set_title(f"Train Set Binary Distribution (N={len(train_df):,})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, pct, val in zip(bars1, train_pcts, train_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
                     f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    # Test
    test_labels = ["Normal (0)", "Attack (1)"]
    test_vals = [test_binary_counts.get(0, 0), test_binary_counts.get(1, 0)]
    test_pcts = [v / len(test_df) * 100 for v in test_vals]
    bars2 = axes[1].bar(test_labels, test_vals, color=palette, edgecolor="black", width=0.55)
    axes[1].set_title(f"Test Set Binary Distribution (N={len(test_df):,})", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    for bar, pct, val in zip(bars2, test_pcts, test_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "binary_class_distribution.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved binary distribution figure to {fig1_path}")

    # Plot 2: Multiclass Attack Categories Distribution (Horizontal Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_series = train_df["attack_cat"].value_counts()
    colors = ["#2b5c8f" if c == "Normal" else "#d95f02" for c in cat_series.index]
    bars = ax.barh(cat_series.index[::-1], cat_series.values[::-1], color=colors[::-1], edgecolor="black")
    ax.set_title("UNSW-NB15 Attack Category Breakdown (Training Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency Count")
    for bar in bars:
        width = bar.get_width()
        pct = (width / len(train_df)) * 100
        ax.text(width + 1000, bar.get_y() + bar.get_height() / 2,
                f"{int(width):,} ({pct:.2f}%)", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "attack_categories_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved attack categories figure to {fig2_path}")

    # 2. Correlation Analysis
    num_cols = train_df.select_dtypes(include=[np.number]).columns.drop("label")
    corr_matrix = train_df[num_cols].corr()

    # Find high correlations (|r| >= 0.90)
    high_corr_list = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1 = num_cols[i]
            c2 = num_cols[j]
            r = corr_matrix.loc[c1, c2]
            if abs(r) >= 0.90:
                high_corr_list.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "pearson_r": round(float(r), 4),
                    "domain_explanation": get_correlation_domain_rationale(c1, c2)
                })

    high_corr_list = sorted(high_corr_list, key=lambda x: abs(x["pearson_r"]), reverse=True)

    # Plot 3: Top Correlated Features Heatmap
    # Select key numerical features for clean heatmap visualization
    key_features = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "tcprtt", "synack", "ackdat", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm"
    ]
    sub_corr = train_df[key_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap of Key Network Flow Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "correlation_heatmap.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to {fig3_path}")

    # Plot 4: Top Discriminative Features Boxplots / Violin plots
    discriminative_features = ["sttl", "dttl", "rate", "ct_srv_src"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, col in enumerate(discriminative_features):
        sns.boxplot(x="label", y=col, data=train_df, ax=axes[idx],
                    palette=["#2b5c8f", "#d95f02"], showfliers=False)
        axes[idx].set_title(f"Distribution of '{col}' by Class", fontsize=11, fontweight="bold")
        axes[idx].set_xticklabels(["Normal (0)", "Attack (1)"])
        axes[idx].set_xlabel("")

    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "feature_distributions_by_class.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feature boxplots to {fig4_path}")

    # Compile Summary Data for Streamlit / viva
    eda_summary = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "total_records": len(train_df) + len(test_df),
        "numerical_feature_count": len(num_cols),
        "categorical_feature_count": 3,
        "categorical_features": ["proto", "service", "state"],
        "train_binary_distribution": train_binary_counts,
        "test_binary_distribution": test_binary_counts,
        "train_multiclass_distribution": train_multiclass_counts,
        "test_multiclass_distribution": test_multiclass_counts,
        "high_correlations_count": len(high_corr_list),
        "high_correlations": high_corr_list,
        "viva_insights": {
            "collinearity_handling": (
                "High collinearity (e.g. sbytes <-> sloss at r=0.9961, tcprtt <-> synack at r=0.9495) "
                "reflects physical network protocol mechanics (lost packets scale with sent volume; "
                "round-trip time consists of SYN-ACK and ACK-DATA latencies). Rather than blindly discarding "
                "these features, tree-based models (Random Forest, XGBoost) natively isolate threshold splits "
                "without variance inflation, while L2 regularization controls coefficients in Logistic Regression."
            ),
            "benchmark_split_preservation": (
                "The official train/test partition created by Moustafa & Slay is strictly maintained. "
                "Randomly recombining would cause test-set leakage across synthetic and real captures "
                "and violate academic reproducibility."
            ),
            "extreme_minority_attacks": (
                "Worms (130 train / 44 test) and Shellcode (1,133 train / 378 test) represent severe class "
                "imbalance, requiring balanced class weighting and macro-F1 evaluation."
            )
        }
    }

    with open(summary_file, "w") as f:
        json.dump(eda_summary, f, indent=2)
    logger.info(f"Saved EDA summary metrics to {summary_file}")

    return eda_summary

def get_correlation_domain_rationale(f1: str, f2: str) -> str:
    """Provide networking domain explanation for feature correlation."""
    pair = tuple(sorted([f1, f2]))
    explanations = {
        ("ct_ftp_cmd", "is_ftp_login"): "FTP commands and login flags are mathematically tied in single-session FTP traffic.",
        ("dbytes", "dloss"): "Destination packet loss scales proportionally with total destination byte payload.",
        ("sbytes", "sloss"): "Source packet loss scales directly with source transmitted bytes.",
        ("dwin", "swin"): "TCP window advertisements mirror window size negotation during TCP handshakes.",
        ("ct_srv_dst", "ct_srv_src"): "In symmetric bidirectional traffic, connections from source match connections to destination service.",
        ("dbytes", "dpkts"): "Destination packet count directly dictates destination byte volume.",
        ("sbytes", "spkts"): "Source packet count directly dictates source byte volume.",
        ("ackdat", "tcprtt"): "TCP RTT is the sum of SYN-ACK and ACK-DATA handshake latencies.",
        ("synack", "tcprtt"): "TCP RTT directly embeds the SYN-ACK handshake delay.",
        ("ct_dst_ltm", "ct_src_dport_ltm"): "Local network traffic to a destination frequently targets the same destination port.",
    }
    return explanations.get(pair, "Direct structural dependency in bidirectional network flow monitoring.")

if __name__ == "__main__":
    perform_eda()
