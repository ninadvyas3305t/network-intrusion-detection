"""Unit tests for data preprocessing module."""

import os
import pytest
import pandas as pd
from src.data_preprocessing import (
    clean_dataframe,
    get_data_summary,
    DROP_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_BINARY,
    TARGET_MULTICLASS,
)

@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic sample dataframe mirroring UNSW-NB15 schema."""
    data = {
        "id": [1, 2, 3],
        "dur": [0.12, 0.05, 1.2],
        "proto": ["TCP ", "udp", " tcp"],
        "service": ["http", "-", "dns "],
        "state": ["FIN", "CON", "int"],
        "spkts": [10, 4, 25],
        "dpkts": [8, 0, 30],
        "sbytes": [1024, 256, 4096],
        "dbytes": [2048, 0, 8192],
        "rate": [50.0, 10.0, 120.0],
        "sttl": [64, 32, 64],
        "dttl": [60, 0, 60],
        "attack_cat": ["Normal", "DoS", "Backdoors"],
        "label": [0, 1, 1],
    }
    return pd.DataFrame(data)

def test_clean_dataframe_drops_id(sample_raw_dataframe):
    """Ensure ID column is removed to prevent data leakage."""
    cleaned = clean_dataframe(sample_raw_dataframe)
    assert "id" not in cleaned.columns
    for drop_col in DROP_COLUMNS:
        assert drop_col not in cleaned.columns

def test_clean_dataframe_standardizes_strings(sample_raw_dataframe):
    """Ensure string columns are trimmed and lowercased."""
    cleaned = clean_dataframe(sample_raw_dataframe)
    # proto should be trimmed and lowercased
    assert list(cleaned["proto"]) == ["tcp", "udp", "tcp"]
    # service '-' should become 'none'
    assert list(cleaned["service"]) == ["http", "none", "dns"]
    # state should be lowercased
    assert list(cleaned["state"]) == ["fin", "con", "int"]
    # attack_cat 'Backdoors' should become 'Backdoor'
    assert list(cleaned["attack_cat"]) == ["Normal", "DoS", "Backdoor"]

def test_get_data_summary(sample_raw_dataframe):
    """Ensure summary statistics calculation works accurately."""
    cleaned = clean_dataframe(sample_raw_dataframe)
    summary = get_data_summary(cleaned)
    assert summary["total_records"] == 3
    assert summary["missing_values"] == 0
    assert summary["binary_distribution"][0] == 1
    assert summary["binary_distribution"][1] == 2
