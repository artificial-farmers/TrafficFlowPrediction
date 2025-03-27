"""
Utility functions for data processing in Boroondara Traffic Flow Prediction System.
"""
import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Dict, Optional

from data.boroondara_preprocessing import (
    prepare_boroondara_dataset,
    load_scats_data,
    prepare_site_data,
    get_available_scats_sites
)


def setup_data_directories(processed_dir: str) -> None:
    """Create necessary data directories if they don't exist.

    Args:
        processed_dir: Base directory for processed data
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "per_site"), exist_ok=True)


def preprocess_data(
    scats_data_path: str,
    metadata_path: str,
    processed_dir: str,
    seq_length: int,
    force_preprocess: bool = False,
) -> None:
    """Preprocess the Boroondara SCATS data.

    Args:
        scats_data_path: Path to SCATS data CSV file
        metadata_path: Path to SCATS metadata CSV file
        processed_dir: Directory to save processed data
        seq_length: Sequence length for prediction
        force_preprocess: Whether to force preprocessing even if data exists
    """
    print("\n" + "="*50)
    print("Preprocessing Boroondara Data")
    print("="*50)

    # Create directories
    setup_data_directories(processed_dir)

    # Check if data already exists and we're not forcing reprocessing
    if not force_preprocess and os.path.exists(os.path.join(processed_dir, "per_site", "site_info.csv")):
        print("Processed data already exists. Use --force_preprocess to recreate.")
        return

    # Process data for individual sites
    prepare_boroondara_dataset(
        scats_data_path,
        metadata_path,
        processed_dir,
        seq_length,
        per_site=True
    )

    print("\nSite-specific data preprocessing completed.")


def prepare_combined_dataset(
    scats_data_path: str,
    metadata_path: str,
    processed_dir: str,
    seq_length: int,
    force_preprocess: bool = False,
) -> None:
    """Prepare a combined dataset from all sites.

    Args:
        scats_data_path: Path to SCATS data CSV file
        metadata_path: Path to SCATS metadata CSV file
        processed_dir: Directory to save processed data
        seq_length: Sequence length for prediction
        force_preprocess: Whether to force preprocessing even if data exists
    """
    combined_data_path = os.path.join(processed_dir, "X_train.npy")

    # Check if data already exists and we're not forcing reprocessing
    if not force_preprocess and os.path.exists(combined_data_path):
        print("Combined dataset already exists. Use --force_preprocess to recreate.")
        return

    print("\nCreating combined dataset from all sites...")

    # Process combined data
    prepare_boroondara_dataset(
        scats_data_path,
        metadata_path,
        processed_dir,
        seq_length,
        per_site=False
    )

    print("\nCombined dataset preprocessing completed.")


def load_site_data(
    site_id: str,
    processed_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Load data for a specific site.

    Args:
        site_id: SCATS site ID
        processed_dir: Directory containing processed data

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler)
    """
    # Ensure site_id is a string
    site_id = str(site_id)
    site_dir = os.path.join(processed_dir, "per_site", site_id)

    if not os.path.exists(site_dir):
        raise FileNotFoundError(f"No processed data found for site {site_id}")

    X_train = np.load(os.path.join(site_dir, "X_train.npy"))
    y_train = np.load(os.path.join(site_dir, "y_train.npy"))
    X_test = np.load(os.path.join(site_dir, "X_test.npy"))
    y_test = np.load(os.path.join(site_dir, "y_test.npy"))
    scaler = joblib.load(os.path.join(site_dir, "scaler.joblib"))

    return X_train, y_train, X_test, y_test, scaler


def load_combined_data(
    processed_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the combined dataset.

    Args:
        processed_dir: Directory containing processed data

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    return X_train, y_train, X_test, y_test


def get_available_sites(processed_dir: str) -> List[str]:
    """Get list of available sites with processed data.

    Args:
        processed_dir: Directory containing processed data

    Returns:
        List of site IDs
    """
    site_info_path = os.path.join(processed_dir, "per_site", "site_info.csv")

    if os.path.exists(site_info_path):
        site_info = pd.read_csv(site_info_path)
        # Ensure site IDs are strings
        return [str(site_id) for site_id in site_info["site_id"].tolist()]
    else:
        # Fallback: Get directories from processed/per_site
        per_site_dir = os.path.join(processed_dir, "per_site")
        if os.path.exists(per_site_dir):
            return [str(d) for d in os.listdir(per_site_dir)
                   if os.path.isdir(os.path.join(per_site_dir, d))]
        return []


def get_site_location(site_id: str, processed_dir: str) -> Optional[str]:
    """Get the location description for a site.

    Args:
        site_id: SCATS site ID
        processed_dir: Directory containing processed data

    Returns:
        Location description or None if not found
    """
    # Ensure site_id is a string
    site_id = str(site_id)
    site_info_path = os.path.join(processed_dir, "per_site", "site_info.csv")

    if os.path.exists(site_info_path):
        site_info = pd.read_csv(site_info_path)
        site_row = site_info[site_info["site_id"] == site_id]
        if not site_row.empty:
            return site_row["location"].iloc[0]

    return None


def reshape_for_rnn(X: np.ndarray) -> np.ndarray:
    """Reshape input data for recurrent neural networks.

    Args:
        X: Input data

    Returns:
        Reshaped input data
    """
    return X.reshape(X.shape[0], X.shape[1], 1)
