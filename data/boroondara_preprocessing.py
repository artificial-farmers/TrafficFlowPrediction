"""
Data processing utilities specifically for the Boroondara traffic dataset.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler


def load_scats_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the SCATS data from October 2006.

    Args:
        file_path: Path to the SCATS data CSV file

    Returns:
        Cleaned DataFrame with traffic flow data
    """
    # Load the raw data with correct header row (second row)
    df = pd.read_csv(file_path, encoding='utf-8', header=1)

    # Create a cleaned DataFrame
    cleaned_data = []

    # Create a list of time columns (V00 to V95)
    time_columns = [f'V{str(i).zfill(2)}' for i in range(96)]

    for _, row in df.iterrows():
        # Skip rows without SCATS number or date
        if pd.isna(row['SCATS Number']) or pd.isna(row['Date']):
            continue

        site = row['SCATS Number']
        try:
            date = pd.to_datetime(row['Date'], format='%m/%d/%Y')
        except:
            # Skip rows with invalid dates
            continue

        # Process each 15-minute interval
        for i, col in enumerate(time_columns):
            if col not in df.columns:
                continue

            # Calculate the timestamp
            hours = i // 4
            minutes = (i % 4) * 15
            timestamp = pd.Timestamp(
                date.year, date.month, date.day, hours, minutes)

            # Get traffic flow value
            flow = pd.to_numeric(row[col], errors='coerce')

            # Add to cleaned data
            cleaned_data.append({
                'SCATS_Site': site,
                'Timestamp': timestamp,
                'Traffic_Flow': flow
            })

    # Create the final DataFrame
    result_df = pd.DataFrame(cleaned_data)

    # Handle missing values
    result_df['Traffic_Flow'] = result_df['Traffic_Flow'].fillna(0)

    return result_df


def prepare_site_data(df: pd.DataFrame, site_id: str, seq_length: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare training and testing data for a specific SCATS site.

    Args:
        df: DataFrame with traffic flow data
        site_id: SCATS site ID to filter by
        seq_length: Number of time steps to use for sequence prediction

    Returns:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        scaler: Fitted MinMaxScaler
    """
    # Filter data for the specified site
    site_data = df[df['SCATS_Site'] == site_id].sort_values('Timestamp')

    if len(site_data) == 0:
        raise ValueError(f"No data found for site {site_id}")

    if len(site_data) < seq_length + 10:
        raise ValueError(
            f"Not enough data points for site {site_id}: {len(site_data)} found, need at least {seq_length + 10}")

    print(f"Processing site {site_id} with {len(site_data)} data points")

    # Extract flow values
    flow_values = site_data['Traffic_Flow'].values

    # Check for missing or invalid values
    if np.isnan(flow_values).any():
        print(
            f"Warning: Site {site_id} has {np.isnan(flow_values).sum()} NaN values that will be treated as zeros")
        flow_values = np.nan_to_num(flow_values)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_flow = scaler.fit_transform(flow_values.reshape(-1, 1)).flatten()

    # Create sequences
    sequences = []
    for i in range(len(scaled_flow) - seq_length):
        sequences.append(scaled_flow[i:i+seq_length+1])

    # Convert to numpy array
    sequences = np.array(sequences)

    if len(sequences) == 0:
        raise ValueError(f"Failed to create sequences for site {site_id}")

    # Split into train (70%) and test (30%)
    split_idx = int(len(sequences) * 0.7)

    # Ensure we have at least one sample in each set
    if split_idx == 0:
        split_idx = 1
    elif split_idx == len(sequences):
        split_idx = len(sequences) - 1

    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]

    # Shuffle training data
    np.random.shuffle(train_data)

    # Split sequences into features (X) and targets (y)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    print(
        f"Site {site_id} - Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    return X_train, y_train, X_test, y_test, scaler


def prepare_multi_site_data(df: pd.DataFrame, site_ids: List[str], seq_length: int = 12) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]]:
    """
    Prepare training and testing data for multiple SCATS sites.

    Args:
        df: DataFrame with traffic flow data
        site_ids: List of SCATS site IDs to process
        seq_length: Number of time steps to use for sequence prediction

    Returns:
        Dictionary mapping site IDs to their respective data tuples
    """
    result = {}
    success_count = 0
    error_count = 0

    for site_id in site_ids:
        try:
            result[site_id] = prepare_site_data(df, site_id, seq_length)
            success_count += 1
        except ValueError as e:
            print(f"Warning: {e}")
            error_count += 1
        except Exception as e:
            print(f"Error processing site {site_id}: {str(e)}")
            error_count += 1

    print(
        f"Successfully processed {success_count} sites with {error_count} errors")

    if not result:
        print("Warning: No sites were successfully processed")

    return result


def load_scats_metadata(file_path: str) -> pd.DataFrame:
    """
    Load SCATS site metadata.

    Args:
        file_path: Path to the SCATS site listing CSV file

    Returns:
        DataFrame with SCATS site information
    """
    # Load the raw data, skipping the metadata rows (first 9 rows)
    # The actual header is on row 10
    df = pd.read_csv(file_path, encoding='utf-8', skiprows=9)

    # Rename columns to more standardized names
    column_names = ['Site_Number', 'Location_Description', 'Site_Type', 'Directory',
                    'Map_Reference', 'Unknown1', 'Unknown2', 'Unknown3', 'Unknown4']

    # Ensure we have enough columns
    if len(df.columns) < len(column_names):
        # If not enough columns, extend with NaN columns
        for i in range(len(df.columns), len(column_names)):
            df[f'Col{i}'] = np.nan

    # Rename the columns
    df.columns = column_names[:len(df.columns)]

    # Filter out rows without site numbers
    df = df[df['Site_Number'].notna()]

    # Convert site numbers to string
    df['Site_Number'] = df['Site_Number'].astype(str)

    return df


def get_boroondara_sites(metadata_df: pd.DataFrame) -> List[str]:
    """
    Extract SCATS site IDs for Boroondara area based on metadata.

    Args:
        metadata_df: DataFrame with SCATS site metadata

    Returns:
        List of SCATS site IDs in the Boroondara area
    """
    # In a real implementation, this would filter based on location information
    # For now, we'll use a simplified approach based on the PDF map
    boroondara_sites = [
        '2000', '2034', '2041', '2042', '2044', '2200', '2820', '2821', '2822',
        '2823', '2824', '2825', '2826', '2827', '2829', '2831', '2832', '2839',
        '2842', '2847', '3000', '3002', '3003', '3004', '3007', '3008', '3120',
        '3121', '3123', '3124', '3125', '3127', '3128', '3129', '3180', '3181',
        '3620', '3621', '3622', '3660', '3661', '3662', '3663', '3664', '3667',
        '3682', '3798', '3799', '3800', '3801', '3802', '3803', '3805', '3806',
        '3807', '3808', '3809', '3811', '3813', '3814', '3815', '3816', '3817',
        '3818', '3819', '3820', '3821', '3822', '3823', '3824', '3826', '3827',
        '3828', '3829', '3837', '3914', '3977', '4031', '4032', '4033', '4037',
        '4039', '4042', '4043', '4044', '4045', '4046', '4049', '4050', '4051',
        '4052', '4053', '4054', '4055', '4056', '4057', '4058', '4059', '4060',
        '4061', '4062', '4064', '4065', '4066', '4067', '4068', '4069', '4260',
        '4261', '4262', '4263', '4265', '4266', '4268', '4269', '4270', '4274',
        '4275', '4276', '4277', '4278', '4279', '4281', '4282', '4283', '4284',
        '4286', '4287', '4289', '4320', '4321', '4322', '4324', '4325', '4330',
        '4331', '4333', '4335', '4336', '4339', '7003'
    ]

    return boroondara_sites


def prepare_boroondara_dataset(scats_data_path: str, metadata_path: str, output_dir: str, seq_length: int = 12) -> None:
    """
    Prepare training and testing datasets for Boroondara SCATS sites.

    Args:
        scats_data_path: Path to SCATS data CSV file
        metadata_path: Path to SCATS metadata CSV file
        output_dir: Directory to save processed data
        seq_length: Number of time steps to use for sequence prediction
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load data
        print("Loading SCATS traffic data...")
        scats_df = load_scats_data(scats_data_path)
        print(f"Loaded {len(scats_df)} records from SCATS data file")

        print("Loading SCATS metadata...")
        metadata_df = load_scats_metadata(metadata_path)
        print(f"Loaded {len(metadata_df)} sites from metadata file")

        # Get Boroondara sites
        boroondara_sites = get_boroondara_sites(metadata_df)
        print(f"Found {len(boroondara_sites)} potential Boroondara sites")

        # Filter for available sites
        available_sites = scats_df['SCATS_Site'].unique()
        print(
            f"Found {len(available_sites)} sites with data in the SCATS dataset")

        valid_sites = [
            site for site in boroondara_sites if site in available_sites]

        if not valid_sites:
            print("Warning: No exact matches between Boroondara sites and available data")
            print("Using all available sites from the SCATS data")
            valid_sites = available_sites

        print(f"Processing {len(valid_sites)} valid sites")

        # Take a subset if too many sites (to speed up development)
        if len(valid_sites) > 10:
            print(f"Limiting to 10 sites for faster processing")
            valid_sites = valid_sites[:10]

        # Prepare data for each site
        site_data = prepare_multi_site_data(scats_df, valid_sites, seq_length)

        # Create a consolidated dataset
        all_X_train, all_y_train = [], []
        all_X_test, all_y_test = [], []

        for site, (X_train, y_train, X_test, y_test, _) in site_data.items():
            all_X_train.append(X_train)
            all_y_train.append(y_train)
            all_X_test.append(X_test)
            all_y_test.append(y_test)

        # Combine data from all sites
        if all_X_train:
            X_train_combined = np.vstack(all_X_train)
            y_train_combined = np.concatenate(all_y_train)
            X_test_combined = np.vstack(all_X_test)
            y_test_combined = np.concatenate(all_y_test)

            # Shuffle the training data
            shuffle_idx = np.random.permutation(len(X_train_combined))
            X_train_combined = X_train_combined[shuffle_idx]
            y_train_combined = y_train_combined[shuffle_idx]

            # Save the datasets
            np.save(os.path.join(output_dir, 'X_train.npy'), X_train_combined)
            np.save(os.path.join(output_dir, 'y_train.npy'), y_train_combined)
            np.save(os.path.join(output_dir, 'X_test.npy'), X_test_combined)
            np.save(os.path.join(output_dir, 'y_test.npy'), y_test_combined)

            print(f"Datasets saved to {output_dir}")
            print(f"X_train shape: {X_train_combined.shape}")
            print(f"y_train shape: {y_train_combined.shape}")
            print(f"X_test shape: {X_test_combined.shape}")
            print(f"y_test shape: {y_test_combined.shape}")
        else:
            print("No data to save - check your data files and site IDs")
    except Exception as e:
        print(f"Error in data preparation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    scats_data_path = "data/raw/Scats Data October 2006.csv"
    metadata_path = "data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv"
    output_dir = "data/processed/boroondara"

    prepare_boroondara_dataset(scats_data_path, metadata_path, output_dir)
