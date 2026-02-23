import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

def parse_csv_data(
    file_path: Path,
    data_column: str,
    sampling_frequency_hz: Optional[float] = None,
    timestamp_column: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Parses a CSV file containing vibration data.

    Args:
        file_path: Path to the CSV file.
        data_column: Name of the column containing the acceleration data.
        sampling_frequency_hz: User-provided sampling frequency if no timestamp column.
        timestamp_column: Name of the column containing timestamps.

    Returns:
        Tuple[np.ndarray, float]:
            - acceleration_data: NumPy array of acceleration values.
            - actual_sampling_frequency_hz: Actual or inferred sampling frequency.
    """
    df = pd.read_csv(file_path)

    if data_column not in df.columns:
        raise ValueError(f"Data column '{data_column}' not found in CSV file.")

    acceleration_data = df[data_column]

    if timestamp_column:
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in CSV file.")
        
        # Determine if it's a Unix timestamp (numeric) or datetime string
        if pd.api.types.is_numeric_dtype(df[timestamp_column]):
            timestamps = pd.to_datetime(df[timestamp_column], unit='s', errors='coerce')
        else:
            timestamps = pd.to_datetime(df[timestamp_column], errors='coerce')

        if timestamps.isnull().any():
            raise ValueError("Could not parse all timestamps. Check format or missing values.")
        
        time_diffs = (timestamps - timestamps.iloc[0]).dt.total_seconds()
        
        # Infer sampling frequency from time differences
        if len(time_diffs) > 1:
            mean_interval = np.mean(np.diff(time_diffs))
            if mean_interval > 0:
                inferred_fs = 1.0 / mean_interval
                # Check for significant variations in sampling frequency
                if np.std(np.diff(time_diffs)) / mean_interval > 0.05: # More than 5% variation
                    print("Warning: Inconsistent sampling intervals detected. Using average frequency.")
                actual_sampling_frequency_hz = inferred_fs
            else:
                raise ValueError("Timestamps are not increasing, cannot infer sampling frequency.")
        else:
            raise ValueError("Not enough timestamps to infer sampling frequency.")

    else:
        if sampling_frequency_hz is None:
            raise ValueError("Sampling frequency must be provided if no timestamp column is specified.")
        actual_sampling_frequency_hz = sampling_frequency_hz
    
    # Ensure acceleration_data is numeric
    if not np.issubdtype(acceleration_data.dtype, np.number):
        try:
            acceleration_data = pd.to_numeric(acceleration_data, errors='coerce').to_numpy()
            if np.isnan(acceleration_data).any():
                raise ValueError("Non-numeric data found in acceleration column after conversion.")
        except ValueError:
            raise ValueError("Acceleration data column contains non-numeric values.")
    else:
        acceleration_data = acceleration_data.to_numpy() # Convert Series to numpy array if it was numeric initially

    if acceleration_data.size == 0:
        raise ValueError("No acceleration data found in the specified column.")

    return acceleration_data, actual_sampling_frequency_hz

