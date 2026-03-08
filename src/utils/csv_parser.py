import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import re

def infer_vibration_columns(columns: List[str]) -> List[str]:
    """
    Infers potential vibration data columns from a list of column names.

    Args:
        columns: List of column names in the CSV.

    Returns:
        List[str]: List of inferred column names.
    """
    x_pattern = re.compile(r".*(x-axis|accel_x|ch1|vib_x|x_axis).*", re.IGNORECASE)
    y_pattern = re.compile(r".*(y-axis|accel_y|ch2|vib_y|y_axis).*", re.IGNORECASE)
    z_pattern = re.compile(r".*(z-axis|accel_z|ch3|vib_z|z_axis).*", re.IGNORECASE)

    detected = []
    for col in columns:
        if x_pattern.match(col) or y_pattern.match(col) or z_pattern.match(col):
            detected.append(col)
    
    return detected

def parse_csv_data(
    file_path: Path,
    data_columns: Union[str, List[str]],
    sampling_frequency_hz: Optional[float] = None,
    timestamp_column: Optional[str] = None,
    synthesize: bool = False
) -> Tuple[np.ndarray, float, Optional[dict[str, np.ndarray]]]:
    """
    Parses a CSV file containing vibration data from one or more columns.

    Args:
        file_path: Path to the CSV file.
        data_columns: Name(s) of the column(s) containing the acceleration data.
        sampling_frequency_hz: User-provided sampling frequency if no timestamp column.
        timestamp_column: Name of the column containing timestamps.
        synthesize: If True and multiple columns are provided, calculates the vector magnitude
                    sqrt(sum(col^2)) after removing DC offset from each.

    Returns:
        Tuple[np.ndarray, float, Optional[dict[str, np.ndarray]]]:
            - acceleration_data: NumPy array of acceleration values (possibly synthesized).
            - actual_sampling_frequency_hz: Actual or inferred sampling frequency.
            - column_data_map: Dictionary mapping column names to their original arrays (if multiple columns).
    """
    df = pd.read_csv(file_path, skipinitialspace=True)
    # Physical validity: Ensure column names are stripped of whitespace for robust matching
    df.columns = [c.strip() for c in df.columns]

    if isinstance(data_columns, str):
        data_columns = [data_columns]
    
    # Strip requested column names as well
    data_columns = [c.strip() for c in data_columns]

    for col in data_columns:
        if col not in df.columns:
            raise ValueError(f"Data column '{col}' not found in CSV file. Available: {list(df.columns)}")

    if timestamp_column:
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in CSV file.")
        
        # Determine if it's a Unix timestamp (numeric) or datetime string
        if pd.api.types.is_numeric_dtype(df[timestamp_column]):
            timestamps = pd.to_datetime(df[timestamp_column], unit='s', errors='coerce')
        else:
            # Use 'mixed' format for flexibility with various time strings
            timestamps = pd.to_datetime(df[timestamp_column], errors='coerce', format='mixed')

        if timestamps.isnull().any():
            raise ValueError("Could not parse all timestamps. Check format or missing values.")
        
        time_diffs = (timestamps - timestamps.iloc[0]).dt.total_seconds()
        total_duration = time_diffs.iloc[-1]
        
        # Infer sampling frequency
        if len(time_diffs) > 1:
            if total_duration > 0:
                # Use total duration and number of samples for more robustness
                # especially when timestamps have low precision (e.g. only 1 sec)
                inferred_fs = (len(df) - 1) / total_duration
                
                # Check consistency
                intervals = np.diff(time_diffs)
                if np.any(intervals == 0):
                    print("Note: Low-precision timestamps detected. Using total duration for FS inference.")
                elif np.std(intervals) / np.mean(intervals) > 0.05:
                    print("Warning: Inconsistent sampling intervals detected. Using average frequency.")
                
                actual_sampling_frequency_hz = inferred_fs
            else:
                raise ValueError("Total duration is zero. Cannot infer sampling frequency from timestamps.")
        else:
            raise ValueError("Not enough timestamps to infer sampling frequency.")

    else:
        if sampling_frequency_hz is None:
            raise ValueError("Sampling frequency must be provided if no timestamp column is specified.")
        actual_sampling_frequency_hz = sampling_frequency_hz
    
    # Process data columns
    processed_arrays = []
    column_data_map = {}
    for col in data_columns:
        col_data = df[col]
        # Ensure numeric
        if not np.issubdtype(col_data.dtype, np.number):
            col_data = pd.to_numeric(col_data, errors='coerce')
        
        arr = col_data.to_numpy()
        
        # Check for non-numeric or NaN values
        if np.isnan(arr).any():
            raise ValueError(f"Non-numeric or NaN data found in column '{col}'")
            
        if arr.size == 0:
            raise ValueError(f"No data found in column '{col}'.")
        
        processed_arrays.append(arr)
        column_data_map[col] = arr

    if synthesize and len(processed_arrays) > 1:
        # Physical validity: Remove DC (bias/gravity) from each axis before synthesis
        # to focus on the dynamic vibration magnitude.
        zero_mean_arrays = [a - np.mean(a) for a in processed_arrays]
        acceleration_data = np.sqrt(np.sum([a**2 for a in zero_mean_arrays], axis=0))
    else:
        # If not synthesizing or only one column, take the first one
        acceleration_data = processed_arrays[0]

    return acceleration_data, actual_sampling_frequency_hz, column_data_map if len(column_data_map) > 1 else None

