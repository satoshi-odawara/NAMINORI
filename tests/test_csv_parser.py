import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.csv_parser import parse_csv_data

# Fixture to create dummy CSV files
@pytest.fixture
def create_dummy_csv(tmp_path):
    def _creator(filename: str, content: str):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _creator

def test_parse_csv_no_timestamp_explicit_sf(create_dummy_csv):
    content = """accel_data
1.0
2.0
3.0
4.0
"""
    file_path = create_dummy_csv("data.csv", content)
    data, fs = parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)
    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0, 3.0, 4.0]))
    assert fs == 100.0

def test_parse_csv_with_timestamp_inferred_sf(create_dummy_csv):
    content = """timestamp,accel_data
2023-01-01 10:00:00.000,1.0
2023-01-01 10:00:00.010,2.0
2023-01-01 10:00:00.020,3.0
2023-01-01 10:00:00.030,4.0
"""
    file_path = create_dummy_csv("data.csv", content)
    data, fs = parse_csv_data(file_path, "accel_data", timestamp_column="timestamp")
    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isclose(fs, 100.0) # 10 ms interval -> 100 Hz

def test_parse_csv_with_unix_timestamp_inferred_sf(create_dummy_csv):
    content = """unix_ts,accel_data
1672573200.000,1.0
1672573200.001,2.0
1672573200.002,3.0
1672573200.003,4.0
"""
    file_path = create_dummy_csv("data.csv", content)
    data, fs = parse_csv_data(file_path, "accel_data", timestamp_column="unix_ts")
    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isclose(fs, 1000.0) # 1 ms interval -> 1000 Hz

def test_parse_csv_missing_data_column(create_dummy_csv):
    content = """col1,col2
1,A
2,B
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Data column 'accel_data' not found"):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)

def test_parse_csv_missing_timestamp_column(create_dummy_csv):
    content = """accel_data,col2
1.0,A
2.0,B
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Timestamp column 'timestamp' not found"):
        parse_csv_data(file_path, "accel_data", timestamp_column="timestamp")

def test_parse_csv_non_numeric_acceleration(create_dummy_csv):
    content = """accel_data
1.0
abc
3.0
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Acceleration data column contains non-numeric values"):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)

def test_parse_csv_no_sf_and_no_timestamp(create_dummy_csv):
    content = """accel_data
1.0
2.0
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Sampling frequency must be provided"):
        parse_csv_data(file_path, "accel_data")

def test_parse_csv_inconsistent_sampling_warning(create_dummy_csv, capsys):
    content = """timestamp,accel_data
2023-01-01 10:00:00.000,1.0
2023-01-01 10:00:00.010,2.0
2023-01-01 10:00:00.030,3.0
2023-01-01 10:00:00.040,4.0
""" # Interval 10ms, 20ms, 10ms
    file_path = create_dummy_csv("data.csv", content)
    data, fs = parse_csv_data(file_path, "accel_data", timestamp_column="timestamp")
    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isclose(fs, 75.0) # Average interval is (10+20+10)/3 = 40/3 ms, so (3/0.04) = 75 Hz
    captured = capsys.readouterr()
    assert "Warning: Inconsistent sampling intervals detected." in captured.out

def test_parse_csv_non_increasing_timestamps(create_dummy_csv):
    content = """timestamp,accel_data
2023-01-01 10:00:00.010,2.0
2023-01-01 10:00:00.000,1.0
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Timestamps are not increasing"):
        parse_csv_data(file_path, "accel_data", timestamp_column="timestamp")

def test_parse_csv_single_timestamp(create_dummy_csv):
    content = """timestamp,accel_data
2023-01-01 10:00:00.000,1.0
"""
    file_path = create_dummy_csv("data.csv", content)
    with pytest.raises(ValueError, match="Not enough timestamps to infer sampling frequency."):
        parse_csv_data(file_path, "accel_data", timestamp_column="timestamp")

def test_parse_csv_empty_file(create_dummy_csv):
    content = ""
    file_path = create_dummy_csv("empty.csv", content)
    with pytest.raises(pd.errors.EmptyDataError):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)

def test_parse_csv_only_headers(create_dummy_csv):
    content = "accel_data"
    file_path = create_dummy_csv("headers.csv", content)
    with pytest.raises(ValueError, match="No acceleration data found in the specified column."):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)