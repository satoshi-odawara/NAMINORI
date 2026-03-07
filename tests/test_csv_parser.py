import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.csv_parser import parse_csv_data, infer_vibration_columns

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
    with pytest.raises(ValueError, match="Non-numeric or NaN data found in column 'accel_data'"):
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
    with pytest.raises(ValueError, match="No data found in column 'accel_data'"):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)

def test_parse_csv_with_leading_spaces_in_header(create_dummy_csv):
    content = "timestamp, accel_data, other_col\n2023-01-01 10:00:00, 1.0, A\n2023-01-01 10:00:01, 2.0, B"
    file_path = create_dummy_csv("leading_spaces.csv", content)
    # Should work even if we pass "accel_data" without space because of skipinitialspace=True
    data, fs = parse_csv_data(file_path, "accel_data", sampling_frequency_hz=1.0)
    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0]))
    assert fs == 1.0

def test_parse_csv_multi_axis_synthesis(create_dummy_csv):
    # 3-axis data with different DC offsets
    # x: sin + 10.0 DC
    # y: cos + 5.0 DC
    # z: 0.0 + 2.0 DC (static)
    t = np.arange(0, 1, 0.1)
    x = np.sin(2 * np.pi * 1 * t) + 10.0
    y = np.cos(2 * np.pi * 1 * t) + 5.0
    z = np.zeros_like(t) + 2.0
    
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    file_path = create_dummy_csv("multi_axis.csv", df.to_csv(index=False))
    
    # Synthesize should remove DC offsets (10, 5, 2) and return sqrt(sin^2 + cos^2 + 0^2) = 1.0
    data, fs = parse_csv_data(file_path, data_columns=['x', 'y', 'z'], sampling_frequency_hz=10.0, synthesize=True)
    
    np.testing.assert_array_almost_equal(data, np.ones_like(t), decimal=5)
    assert fs == 10.0

def test_parse_csv_with_nan_values(create_dummy_csv):
    content = """accel_data
1.0
NaN
3.0
"""
    file_path = create_dummy_csv("data_nan.csv", content)
    with pytest.raises(ValueError, match="Non-numeric or NaN data found in column 'accel_data'"):
        parse_csv_data(file_path, "accel_data", sampling_frequency_hz=100.0)

def test_parse_csv_inconsistent_row_length(create_dummy_csv):
    # Pandas will typically pad with NaN or handle this, let's see how our numeric check behaves
    content = """col1,col2
1.0,2.0
3.0
4.0,5.0
"""
    file_path = create_dummy_csv("bad_rows.csv", content)
    # col2 will have a NaN in the middle row
    with pytest.raises(ValueError, match="Non-numeric or NaN data found in column 'col2'"):
        parse_csv_data(file_path, "col2", sampling_frequency_hz=100.0)

def test_infer_vibration_columns():
    cols = ["timestamp", "X-axis", "Y-Axis", "Z_axis", "temp", "Other"]
    inferred = infer_vibration_columns(cols)
    assert "X-axis" in inferred
    assert "Y-Axis" in inferred
    assert "Z_axis" in inferred
    assert "timestamp" not in inferred
    assert "temp" not in inferred

    cols = ["ch1", "CH2", "vibration_3", "accel_x", "vib_z"]
    inferred = infer_vibration_columns(cols)
    assert "ch1" in inferred
    assert "CH2" in inferred
    assert "accel_x" in inferred
    assert "vib_z" in inferred

def test_infer_vibration_columns_no_match():
    cols = ["date", "time", "pressure", "humidity"]
    assert infer_vibration_columns(cols) == []
