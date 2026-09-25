"""
Unit and integration tests for M2D File Parser (Condition Catcher format).
Verifies parsing integrity, metadata extraction, error handling, and exact numerical match
with dedicated tool export (.m2d.txt) across all 4 sample datasets (>2 million points).
"""

from datetime import datetime
import io
from pathlib import Path
import pytest
import numpy as np

from src.utils.m2d_parser import (
    parse_m2d_data,
    load_m2d_file,
    M2DData,
    _bcd_to_int,
    _parse_bcd_datetime,
    MAGIC_HEADER,
    HEADER_SIZE_BYTES,
    FOOTER_SIZE_BYTES,
)

DATA_DIR = Path("data/condition-chacher")

SAMPLE_FILES = [
    ("normal.m2d", 430080, 50000.0, datetime(2026, 9, 25, 18, 13, 24), datetime(2026, 9, 25, 18, 13, 33)),
    ("abnormal_ballFault.m2d", 458752, 50000.0, datetime(2026, 9, 25, 18, 15, 31), datetime(2026, 9, 25, 18, 15, 40)),
    ("abnormal_innerRaceFault.m2d", 495616, 50000.0, datetime(2026, 9, 25, 18, 14, 30), datetime(2026, 9, 25, 18, 14, 40)),
    ("abnormal_outerRaceFault.m2d", 702464, 50000.0, datetime(2026, 9, 25, 18, 16, 5), datetime(2026, 9, 25, 18, 16, 19)),
]


def test_bcd_helpers():
    """Test BCD decoding helpers."""
    assert _bcd_to_int(0x24) == 24
    assert _bcd_to_int(0x09) == 9
    assert _bcd_to_int(0x00) == 0
    assert _bcd_to_int(0x59) == 59

    # 24 sec, 13 min, 18 hour, 25 day, 9 month, 26 year
    raw = bytes([0x24, 0x13, 0x18, 0x25, 0x09, 0x26])
    dt = _parse_bcd_datetime(raw)
    assert dt == datetime(2026, 9, 25, 18, 13, 24)

    # Invalid length
    assert _parse_bcd_datetime(b"\x00") is None


@pytest.mark.parametrize("filename, expected_samples, expected_fs, exp_start, exp_end", SAMPLE_FILES)
def test_parse_m2d_sample_files(filename, expected_samples, expected_fs, exp_start, exp_end):
    """Test parsing of real-world .m2d files."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        pytest.skip(f"Sample data file {file_path} not found.")

    res = parse_m2d_data(file_path)

    assert isinstance(res, M2DData)
    assert res.total_samples == expected_samples
    assert len(res.data) == expected_samples
    assert res.sampling_frequency_hz == expected_fs
    assert res.channel_name == "Ch2"
    assert res.voltage_range == (-5.0, 5.0)
    assert res.start_datetime == exp_start
    assert res.end_datetime == exp_end
    assert len(res.file_hash) == 64  # SHA-256 hex string

    # Test load_m2d_file signature
    fs, data, file_hash = load_m2d_file(file_path)
    assert fs == expected_fs
    assert len(data) == expected_samples
    assert file_hash == res.file_hash


@pytest.mark.parametrize("filename, expected_samples, expected_fs, exp_start, exp_end", SAMPLE_FILES)
def test_m2d_vs_txt_numerical_exact_match(filename, expected_samples, expected_fs, exp_start, exp_end):
    """
    Validates that the M2D parser output matches the official tool's export (.m2d.txt)
    within the rounding tolerance (1e-6 V) across ALL samples.
    """
    m2d_path = DATA_DIR / filename
    txt_path = DATA_DIR / f"{filename}.txt"

    if not m2d_path.exists() or not txt_path.exists():
        pytest.skip(f"Data pair for {filename} not found.")

    # 1. Parse binary
    res = parse_m2d_data(m2d_path)
    binary_data = res.data

    # 2. Read text export values
    with open(txt_path, "r", encoding="cp932") as f:
        # Skip 7 header lines
        for _ in range(7):
            f.readline()
        txt_values = np.array([float(line.strip().split(",")[3]) for line in f])

    assert len(binary_data) == len(txt_values)
    assert len(binary_data) == expected_samples

    # 3. Verify maximum absolute difference is < 1e-6 V (official tool precision is 6 decimals)
    abs_diff = np.abs(binary_data - txt_values)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    assert max_diff < 1e-6, f"Max difference ({max_diff:.3e}) exceeds 1e-6 V"
    assert mean_diff < 5e-7, f"Mean difference ({mean_diff:.3e}) is too high"


def test_parse_m2d_from_bytes_and_stream():
    """Test that parser works with BytesIO (as in Streamlit upload)."""
    m2d_path = DATA_DIR / "normal.m2d"
    if not m2d_path.exists():
        pytest.skip("normal.m2d not found.")

    with open(m2d_path, "rb") as f:
        raw_bytes = f.read()

    # Test from bytes
    res_bytes = parse_m2d_data(raw_bytes)
    assert res_bytes.total_samples == 430080

    # Test from BytesIO
    stream = io.BytesIO(raw_bytes)
    res_stream = parse_m2d_data(stream)
    assert res_stream.total_samples == 430080
    assert np.allclose(res_bytes.data, res_stream.data)


def test_parse_m2d_invalid_inputs():
    """Test error handling for corrupt or invalid M2D payloads."""
    # Too small
    with pytest.raises(ValueError, match="too small"):
        parse_m2d_data(b"SHORT")

    # Invalid header magic
    bad_header = b"BAD_" + b"\x00" * (HEADER_SIZE_BYTES + FOOTER_SIZE_BYTES)
    with pytest.raises(ValueError, match="Invalid M2D file header"):
        parse_m2d_data(bad_header)

    # Odd length waveform payload
    odd_payload = MAGIC_HEADER + b"\x00" * (HEADER_SIZE_BYTES - len(MAGIC_HEADER)) + b"\x01" + b"\x00" * FOOTER_SIZE_BYTES
    with pytest.raises(ValueError, match="odd"):
        parse_m2d_data(odd_payload)
