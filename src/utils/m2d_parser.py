"""
M2D File Parser Module for Condition Catcher series vibration data loggers.

Provides high-accuracy binary parsing for .m2d files into physical voltage arrays
and associated metadata (sampling frequency, channels, timestamps, ranges).
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
from pathlib import Path
import struct
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np

# Header and Footer Constants
HEADER_SIZE_BYTES: int = 5120
FOOTER_SIZE_BYTES: int = 1032
MAGIC_HEADER: bytes = b"CCS_\x04"
MAGIC_FOOTER: int = 0xFFFFFFFF

# Calibration & Conversion Constants
# Full-scale conversion: V = SCALE_FACTOR * u16 + OFFSET_V
SCALE_FACTOR: float = 10.0 / 52675.0  # ~1.898433801476e-04
OFFSET_V: float = -6.251637430185
DEFAULT_SAMPLING_RATE_HZ: float = 50000.0  # 20 microseconds period -> 50 kHz


@dataclass
class M2DData:
    """Parsed M2D data container."""
    sampling_frequency_hz: float
    data: np.ndarray  # 1D float64 array of physical voltage values [V]
    file_hash: str
    channel_name: str
    voltage_range: Tuple[float, float]
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    total_samples: int = 0


def _bcd_to_int(b: int) -> int:
    """Converts a BCD (Binary-Coded Decimal) byte to an integer."""
    high = (b >> 4) & 0x0F
    low = b & 0x0F
    return high * 10 + low


def _parse_bcd_datetime(raw_bytes: bytes) -> Optional[datetime]:
    """Parses 6 bytes of BCD datetime: [sec, min, hour, day, month, year]."""
    if len(raw_bytes) < 6:
        return None
    try:
        sec = _bcd_to_int(raw_bytes[0])
        minute = _bcd_to_int(raw_bytes[1])
        hour = _bcd_to_int(raw_bcd_hour := raw_bytes[2])
        day = _bcd_to_int(raw_bytes[3])
        month = _bcd_to_int(raw_bytes[4])
        year_short = _bcd_to_int(raw_bytes[5])
        year = 2000 + year_short if year_short < 100 else year_short
        return datetime(year, month, day, hour, minute, sec)
    except Exception:
        return None


def parse_m2d_data(
    file_input: Union[str, Path, BinaryIO, bytes],
    sampling_frequency_hz: Optional[float] = None
) -> M2DData:
    """
    Parses a Condition Catcher .m2d binary file and extracts physical voltage signals.

    Args:
        file_input: File path (str/Path), file-like object (BinaryIO / BytesIO), or raw bytes.
        sampling_frequency_hz: Optional override for sampling frequency. Defaults to 50,000 Hz.

    Returns:
        M2DData containing sampling rate, physical voltage array [V], and metadata.

    Raises:
        ValueError: If file structure or header is invalid or corrupted.
    """
    if isinstance(file_input, (str, Path)):
        with open(file_input, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_input, (io.BytesIO, io.BufferedReader, BinaryIO)):
        current_pos = file_input.tell()
        file_input.seek(0)
        raw_bytes = file_input.read()
        file_input.seek(current_pos)
    elif isinstance(file_input, bytes):
        raw_bytes = file_input
    else:
        raise ValueError(f"Unsupported file_input type: {type(file_input)}")

    file_size = len(raw_bytes)
    min_size = HEADER_SIZE_BYTES + FOOTER_SIZE_BYTES
    if file_size < min_size:
        raise ValueError(f"File size ({file_size} bytes) is too small to be a valid .m2d file.")

    # 1. Verify Magic Header
    header = raw_bytes[:HEADER_SIZE_BYTES]
    if not header.startswith(MAGIC_HEADER):
        raise ValueError(
            f"Invalid M2D file header: expected start with '{MAGIC_HEADER!r}', "
            f"got '{header[:len(MAGIC_HEADER)]!r}'"
        )

    # Calculate SHA-256 file hash
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    # 2. Extract Channel Configuration & Target Channel
    # Check target channel byte at offset 0x0044
    target_ch_num = header[0x0044]
    if target_ch_num == 0:
        target_ch_num = 2  # Default to Ch2 as per hardware standard

    ch_name = f"Ch{target_ch_num}"
    v_min, v_max = -5.0, 5.0

    # Scan channel configuration blocks (0x0080 - 0x0280, 16 channels x 32 bytes)
    for ch_idx in range(16):
        block_offset = 0x0080 + ch_idx * 32
        block = header[block_offset : block_offset + 32]
        c_name = block[:8].split(b"\x00")[0].decode("ascii", errors="ignore").strip()
        if c_name == ch_name or (ch_idx + 1 == target_ch_num):
            try:
                min_str = block[8:16].split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                max_str = block[16:24].split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                if min_str and max_str:
                    v_min = float(min_str)
                    v_max = float(max_str)
            except Exception:
                pass
            break

    # 3. Extract Start Timestamp
    start_dt = _parse_bcd_datetime(header[0x0288 : 0x0288 + 6])

    # 4. Parse Footer & Sample Count
    footer = raw_bytes[file_size - FOOTER_SIZE_BYTES :]
    footer_magic, n_samples_footer = struct.unpack_from("<II", footer, 0)

    # Number of data samples from file size
    data_bytes_len = file_size - HEADER_SIZE_BYTES - FOOTER_SIZE_BYTES
    if data_bytes_len % 2 != 0:
        raise ValueError(f"Corrupted M2D waveform payload: data length ({data_bytes_len}) is odd.")
    n_samples_calc = data_bytes_len // 2

    if footer_magic == MAGIC_FOOTER and n_samples_footer > 0:
        n_samples = n_samples_footer
    else:
        n_samples = n_samples_calc

    if n_samples != n_samples_calc:
        # Align to available data bytes
        n_samples = min(n_samples, n_samples_calc)

    # Extract End Timestamp
    end_dt = _parse_bcd_datetime(footer[0x000C : 0x000C + 6])

    # 5. Extract and Decode Waveform Data
    data_bytes = raw_bytes[HEADER_SIZE_BYTES : HEADER_SIZE_BYTES + n_samples * 2]
    u16_raw = np.frombuffer(data_bytes, dtype="<u2")

    # Linear transformation to Voltage [V]: V = a * u16 + b
    voltage_v = SCALE_FACTOR * u16_raw.astype(np.float64) + OFFSET_V

    fs_hz = sampling_frequency_hz if sampling_frequency_hz is not None else DEFAULT_SAMPLING_RATE_HZ

    return M2DData(
        sampling_frequency_hz=fs_hz,
        data=voltage_v,
        file_hash=file_hash,
        channel_name=ch_name,
        voltage_range=(v_min, v_max),
        start_datetime=start_dt,
        end_datetime=end_dt,
        total_samples=len(voltage_v),
    )


def load_m2d_file(
    file_input: Union[str, Path, BinaryIO, bytes],
    sampling_frequency_hz: Optional[float] = None
) -> Tuple[float, np.ndarray, str]:
    """
    Convenience loader matching the signature of load_wav_file:
    Returns (fs_hz, data_raw, file_hash).
    """
    res = parse_m2d_data(file_input, sampling_frequency_hz=sampling_frequency_hz)
    return res.sampling_frequency_hz, res.data, res.file_hash
