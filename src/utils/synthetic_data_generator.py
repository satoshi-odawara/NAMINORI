# -*- coding: utf-8 -*-
"""
Module for generating synthetic vibration signals for testing and analysis.

This module provides tools to create complex signals by combining pure tones,
amplitude/frequency modulation, impulses, and various types of noise. This is
useful for creating robust test cases for vibration analysis algorithms.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import numpy as np

class NoiseType(Enum):
    """Enum for different types of background noise."""
    WHITE = "white"
    PINK = "pink"
    BROWNIAN = "brownian"

@dataclass
class AMConfig:
    """Configuration for Amplitude Modulation."""
    mod_freq_hz: float
    mod_index: float # Modulation index (0 to 1)

@dataclass
class FMConfig:
    """Configuration for Frequency Modulation."""
    mod_freq_hz: float
    mod_index: float # Modulation index (deviation / mod_freq)

@dataclass
class ImpulseConfig:
    """Configuration for adding periodic impulses."""
    impulse_rate_hz: float
    impulse_amplitude: float

@dataclass
class NoiseConfig:
    """Configuration for adding background noise."""
    noise_type: NoiseType
    snr_db: Optional[float] = None # Signal-to-Noise Ratio in dB

@dataclass
class SignalConfig:
    """
    Comprehensive configuration for generating a synthetic signal.
    """
    fs_hz: int
    duration_s: float
    base_freq_hz: float
    base_amplitude: float
    am_config: Optional[AMConfig] = None
    fm_config: Optional[FMConfig] = None
    impulses: Optional[List[ImpulseConfig]] = field(default_factory=list)
    noise_config: Optional[NoiseConfig] = None

def generate_signal(config: SignalConfig) -> np.ndarray:
    """
    Generates a synthetic signal based on the provided configuration.

    Args:
        config (SignalConfig): The configuration object detailing the signal to generate.

    Returns:
        np.ndarray: The generated signal as a numpy array, normalized to [-1.0, 1.0].
    """
    # 1. Create time vector
    t = np.linspace(0., config.duration_s, int(config.fs_hz * config.duration_s), endpoint=False)

    # 2. Generate base carrier signal (pure tone)
    carrier_signal = _generate_pure_tone(t, config.base_freq_hz, config.base_amplitude)

    # 3. Apply Frequency Modulation (FM) if configured
    if config.fm_config:
        carrier_signal = _apply_fm_modulation(t, carrier_signal, config)

    # 4. Apply Amplitude Modulation (AM) if configured
    if config.am_config:
        carrier_signal = _apply_am_modulation(t, carrier_signal, config)

    # 5. Add impulses
    if config.impulses:
        for impulse_conf in config.impulses:
            carrier_signal = _add_impulses(t, carrier_signal, impulse_conf)

    # 6. Add background noise
    if config.noise_config:
        carrier_signal = _add_noise(carrier_signal, config.noise_config)
    
    # 7. Normalize final signal
    max_abs = np.max(np.abs(carrier_signal))
    if max_abs == 0:
        return carrier_signal # Avoid division by zero
    return carrier_signal / max_abs

def _generate_pure_tone(t: np.ndarray, freq_hz: float, amplitude: float) -> np.ndarray:
    """Generates a simple sine wave."""
    # Implementation deferred to Subtask 2
    raise NotImplementedError("This function will be implemented in a future task.")

def _apply_fm_modulation(t: np.ndarray, carrier_signal: np.ndarray, config: SignalConfig) -> np.ndarray:
    """Applies Frequency Modulation to the carrier signal."""
    # Implementation deferred to Subtask 2
    raise NotImplementedError("This function will be implemented in a future task.")

def _apply_am_modulation(t: np.ndarray, carrier_signal: np.ndarray, config: SignalConfig) -> np.ndarray:
    """Applies Amplitude Modulation to the carrier signal."""
    # Implementation deferred to Subtask 2
    raise NotImplementedError("This function will be implemented in a future task.")

def _add_impulses(t: np.ndarray, signal: np.ndarray, impulse_config: ImpulseConfig) -> np.ndarray:
    """Adds periodic impulses to the signal."""
    # Implementation deferred to Subtask 2
    raise NotImplementedError("This function will be implemented in a future task.")

def _add_noise(signal: np.ndarray, noise_config: NoiseConfig) -> np.ndarray:
    """Adds specified background noise to the signal based on SNR."""
    # Implementation deferred to Subtask 2
    raise NotImplementedError("This function will be implemented in a future task.")
