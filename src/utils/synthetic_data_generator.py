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
    return amplitude * np.sin(2. * np.pi * freq_hz * t)

def _apply_fm_modulation(t: np.ndarray, carrier_signal: np.ndarray, config: SignalConfig) -> np.ndarray:
    """Applies Frequency Modulation to the carrier signal."""
    fm_conf = config.fm_config
    # FM is a change in frequency, so we recalculate the signal's phase
    modulator = np.sin(2. * np.pi * fm_conf.mod_freq_hz * t)
    deviation = fm_conf.mod_index * fm_conf.mod_freq_hz
    phase = 2. * np.pi * config.base_freq_hz * t + deviation / fm_conf.mod_freq_hz * modulator
    return config.base_amplitude * np.sin(phase)

def _apply_am_modulation(t: np.ndarray, carrier_signal: np.ndarray, config: SignalConfig) -> np.ndarray:
    """Applies Amplitude Modulation to the carrier signal."""
    am_conf = config.am_config
    modulator = 1 + am_conf.mod_index * np.sin(2. * np.pi * am_conf.mod_freq_hz * t)
    return carrier_signal * modulator

def _add_impulses(t: np.ndarray, signal: np.ndarray, impulse_config: ImpulseConfig) -> np.ndarray:
    """Adds periodic impulses to the signal."""
    impulse_period = 1.0 / impulse_config.impulse_rate_hz
    num_samples = len(t)
    fs = num_samples / t[-1] # Recalculate fs from time vector
    
    impulse_train = np.zeros(num_samples)
    impulse_indices = np.arange(0, num_samples, int(impulse_period * fs)).astype(int)
    impulse_train[impulse_indices] = impulse_config.impulse_amplitude
    
    # Simple convolution can be slow, for a simple impulse, direct addition is fine
    return signal + impulse_train

def _add_noise(signal: np.ndarray, noise_config: NoiseConfig) -> np.ndarray:
    """Adds specified background noise to the signal based on SNR."""
    if noise_config.noise_type != NoiseType.WHITE:
        raise NotImplementedError(f"Noise type '{noise_config.noise_type.value}' is not yet implemented.")

    if noise_config.snr_db is None:
        return signal # No noise to add if SNR is not specified
        
    # Calculate signal power
    signal_power = np.mean(signal**2)
    # Calculate noise power from SNR
    snr_linear = 10**(noise_config.snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate white noise with the calculated power
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    
    return signal + noise
