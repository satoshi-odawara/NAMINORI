import pytest
import numpy as np
from src.utils import synthetic_data_generator as sdg

# Constants for testing
FS = 48000
DURATION = 1.0
T = np.linspace(0., DURATION, int(FS * DURATION), endpoint=False)

def test_generate_pure_tone():
    """Test the generation of a simple sine wave."""
    freq = 100
    amp = 0.5
    signal = sdg._generate_pure_tone(T, freq, amp)
    
    assert len(signal) == FS * DURATION
    assert np.isclose(np.max(signal), amp)
    
    # Verify frequency
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/FS)
    peak_freq_index = np.argmax(np.abs(fft))
    assert np.isclose(freqs[peak_freq_index], freq, atol=1)

def test_apply_am_modulation():
    """Test amplitude modulation."""
    base_signal = np.ones(len(T)) # Use a DC signal for simplicity
    config = sdg.SignalConfig(
        fs_hz=FS,
        duration_s=DURATION,
        base_freq_hz=100, # Not used directly by AM modulator but required
        base_amplitude=1.0,
        am_config=sdg.AMConfig(mod_freq_hz=10, mod_index=0.5)
    )
    
    am_signal = sdg._apply_am_modulation(T, base_signal, config)
    
    # The modulating envelope should vary between (1-mod_index) and (1+mod_index)
    assert np.isclose(np.min(am_signal), 1 - 0.5)
    assert np.isclose(np.max(am_signal), 1 + 0.5)

def test_apply_fm_modulation():
    """Test frequency modulation."""
    config = sdg.SignalConfig(
        fs_hz=FS,
        duration_s=DURATION,
        base_freq_hz=1000,
        base_amplitude=1.0,
        fm_config=sdg.FMConfig(mod_freq_hz=100, mod_index=2.0)
    )
    
    # Carrier signal is not used, FM generates a new signal
    fm_signal = sdg._apply_fm_modulation(T, np.zeros_like(T), config)
    
    assert len(fm_signal) == len(T)
    assert np.isclose(np.max(fm_signal), 1.0)
    
    # Verifying FM in the frequency domain (Carson's rule) is complex.
    # For a unit test, we'll just check that the signal is not a pure tone.
    fft = np.fft.fft(fm_signal)
    # A pure tone would have one major peak. FM spreads power across sidebands.
    # Check if power is distributed over more than just a few bins.
    significant_peaks = np.sum(np.abs(fft) > np.max(np.abs(fft)) * 0.1)
    assert significant_peaks > 4 # Expect carrier + at least two pairs of sidebands

def test_add_impulses():
    """Test adding periodic impulses."""
    signal = np.zeros(len(T))
    impulse_rate = 10 # 10 Hz
    impulse_amp = 0.8
    config = sdg.ImpulseConfig(impulse_rate_hz=impulse_rate, impulse_amplitude=impulse_amp)
    
    signal_with_impulses = sdg._add_impulses(T, signal, config)
    
    # Number of impulses should be duration * rate
    expected_num_impulses = int(DURATION * impulse_rate)
    num_found_impulses = np.sum(signal_with_impulses > 0)
    
    assert num_found_impulses == expected_num_impulses
    assert np.all(signal_with_impulses[signal_with_impulses > 0] == impulse_amp)

def test_add_white_noise_snr():
    """Test adding white noise with a specific SNR."""
    signal = sdg._generate_pure_tone(T, 100, 0.5)
    target_snr_db = 10
    config = sdg.NoiseConfig(noise_type=sdg.NoiseType.WHITE, snr_db=target_snr_db)
    
    noisy_signal = sdg._add_noise(signal, config)
    
    # Verify resulting SNR
    signal_power = np.mean(signal**2)
    noise_power = np.mean((noisy_signal - signal)**2)
    
    # Handle potential division by zero if noise power is zero
    if noise_power > 0:
        result_snr_linear = signal_power / noise_power
        result_snr_db = 10 * np.log10(result_snr_linear)
        assert np.isclose(result_snr_db, target_snr_db, atol=0.5) # Allow some tolerance
    else:
        # If no noise was added, this is unexpected
        pytest.fail("Noise power was zero, SNR test could not be completed.")

def test_add_noise_unimplemented():
    """Test that unimplemented noise types raise an error."""
    signal = np.zeros(100)
    with pytest.raises(NotImplementedError):
        sdg._add_noise(signal, sdg.NoiseConfig(noise_type=sdg.NoiseType.PINK, snr_db=10))
    with pytest.raises(NotImplementedError):
        sdg._add_noise(signal, sdg.NoiseConfig(noise_type=sdg.NoiseType.BROWNIAN, snr_db=10))

def test_generate_signal_normalization():
    """Test the final normalization of the generated signal."""
    # Create a config that would produce a signal with max amplitude > 1.0
    config = sdg.SignalConfig(
        fs_hz=FS,
        duration_s=DURATION,
        base_freq_hz=100,
        base_amplitude=1.0, # Base amplitude is 1.0
        impulses=[sdg.ImpulseConfig(impulse_rate_hz=5, impulse_amplitude=0.5)] # Adding impulses will exceed 1.0
    )
    
    final_signal = sdg.generate_signal(config)
    
    assert np.max(np.abs(final_signal)) <= 1.0
    # The max should be exactly 1.0 because it was normalized
    assert np.isclose(np.max(np.abs(final_signal)), 1.0)
    
def test_generate_signal_complex_case():
    """
    Integration-style test for a complex signal configuration.
    """
    config = sdg.SignalConfig(
        fs_hz=48000,
        duration_s=2.0,
        base_freq_hz=500,
        base_amplitude=0.4,
        am_config=sdg.AMConfig(mod_freq_hz=20, mod_index=0.4),
        fm_config=sdg.FMConfig(mod_freq_hz=10, mod_index=2.0),
        impulses=[sdg.ImpulseConfig(impulse_rate_hz=5, impulse_amplitude=0.5)],
        noise_config=sdg.NoiseConfig(noise_type=sdg.NoiseType.WHITE, snr_db=15)
    )
    
    signal = sdg.generate_signal(config)
    
    assert len(signal) == config.fs_hz * config.duration_s
    assert np.max(np.abs(signal)) <= 1.0
    # Check that something was actually generated
    assert np.std(signal) > 0
