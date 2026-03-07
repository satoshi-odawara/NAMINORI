import json
import os
from pathlib import Path
from typing import Dict, List
from src.core.models import AnalysisPreset, AnalysisConfig, SignalQuantity, WindowFunction

PRESET_FILE = Path("data/analysis_presets.json")

def load_presets() -> Dict[str, AnalysisPreset]:
    """Loads analysis presets from a JSON file."""
    if not PRESET_FILE.exists():
        return {}
    
    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        presets = {}
        for name, p in data.items():
            config_data = p["config"]
            config = AnalysisConfig(
                quantity=SignalQuantity(config_data["quantity"]),
                window=WindowFunction(config_data["window"]),
                highpass_hz=config_data.get("highpass_hz"),
                lowpass_hz=config_data.get("lowpass_hz"),
                filter_order=config_data.get("filter_order", 4),
                hpf_enabled=config_data.get("hpf_enabled", False),
                lpf_enabled=config_data.get("lpf_enabled", False),
                noise_reduction_plugin_name=config_data.get("noise_reduction_plugin_name"),
                noise_reduction_plugin_params=config_data.get("noise_reduction_plugin_params")
            )
            presets[name] = AnalysisPreset(
                name=name,
                config=config,
                top_n_peaks=p.get("top_n_peaks", 5),
                min_peak_height_percent=p.get("min_peak_height_percent", 10.0),
                peak_distance_hz=p.get("peak_distance_hz", 10.0),
                fft_log_x=p.get("fft_log_x", False),
                show_raw_signal=p.get("show_raw_signal", True),
                spec_nperseg=p.get("spec_nperseg", 512)
            )
        return presets
    except Exception as e:
        print(f"Error loading presets: {e}")
        return {}

def save_presets(presets: Dict[str, AnalysisPreset]):
    """Saves analysis presets to a JSON file."""
    PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for name, p in presets.items():
        data[name] = {
            "config": {
                "quantity": p.config.quantity.value,
                "window": p.config.window.value,
                "highpass_hz": p.config.highpass_hz,
                "lowpass_hz": p.config.lowpass_hz,
                "filter_order": p.config.filter_order,
                "hpf_enabled": p.config.hpf_enabled,
                "lpf_enabled": p.config.lpf_enabled,
                "noise_reduction_plugin_name": p.config.noise_reduction_plugin_name,
                "noise_reduction_plugin_params": p.config.noise_reduction_plugin_params
            },
            "top_n_peaks": p.top_n_peaks,
            "min_peak_height_percent": p.min_peak_height_percent,
            "peak_distance_hz": p.peak_distance_hz,
            "fft_log_x": p.fft_log_x,
            "show_raw_signal": p.show_raw_signal,
            "spec_nperseg": p.spec_nperseg
        }
    
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
