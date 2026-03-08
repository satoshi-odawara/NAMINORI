import pytest
import json
import os
from pathlib import Path
from src.utils.preset_manager import load_presets, save_presets, PRESET_FILE
from src.core.models import AnalysisPreset, AnalysisConfig, SignalQuantity, WindowFunction

@pytest.fixture
def temp_preset_file(tmp_path):
    """Temporary preset file fixture."""
    original_file = PRESET_FILE
    temp_file = tmp_path / "analysis_presets.json"
    
    # Mock PRESET_FILE in the module (if possible, but let's just use it and restore)
    import src.utils.preset_manager
    src.utils.preset_manager.PRESET_FILE = temp_file
    
    yield temp_file
    
    # Restore
    src.utils.preset_manager.PRESET_FILE = original_file

def test_save_and_load_multiple_presets(temp_preset_file):
    # Setup multiple presets
    config1 = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=10.0,
        hpf_enabled=True
    )
    preset1 = AnalysisPreset(
        name="Preset1",
        config=config1,
        unit_space_name="PumpA_Normal"
    )
    
    config2 = AnalysisConfig(
        quantity=SignalQuantity.VELOCITY,
        window=WindowFunction.FLATTOP,
        lowpass_hz=1000.0,
        lpf_enabled=True
    )
    preset2 = AnalysisPreset(
        name="Preset2",
        config=config2,
        unit_space_name=None # Testing null handling
    )
    
    presets = {"Preset1": preset1, "Preset2": preset2}
    
    # Save
    save_presets(presets)
    
    # Verify file content
    with open(temp_preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "Preset1" in data
        assert "Preset2" in data
        assert data["Preset1"]["unit_space_name"] == "PumpA_Normal"
        assert data["Preset2"]["unit_space_name"] is None
    
    # Load
    loaded_presets = load_presets()
    
    # Verify loaded data
    assert len(loaded_presets) == 2
    assert loaded_presets["Preset1"].name == "Preset1"
    assert loaded_presets["Preset1"].unit_space_name == "PumpA_Normal"
    assert loaded_presets["Preset1"].config.quantity == SignalQuantity.ACCEL
    
    assert loaded_presets["Preset2"].name == "Preset2"
    assert loaded_presets["Preset2"].unit_space_name is None
    assert loaded_presets["Preset2"].config.quantity == SignalQuantity.VELOCITY
