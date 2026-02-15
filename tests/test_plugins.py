import pytest
import numpy as np
import os
from pathlib import Path
import sys
import shutil

# Temporarily add the src directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.plugins import PluginManager, NoiseReductionPlugin, PluginParameter, plugin_manager as global_plugin_manager

# Fixture to create a temporary plugin directory and clean it up
@pytest.fixture
def temp_plugin_dir(tmp_path):
    # Create the directory structure expected by the plugin manager
    # For testing, we mimic src/plugins/noise_reduction
    test_plugin_path = tmp_path / "plugins" / "noise_reduction"
    test_plugin_path.mkdir(parents=True, exist_ok=True)
    return test_plugin_path

# Fixture to get a fresh PluginManager instance for each test
@pytest.fixture
def fresh_plugin_manager():
    # Reset the singleton instance for a clean slate
    PluginManager._instance = None
    manager = PluginManager()
    manager._is_loaded = False # Ensure it attempts to load plugins
    yield manager
    # Clean up after test if needed
    PluginManager._instance = None # Ensure next test gets a fresh instance

# Helper to write dummy plugin files
def create_dummy_plugin_file(directory: Path, filename: str, content: str):
    file_path = directory / filename
    with open(file_path, "w") as f:
        f.write(content)
    # Ensure the directory containing the plugin is in sys.path for importlib to find it
    if str(directory.parent.parent) not in sys.path:
        sys.path.insert(0, str(directory.parent.parent))

# Test that PluginManager is a singleton
def test_plugin_manager_singleton():
    manager1 = PluginManager()
    manager2 = PluginManager()
    assert manager1 is manager2

# Test that the global plugin_manager instance is the singleton
def test_global_plugin_manager_is_singleton():
    manager1 = PluginManager()
    assert manager1 is global_plugin_manager

# Test plugin discovery and loading
def test_plugin_manager_discovery(fresh_plugin_manager, temp_plugin_dir):
    # Create a dummy plugin
    plugin_content = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any

class DummyPlugin(NoiseReductionPlugin):
    def get_name(self) -> str:
        return "dummy_plugin"
    def get_display_name(self) -> str:
        return "Dummy Plugin"
    def get_parameters(self) -> List[PluginParameter]:
        return [PluginParameter("factor", "Factor", "number_input", 1.0)]
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        return data * params.get("factor", 1.0)
"""
    create_dummy_plugin_file(temp_plugin_dir, "dummy_plugin.py", plugin_content)

    # Load plugins from the temporary directory
    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    
    # Assert discovery
    assert "dummy_plugin" in fresh_plugin_manager.plugins
    assert fresh_plugin_manager.get_plugin("dummy_plugin").get_display_name() == "Dummy Plugin"

# Test getting a specific plugin
def test_plugin_manager_get_plugin(fresh_plugin_manager, temp_plugin_dir):
    plugin_content = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any

class AnotherDummyPlugin(NoiseReductionPlugin):
    def get_name(self) -> str:
        return "another_dummy_plugin"
    def get_display_name(self) -> str:
        return "Another Dummy Plugin"
    def get_parameters(self) -> List[PluginParameter]:
        return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        return data
"""
    create_dummy_plugin_file(temp_plugin_dir, "another_dummy_plugin.py", plugin_content)
    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))

    plugin = fresh_plugin_manager.get_plugin("another_dummy_plugin")
    assert plugin is not None
    assert plugin.get_name() == "another_dummy_plugin"

# Test listing all plugins
def test_plugin_manager_list_plugins(fresh_plugin_manager, temp_plugin_dir):
    plugin_content1 = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any
class ListPlugin1(NoiseReductionPlugin):
    def get_name(self) -> str: return "list_plugin1"
    def get_display_name(self) -> str: return "List Plugin 1"
    def get_parameters(self) -> List[PluginParameter]: return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data
"""
    plugin_content2 = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any
class ListPlugin2(NoiseReductionPlugin):
    def get_name(self) -> str: return "list_plugin2"
    def get_display_name(self) -> str: return "List Plugin 2"
    def get_parameters(self) -> List[PluginParameter]: return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data
"""
    create_dummy_plugin_file(temp_plugin_dir, "list_plugin1.py", plugin_content1)
    create_dummy_plugin_file(temp_plugin_dir, "list_plugin2.py", plugin_content2)
    
    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    
    plugins = fresh_plugin_manager.list_plugins()
    plugin_names = {p.get_name() for p in plugins}
    
    assert len(plugins) == 2
    assert "list_plugin1" in plugin_names
    assert "list_plugin2" in plugin_names

# Test handling of duplicate plugin names (latter one should overwrite)
def test_plugin_manager_duplicate_names(fresh_plugin_manager, temp_plugin_dir, capsys):
    plugin_content1 = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any
class DupPlugin1(NoiseReductionPlugin):
    def get_name(self) -> str: return "duplicate_name"
    def get_display_name(self) -> str: return "First Duplicate"
    def get_parameters(self) -> List[PluginParameter]: return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data + 1
"""
    plugin_content2 = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any
class DupPlugin2(NoiseReductionPlugin):
    def get_name(self) -> str: return "duplicate_name"
    def get_display_name(self) -> str: return "Second Duplicate"
    def get_parameters(self) -> List[PluginParameter]: return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data + 2
"""
    create_dummy_plugin_file(temp_plugin_dir, "dup_plugin1.py", plugin_content1)
    create_dummy_plugin_file(temp_plugin_dir, "dup_plugin2.py", plugin_content2) # This one should overwrite
    
    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    
    plugin = fresh_plugin_manager.get_plugin("duplicate_name")
    assert plugin.get_display_name() == "Second Duplicate"
    
    # Check for warning message
    captured = capsys.readouterr()
    assert "Warning: Duplicate plugin name 'duplicate_name'. Overwriting." in captured.out

# Test handling of invalid plugin files (e.g., syntax errors, not inheriting ABC)
def test_plugin_manager_invalid_plugin_syntax(fresh_plugin_manager, temp_plugin_dir, capsys):
    invalid_content = "def invalid_syntax: pass"
    create_dummy_plugin_file(temp_plugin_dir, "invalid_syntax.py", invalid_content)

    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    
    # Assert that no plugins were loaded from this file
    assert "invalid_syntax" not in fresh_plugin_manager.plugins
    
    # Check for error message
    captured = capsys.readouterr()
    assert "Error loading plugin from invalid_syntax.py" in captured.out

def test_plugin_manager_invalid_plugin_no_abc(fresh_plugin_manager, temp_plugin_dir):
    invalid_content = """
import numpy as np
class NotAPlugin:
    def get_name(self) -> str: return "not_a_plugin"
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data
"""
    create_dummy_plugin_file(temp_plugin_dir, "not_a_plugin.py", invalid_content)

    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    
    # Assert that no plugins were loaded from this file because it doesn't inherit NoiseReductionPlugin
    assert "not_a_plugin" not in fresh_plugin_manager.plugins

# Test that plugins are only loaded once (due to _is_loaded flag)
def test_plugin_manager_load_once(fresh_plugin_manager, temp_plugin_dir):
    plugin_content = """
from src.core.plugins import NoiseReductionPlugin, PluginParameter
import numpy as np
from typing import List, Any

class OncePlugin(NoiseReductionPlugin):
    def get_name(self) -> str: return "once_plugin"
    def get_display_name(self) -> str: return "Once Plugin"
    def get_parameters(self) -> List[PluginParameter]: return []
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray: return data
"""
    create_dummy_plugin_file(temp_plugin_dir, "once_plugin.py", plugin_content)

    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir))
    initial_plugin_count = len(fresh_plugin_manager.plugins)
    
    fresh_plugin_manager.load_plugins(plugin_dir=str(temp_plugin_dir)) # Call again
    
    # Plugin count should not change, as it only loads once
    assert len(fresh_plugin_manager.plugins) == initial_plugin_count
    assert fresh_plugin_manager._is_loaded == True

# Remove the sys.path modification
sys.path.pop(0)
