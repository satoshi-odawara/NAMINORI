from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
import numpy as np
from dataclasses import dataclass, field
import importlib
import os
import inspect
from pathlib import Path
import sys # Added for dynamic module loading

# Define a type for the parameter dictionary to improve readability
ParameterType = Literal['number_input', 'slider']

@dataclass
class PluginParameter:
    """Defines a parameter for a plugin that can be rendered in a UI."""
    name: str
    label: str
    param_type: ParameterType
    default: float
    min_value: float = None
    max_value: float = None
    help_text: str = ""

class NoiseReductionPlugin(ABC):
    """
    Abstract Base Class for a noise reduction plugin.
    All custom noise reduction algorithms must inherit from this class.
    """
    @abstractmethod
    def get_name(self) -> str:
        """Returns a unique, machine-readable name for the plugin (e.g., 'notch_filter')."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Returns a human-readable name for display in the UI (e.g., 'Notch Filter')."""
        pass

    @abstractmethod
    def get_parameters(self) -> List[PluginParameter]:
        """
        Returns a list of PluginParameter objects that define the configurable
        parameters for this plugin.
        """
        pass

    @abstractmethod
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        """
        Processes the input signal to reduce noise.

        Args:
            data: The input signal as a NumPy array.
            fs: The sampling frequency of the signal.
            **params: Keyword arguments corresponding to the 'name' of each
                      PluginParameter defined in get_parameters().

        Returns:
            The processed (filtered) signal as a NumPy array.
        """
        pass

class PluginManager:
    """
    A singleton class to discover, load, and manage noise reduction plugins.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance.plugins: Dict[str, NoiseReductionPlugin] = {}
            cls._instance._is_loaded = False
        return cls._instance

    def load_plugins(self, plugin_dir: str = "src/plugins/noise_reduction"):
        """
        Discovers and loads all plugins from the specified directory.
        This method should only be called once.
        """
        if self._is_loaded:
            return
            
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            print(f"Plugin directory {plugin_dir} not found. Skipping plugin loading.")
            self._is_loaded = True
            return

        for file in plugin_path.glob("*.py"):
            if file.name == "__init__.py":
                continue

            # Load module directly from file path
            module_name = file.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module # Add to sys.modules to prevent re-import issues
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if the class is a subclass of NoiseReductionPlugin and not the base class itself
                    if issubclass(obj, NoiseReductionPlugin) and obj is not NoiseReductionPlugin:
                        plugin_instance = obj()
                        plugin_name = plugin_instance.get_name()
                        if plugin_name in self.plugins:
                            print(f"Warning: Duplicate plugin name '{plugin_name}'. Overwriting.")
                        self.plugins[plugin_name] = plugin_instance
                        print(f"Successfully loaded plugin: '{plugin_instance.get_display_name()}'")

            except Exception as e:
                print(f"Error loading plugin from {file.name}: {e}")

        self._is_loaded = True

    def get_plugin(self, name: str) -> NoiseReductionPlugin:
        """Returns a plugin instance by its unique name."""
        return self.plugins.get(name)

    def list_plugins(self) -> List[NoiseReductionPlugin]:
        """Returns a list of all loaded plugin instances."""
        return list(self.plugins.values())

# Global instance of the plugin manager
plugin_manager = PluginManager()
