"""Configuration modules for MarsHab."""

# Import and re-export from the parent config.py module
# This allows marshab.config to work as both a package (for criteria_config)
# and provide access to the config.py module functions
import importlib.util
from pathlib import Path

_config_file = Path(__file__).parent.parent / "config.py"
if _config_file.exists():
    spec = importlib.util.spec_from_file_location("marshab._config_module", _config_file)
    _config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_config_mod)

    # Re-export key functions and classes
    get_config = _config_mod.get_config
    Config = _config_mod.Config
    reset_config = _config_mod.reset_config
    DataSource = _config_mod.DataSource
    AnalysisConfig = _config_mod.AnalysisConfig
    NavigationConfig = _config_mod.NavigationConfig
    PathsConfig = _config_mod.PathsConfig
    PathfindingStrategy = _config_mod.PathfindingStrategy
    MarsParameters = _config_mod.MarsParameters
    LoggingConfig = _config_mod.LoggingConfig

    __all__ = [
        "get_config",
        "Config",
        "reset_config",
        "DataSource",
        "AnalysisConfig",
        "NavigationConfig",
        "PathsConfig",
        "PathfindingStrategy",
        "MarsParameters",
        "LoggingConfig"
    ]

