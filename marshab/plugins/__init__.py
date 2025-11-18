"""Plugin system for extensibility."""

from typing import Dict, List, Optional
from pathlib import Path

from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# Plugin registry
PLUGIN_REGISTRY: Dict[str, Dict] = {
    "datasets": {},
    "criteria": {}
}


def register_dataset_plugin(plugin):
    """Register a dataset plugin.
    
    Args:
        plugin: Plugin object with get_datasets() method
    """
    try:
        datasets = plugin.get_datasets()
        for dataset in datasets:
            plugin_id = dataset.get("id") or plugin.__class__.__name__
            PLUGIN_REGISTRY["datasets"][plugin_id] = {
                "plugin": plugin,
                "dataset": dataset
            }
        logger.info("Registered dataset plugin", plugin=plugin.__class__.__name__, datasets=len(datasets))
    except Exception as e:
        logger.error("Failed to register dataset plugin", error=str(e))


def register_criterion_plugin(plugin):
    """Register a criterion plugin.
    
    Args:
        plugin: Plugin object with get_criteria() method
    """
    try:
        criteria = plugin.get_criteria()
        for criterion in criteria:
            criterion_id = criterion.get("id") or plugin.__class__.__name__
            PLUGIN_REGISTRY["criteria"][criterion_id] = {
                "plugin": plugin,
                "criterion": criterion
            }
        logger.info("Registered criterion plugin", plugin=plugin.__class__.__name__, criteria=len(criteria))
    except Exception as e:
        logger.error("Failed to register criterion plugin", error=str(e))


def load_plugins_from_config(config_path: Optional[Path] = None):
    """Load plugins from configuration file.
    
    Args:
        config_path: Path to plugins.yaml config file
    """
    if config_path is None:
        from marshab.config import get_config
        config = get_config()
        config_path = Path(config.paths.data_dir) / "config" / "plugins.yaml"
    
    if not config_path.exists():
        logger.info("No plugins config found, skipping plugin loading", config_path=str(config_path))
        return
    
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    plugins_config = config_data.get("plugins", [])
    
    for plugin_config in plugins_config:
        if not plugin_config.get("enabled", True):
            continue
        
        plugin_path = plugin_config.get("path")
        if not plugin_path:
            continue
        
        try:
            # Import plugin module
            module_path, class_name = plugin_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            
            # Register based on plugin type
            if hasattr(plugin_instance, "get_datasets"):
                register_dataset_plugin(plugin_instance)
            if hasattr(plugin_instance, "get_criteria"):
                register_criterion_plugin(plugin_instance)
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}", error=str(e))

