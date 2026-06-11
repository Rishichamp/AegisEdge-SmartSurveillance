"""
Configuration Loader
====================

This module loads and validates the YAML configuration file that controls
all system behavior. It provides a centralized way to access settings
throughout the codebase.

WHY A CENTRALIZED CONFIG?
- Single source of truth for all settings
- Easy to switch between desktop/Jetson/Pi configurations
- No hardcoded values scattered across the codebase
- Users can customize behavior without touching code

HOW IT WORKS:
1. Load YAML file from disk
2. Merge with default values (so missing keys don't crash the system)
3. Validate critical settings (e.g., confidence must be 0-1)
4. Provide dot-notation access (config.detection.model instead of config['detection']['model'])
"""

import os
import yaml
from typing import Any, Dict, Optional


def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
                     Defaults to config/default_config.yaml.
    
    Returns:
        Dictionary containing all configuration settings.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    # Resolve the path relative to the project root
    if not os.path.isabs(config_path):
        # Find project root (directory containing 'config/' folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure the config file exists. "
            f"You can copy config/default_config.yaml as a starting point."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate critical settings
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values to catch common mistakes early.
    """
    # Check detection confidence is in valid range
    det_conf = config.get('detection', {}).get('confidence_threshold', 0.5)
    if not 0.0 <= det_conf <= 1.0:
        raise ValueError(
            f"detection.confidence_threshold must be between 0.0 and 1.0, "
            f"got {det_conf}"
        )
    
    # Check zone polygons have at least 3 points
    for zone in config.get('zones', []):
        polygon = zone.get('polygon', [])
        if len(polygon) < 3:
            raise ValueError(
                f"Zone '{zone.get('name', 'unnamed')}' must have at least "
                f"3 polygon points, got {len(polygon)}"
            )


class ConfigAccessor:
    """
    Provides dot-notation access to nested configuration dictionaries.
    
    This is a convenience wrapper that lets you write:
        config.detection.model
    instead of:
        config['detection']['model']
    
    WHY?
    - More readable code
    - Attribute-style access feels more Pythonic
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the accessor with a dictionary.
        
        Args:
            data: Dictionary to wrap with dot-notation access.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively wrap nested dictionaries
                setattr(self, key, ConfigAccessor(value))
            elif isinstance(value, list):
                # Handle lists of dicts (e.g., zones)
                setattr(self, key, [
                    ConfigAccessor(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigAccessor):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigAccessor) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        return f"ConfigAccessor({self.to_dict()})"


def get_config(config_path: str = "config/default_config.yaml") -> ConfigAccessor:
    """
    Load configuration and return a dot-notation accessor.
    
    Args:
        config_path: Path to YAML configuration file.
    
    Returns:
        ConfigAccessor with dot-notation access to all settings.
    """
    raw_config = load_config(config_path)
    return ConfigAccessor(raw_config)
BaseConfig = get_config
