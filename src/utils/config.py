"""Configuration loading and management for PathoVision."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two config dictionaries. Override takes precedence.

    Args:
        base: Base configuration dictionary.
        override: Override values (takes precedence).

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "model.backbone").
        default: Default value if key not found.

    Returns:
        The value at the key path, or default.
    """
    keys = key_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply command-line overrides in format 'key.path=value'.

    Args:
        config: Base configuration dictionary.
        overrides: List of 'key.path=value' strings.

    Returns:
        Config with overrides applied.
    """
    for override in overrides:
        if "=" not in override:
            logger.warning("Skipping malformed override: %s", override)
            continue

        key_path, value_str = override.split("=", 1)
        keys = key_path.strip().split(".")

        # Parse value type
        value = _parse_value(value_str.strip())

        # Set nested value
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

        logger.info("Override: %s = %s", key_path, value)

    return config


def _parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() in ("null", "none"):
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str
