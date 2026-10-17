"""Configuration utility module for underwater hull inspection."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and parse YAML configuration file with path resolution.

    Args:
        config_path: Path to configuration YAML. Defaults to config/config.yaml.

    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        env_config = os.getenv("FOML_CONFIG_PATH")
        config_path = Path(env_config) if env_config else DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]
