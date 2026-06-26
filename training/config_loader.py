import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader and manager."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the config YAML file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config['data']['data_files']"""
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict:
        """Return the entire config as a dictionary."""
        return self._config

    def __repr__(self) -> str:
        return f"Config({self.config_path})"


def load_config(config_path: str = "training/config.yaml") -> Config:
    """
    Convenience function to load config.

    Args:
        config_path: Path to config YAML

    Returns:
        Config object
    """
    return Config(config_path)
