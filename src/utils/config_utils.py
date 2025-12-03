"""
Configuration utilities for loading and managing project configs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists():
            return parent
    return current.parent.parent.parent


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_yaml(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a configuration file by name.
    
    Args:
        config_name: Name of the config (e.g., 'project', 'gan', 'quantum')
        config_dir: Optional directory path. Defaults to project's configs/
        
    Returns:
        Configuration dictionary
    """
    if config_dir is None:
        config_dir = get_project_root() / "configs"
    else:
        config_dir = Path(config_dir)
    
    # Add .yaml extension if not present
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    
    config_path = config_dir / config_name
    return load_yaml(config_path)


def load_all_configs(config_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files from the config directory.
    
    Args:
        config_dir: Optional directory path
        
    Returns:
        Dictionary with config names as keys
    """
    if config_dir is None:
        config_dir = get_project_root() / "configs"
    else:
        config_dir = Path(config_dir)
    
    configs = {}
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_yaml(config_file)
    
    return configs


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_env() -> None:
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try loading from .env.example
        env_example = project_root / ".env.example"
        if env_example.exists():
            load_dotenv(env_example)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable."""
    return os.environ.get(key, default)


class Config:
    """Configuration manager class."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir) if config_dir else get_project_root() / "configs"
        self._configs = {}
        self._load_all()
        
        # Load environment variables
        load_env()
    
    def _load_all(self) -> None:
        """Load all configuration files."""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            self._configs[config_name] = load_yaml(config_file)
    
    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            config_name: Name of the configuration file
            key: Optional dot-notation key (e.g., 'model.hidden_dims')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if config_name not in self._configs:
            return default
        
        config = self._configs[config_name]
        
        if key is None:
            return config
        
        # Navigate nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, config_name: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            config_name: Name of the configuration file
            key: Dot-notation key
            value: Value to set
        """
        if config_name not in self._configs:
            self._configs[config_name] = {}
        
        keys = key.split('.')
        config = self._configs[config_name]
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_name: str) -> None:
        """Save a configuration to file."""
        if config_name in self._configs:
            config_path = self.config_dir / f"{config_name}.yaml"
            save_yaml(self._configs[config_name], config_path)
    
    def reload(self, config_name: Optional[str] = None) -> None:
        """Reload configuration(s) from disk."""
        if config_name:
            config_path = self.config_dir / f"{config_name}.yaml"
            if config_path.exists():
                self._configs[config_name] = load_yaml(config_path)
        else:
            self._load_all()
    
    @property
    def project(self) -> Dict[str, Any]:
        return self._configs.get('project', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._configs.get('data', {})
    
    @property
    def gan(self) -> Dict[str, Any]:
        return self._configs.get('gan', {})
    
    @property
    def quantum(self) -> Dict[str, Any]:
        return self._configs.get('quantum', {})
    
    @property
    def rl(self) -> Dict[str, Any]:
        return self._configs.get('rl', {})
    
    @property
    def qsar(self) -> Dict[str, Any]:
        return self._configs.get('qsar', {})
    
    @property
    def docking(self) -> Dict[str, Any]:
        return self._configs.get('docking', {})
    
    @property
    def tox_admet(self) -> Dict[str, Any]:
        return self._configs.get('tox_admet', {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        return self._configs.get('ui', {})


# Global config instance
_config_instance = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
