import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    DESCRIPTION: Load a YAML config file and return its contents as a python dictionary.
    INPUTS: 
            config_path - path to the YAML config file
    OUTPUTS: 
            config - dict containing all config parameters""" 
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def print_config(config: Dict[str, Any]) -> None:
    """
    DESCRIPTION: Pretty print nested config.
    INPUTS: 
            config - dict containing all config parameters
    OUTPUTS: None""" 
    print("\n========== Experiment Config ==========")
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
    print("\n======================================\n")
