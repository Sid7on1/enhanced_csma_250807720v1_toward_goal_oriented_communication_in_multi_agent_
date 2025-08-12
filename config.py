import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
from threading import Lock
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'type': 'cooperative_semantic_communication',
        'velocity_threshold': 10.0,
        'flow_theory_threshold': 0.5
    },
    'environment': {
        'num_vehicles': 5,
        'num_cameras': 3,
        'image_resolution': (640, 480)
    }
}

# Define an Enum for agent types
class AgentType(Enum):
    COOPERATIVE_SEMANTIC_COMMUNICATION = 'cooperative_semantic_communication'

# Define a dataclass for agent configuration
@dataclass
class AgentConfig:
    name: str
    type: AgentType
    velocity_threshold: float
    flow_theory_threshold: float

# Define a dataclass for environment configuration
@dataclass
class EnvironmentConfig:
    num_vehicles: int
    num_cameras: int
    image_resolution: Tuple[int, int]

# Define a dataclass for configuration
@dataclass
class Config:
    agent: AgentConfig
    environment: EnvironmentConfig

# Define a lock for thread safety
lock = Lock()

# Define a context manager for configuration loading
@contextmanager
def load_config(config_file: str = CONFIG_FILE):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        yield config
    except FileNotFoundError:
        logger.warning(f'Config file not found: {config_file}')
        yield DEFAULT_CONFIG
    except yaml.YAMLError as e:
        logger.error(f'Error parsing config file: {e}')
        raise

# Define a function to load configuration
def load_config_file(config_file: str = CONFIG_FILE) -> Dict:
    with load_config(config_file) as config:
        return config

# Define a function to validate configuration
def validate_config(config: Dict) -> bool:
    try:
        agent_config = AgentConfig(**config['agent'])
        environment_config = EnvironmentConfig(**config['environment'])
        return True
    except (KeyError, ValueError) as e:
        logger.error(f'Invalid config: {e}')
        return False

# Define a function to save configuration
def save_config(config: Dict, config_file: str = CONFIG_FILE) -> None:
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Define a function to get configuration
def get_config() -> Config:
    config_file = load_config_file()
    if not validate_config(config_file):
        logger.error('Invalid config')
        return Config(AgentConfig(**DEFAULT_CONFIG['agent']), EnvironmentConfig(**DEFAULT_CONFIG['environment']))
    return Config(AgentConfig(**config_file['agent']), EnvironmentConfig(**config_file['environment']))

# Define a function to update configuration
def update_config(config: Dict) -> None:
    with lock:
        save_config(config)

# Define a function to get agent configuration
def get_agent_config() -> AgentConfig:
    config = get_config()
    return config.agent

# Define a function to get environment configuration
def get_environment_config() -> EnvironmentConfig:
    config = get_config()
    return config.environment

# Define a function to get image resolution
def get_image_resolution() -> Tuple[int, int]:
    config = get_environment_config()
    return config.image_resolution

# Define a function to get number of vehicles
def get_num_vehicles() -> int:
    config = get_environment_config()
    return config.num_vehicles

# Define a function to get number of cameras
def get_num_cameras() -> int:
    config = get_environment_config()
    return config.num_cameras

# Define a function to get velocity threshold
def get_velocity_threshold() -> float:
    config = get_agent_config()
    return config.velocity_threshold

# Define a function to get flow theory threshold
def get_flow_theory_threshold() -> float:
    config = get_agent_config()
    return config.flow_theory_threshold