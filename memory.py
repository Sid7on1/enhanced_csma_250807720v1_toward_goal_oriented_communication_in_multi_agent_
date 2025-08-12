import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, config: Config):
        self.config = config
        self.memory_size = config.memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.lock = Lock()

    def add_experience(self, experience: Dict):
        with self.lock:
            self.memory.append(experience)

    def sample_experiences(self, batch_size: int) -> List[Dict]:
        with self.lock:
            if len(self.memory) < batch_size:
                return []
            return list(np.random.choice(self.memory, size=batch_size, replace=False))

    def get_memory_size(self) -> int:
        with self.lock:
            return len(self.memory)

    def get_memory(self) -> List[Dict]:
        with self.lock:
            return list(self.memory)

class Experience:
    def __init__(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ExperienceReplayBuffer:
    def __init__(self, config: Config):
        self.config = config
        self.memory = Memory(config)
        self.experience_class = Experience

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = self.experience_class(state, action, reward, next_state, done)
        self.memory.add_experience(experience.__dict__)

    def sample_experiences(self, batch_size: int) -> List[Dict]:
        return self.memory.sample_experiences(batch_size)

    def get_memory_size(self) -> int:
        return self.memory.get_memory_size()

    def get_memory(self) -> List[Dict]:
        return self.memory.get_memory()

class ExperienceReplayBufferConfig(Config):
    def __init__(self):
        super().__init__()
        self.memory_size = 10000

class ExperienceReplayBufferAgent:
    def __init__(self, config: Config):
        self.config = config
        self.experience_replay_buffer = ExperienceReplayBuffer(ExperienceReplayBufferConfig())

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.experience_replay_buffer.add_experience(state, action, reward, next_state, done)

    def sample_experiences(self, batch_size: int) -> List[Dict]:
        return self.experience_replay_buffer.sample_experiences(batch_size)

    def get_memory_size(self) -> int:
        return self.experience_replay_buffer.get_memory_size()

    def get_memory(self) -> List[Dict]:
        return self.experience_replay_buffer.get_memory()

class ExperienceReplayBufferAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.experience_replay_buffer_config = ExperienceReplayBufferConfig()

class ExperienceReplayBufferAgentFactory:
    def __init__(self, config: Config):
        self.config = config

    def create_agent(self) -> ExperienceReplayBufferAgent:
        return ExperienceReplayBufferAgent(ExperienceReplayBufferAgentConfig())

def main():
    config = Config()
    agent_factory = ExperienceReplayBufferAgentFactory(config)
    agent = agent_factory.create_agent()

    state = np.random.rand(4)
    action = 1
    reward = 10.0
    next_state = np.random.rand(4)
    done = False

    agent.add_experience(state, action, reward, next_state, done)

    batch_size = 10
    experiences = agent.sample_experiences(batch_size)
    logger.info(f"Sampled {batch_size} experiences from memory")

    memory_size = agent.get_memory_size()
    logger.info(f"Memory size: {memory_size}")

    memory = agent.get_memory()
    logger.info(f"Memory: {memory}")

if __name__ == "__main__":
    main()