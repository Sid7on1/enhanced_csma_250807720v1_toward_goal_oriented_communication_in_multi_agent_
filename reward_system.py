import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions and the environment's state.
    It uses the Cooperative Semantic Communication (Co-SC) framework to extract semantic features from the data.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        :param config: Configuration object
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict) -> float:
        """
        Calculate the reward for the given state, action, and next state.

        :param state: Current state
        :param action: Action taken
        :param next_state: Next state
        :return: Reward value
        """
        try:
            # Calculate velocity and flow
            velocity = calculate_velocity(state, action, next_state)
            flow = calculate_flow(state, action, next_state)

            # Calculate reward using the reward model
            reward = self.reward_model.calculate_reward(velocity, flow)

            return reward
        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to fit the agent's learning curve.

        :param reward: Reward value
        :return: Shaped reward value
        """
        try:
            # Apply reward shaping using the Co-SC framework
            shaped_reward = self.config.reward_shaping * reward

            return shaped_reward
        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class RewardModel:
    """
    Reward model based on the Cooperative Semantic Communication (Co-SC) framework.

    This class uses the Flow Theory to calculate the reward.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        :param config: Configuration object
        """
        self.config = config

    def calculate_reward(self, velocity: float, flow: float) -> float:
        """
        Calculate the reward using the Flow Theory.

        :param velocity: Velocity value
        :param flow: Flow value
        :return: Reward value
        """
        try:
            # Calculate reward using the Flow Theory formula
            reward = self.config.flow_theory_reward * (velocity + flow)

            return reward
        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

def calculate_velocity(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the velocity based on the state, action, and next state.

    :param state: Current state
    :param action: Action taken
    :param next_state: Next state
    :return: Velocity value
    """
    try:
        # Calculate velocity using the velocity-threshold algorithm
        velocity = np.sqrt((next_state['x'] - state['x']) ** 2 + (next_state['y'] - state['y']) ** 2) / self.config.velocity_threshold

        return velocity
    except RewardSystemError as e:
        logger.error(f"Error calculating velocity: {e}")
        return 0.0

def calculate_flow(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the flow based on the state, action, and next state.

    :param state: Current state
    :param action: Action taken
    :param next_state: Next state
    :return: Flow value
    """
    try:
        # Calculate flow using the Flow Theory formula
        flow = np.sqrt((next_state['x'] - state['x']) ** 2 + (next_state['y'] - state['y']) ** 2) / self.config.flow_threshold

        return flow