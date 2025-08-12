import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agents.utils import Experience, Transition, update_network
from models.critic import Critic

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        std_init: float = 0.3,
    ):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.std_init = std_init

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_mean.weight.data.uniform_(-0.003, 0.003)
        self.fc_mean.bias.data.uniform_(-0.003, 0.003)
        self.fc_log_std.weight.data.uniform_(-0.003, 0.003)
        self.fc_log_std.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(state))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        return action, log_prob

    def to(self, device):
        self.device = device
        self.fc1.to(device)
        self.fc_mean.to(device)
        self.fc_log_std.to(device)


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        critic_learning_rate: float = 3e-4,
        policy_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        policy_hidden_dim: int = 256,
        policy_std_init: float = 0.3,
        weight_decay: float = 0.0,
        policy_update_frequency: int = 2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_frequency = policy_update_frequency

        # Experience buffer
        self.buffer = Experience(buffer_capacity)

        # Networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate, weight_decay=weight_decay
        )

        self.policy = Policy(
            state_dim, action_dim, max_action, policy_hidden_dim, policy_std_init
        ).to(device)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=policy_learning_rate, weight_decay=weight_decay
        )

        # Copy target network weights
        update_network(self.critic_target, self.critic, self.tau)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.policy(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t.detach().cpu().numpy() * self.max_action
        return action

    def update_critic(self, experiences: List[Experience]) -> torch.Tensor:
        states, actions, rewards, next_states, dones = (
            torch.FloatTensor(np.vstack(experiences.states)),
            torch.FloatTensor(np.vstack(experiences.actions)),
            torch.FloatTensor(np.vstack(experiences.rewards)),
            torch.FloatTensor(np.vstack(experiences.next_states)),
            torch.FloatTensor(np.vstack(experiences.dones)),
        ).to(device)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_states)
            q_next = self.critic_target(next_states, next_state_action)
            q_next = q_next.view(-1, 1)
            next_q_value = rewards + (1.0 - dones) * self.gamma * q_next

        current_q_value = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q_value, next_q_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target network
        update_network(self.critic_target, self.critic, self.tau)

        return critic_loss.item()

    def update_policy(self, experiences: List[Experience]) -> Tuple[torch.Tensor, ...]:
        states = torch.FloatTensor(np.vstack(experiences.states)).to(device)
        old_actions = torch.FloatTensor(np.vstack(experiences.actions)).to(device)

        pi, log_pi = self.policy.sample(states)
        q_value = self.critic(states, pi)

        policy_loss = (-q_value - log_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), log_pi.mean().item()

    def train(
        self,
        dataset: DataLoader,
        epochs: int,
        log_interval: int = 100,
        save_dir: Optional[str] = None,
        model_prefix: str = "sac_model",
    ):
        os.makedirs(save_dir, exist_ok=True)

        total_steps = len(dataset) * epochs

        logger.info(f"Starting training for {total_steps} steps")

        global_step = 0
        epoch_losses = []

        writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

        for epoch in range(epochs):
            for i, transitions in enumerate(dataset):
                global_step += 1

                self.buffer.add(transitions)

                if len(self.buffer) < self.batch_size:
                    continue

                batch = self.buffer.sample(self.batch_size)

                critic_loss = self.update_critic(batch)

                epoch_losses.append(critic_loss)

                if global_step % self.policy_update_frequency == 0:
                    policy_loss, log_pi = self.update_policy(batch)

                    writer.add_scalar("losses/critic_loss", critic_loss, global_step)
                    writer.add_scalar("losses/policy_loss", policy_loss, global_step)
                    writer.add_scalar("metrics/log_pi", log_pi, global_step)

                    logger.debug(
                        f"Step {global_step}: Critic Loss: {critic_loss:.4f}, Policy Loss: {policy_loss:.4f}, Log Pi: {log_pi:.4f}"
                    )

                if global_step % log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}], Step [{global_step}/{total_steps}], Critic Loss: {np.mean(epoch_losses):.4f}"
                    )
                    epoch_losses = []

            # Save model checkpoints
            if epoch % 10 == 0:
                torch.save(
                    self.critic.state_dict(), os.path.join(save_dir, f"{model_prefix}_critic.pth")
                )
                torch.save(
                    self.policy.state_dict(), os.path.join(save_dir, f"{model_prefix}_policy.pth")
                )

        writer.close()

    def save(self, save_dir: str, model_prefix: str = "sac_model"):
        torch.save(self.critic.state_dict(), os.path.join(save_dir, f"{model_prefix}_critic.pth"))
        torch.save(self.policy.state_dict(), os.path.join(save_dir, f"{model_prefix}_policy.pth"))

    def load(self, load_dir: str, model_prefix: str = "sac_model"):
        self.critic.load_state_dict(
            torch.load(os.path.join(load_dir, f"{model_prefix}_critic.pth"), map_location=device)
        )
        self.policy.load_state_dict(
            torch.load(os.path.join(load_dir, f"{model_prefix}_policy.pth"), map_location=device)
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Pendulum-v0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--model_prefix", type=str, default="sac_model")
    args = parser.parse_args()

    env = gym.make(args.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = Agent(
        state_dim,
        action_dim,
        max_action,
        batch_size=128,
        policy_hidden_dim=256,
        policy_std_init=0.3,
        policy_update_frequency=2,
    )

    dataset = DataLoader(
        Transition(
            np.random.random((1000, state_dim)),
            np.random.random((1000, action_dim)),
            np.random.random(1000),
            np.random.random((1000, state_dim)),
            np.random.randint(0, 2, size=(1000,)),
        ),
        batch_size=64,
        shuffle=True,
    )

    agent.train(dataset, args.epochs, args.log_interval, args.save_dir, args.model_prefix)


if __name__ == "__main__":
    main()