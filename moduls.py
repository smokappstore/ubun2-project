from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


# Base trainer class for training-related modules like MiAgente.
class BaseTrainer:

    def __init__(self, environment: Any, gamma: float = 0.99):
        """
        Initializes the BaseTrainer with environment and discount factor.

        Args:
            environment (Any): The environment to train in.
            gamma (float): The discount factor for rewards.
        """
        self.reward_storage = None
        self.value_estimates = None
        self.log_probabilities = None
        self.optimizer_critic = None
        self.prediction_module = None
        self.optimizer_prediction = None
        self.optimizer_actor = None
        self.environment = environment
        self.gamma = gamma
        # Actor-Critic module
        self.actor = self._create_actor_network(environment.observation_dim, environment.action_dim)
        self.critic = self._create_critic_network(environment.observation_dim, 1)
        lr_actor_critic = 1e-3  # Default learning rate for actor-critic
        lr_prediction = 1e-3  # Default learning rate for prediction module
        self.initialize_optimizers(lr_actor_critic, lr_prediction)

    def initialize_optimizers(self, lr_actor_critic: float, lr_prediction: float) -> None:
        """
        Initializes the optimizers for the actor, critic, and prediction network.

        Args:
            lr_actor_critic (float): Learning rate for the actor and critic.
            lr_prediction (float): Learning rate for the prediction network.
        """
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor_critic)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_actor_critic)

        # Future state prediction module
        self.prediction_module = self._create_prediction_network(self.environment.observation_dim,
                                                                 self.environment.observation_dim
                                                                 )
        self.optimizer_prediction = optim.Adam(self.prediction_module.parameters(), lr=lr_prediction)

        # Training-related variables
        self.log_probabilities = []
        self.reward_storage = []
        self.value_estimates = []

        value = self.critic(torch.tensor([0.0], dtype=torch.float32))
        if isinstance(value, torch.Tensor):
            self.value_estimates.append(value.item())
        else:
            self.value_estimates.append(value)
        action_probabilities = self.actor(torch.tensor([0.0], dtype=torch.float32))  # Replace with actual state tensor

        action = torch.distributions.Categorical(action_probabilities).sample()
        self.log_probabilities.append(torch.distributions.Categorical(action_probabilities).log_prob(action))

    def _reset_training_data(self):
        """
        Clears logs and training data.
        """
        self.log_probabilities.clear()
        self.reward_storage.clear()
        self.value_estimates.clear()

    @staticmethod
    def _create_actor_network(input_dim: int, output_dim: int) -> nn.Module:
        """
        Creates the Actor network.

        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.

        Returns:
            nn.Module: The actor network.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    @staticmethod
    def _create_critic_network(input_dim: int, output_dim: int) -> nn.Module:
        """
        Creates the Critic network.

        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.

        Returns:
            nn.Module: The critic network.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    @staticmethod
    def _create_prediction_network(input_dim: int, output_dim: int) -> nn.Module:
        """Creates a prediction network (optional: RNN, LSTM, GRU-based)."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def select_action(self, current_state):
        """
        Selects an action based on the Actor policy.

        Args:
            current_state (Any): The current state of the environment.

        Returns:
            int: The selected action.
        """
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32)
        action_probabilities = self.actor(current_state_tensor)
        action_distribution = torch.distributions.Categorical(action_probabilities)
        selected_action = action_distribution.sample()
        self.log_probabilities.append(action_distribution.log_prob(selected_action))

        return selected_action.item()