"""
Deep Q-Network (DQN) with Fixed Targets, Double DQN, and Dueling Architecture.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "RL Algorithms -> DQN", Equation (5) (Page 5).
  "By adopting a neural network, a DQN approximates the state-action value function (Q function)...
   L(theta) = E [ ( Q_theta(S, A) - Q*_theta(S, A) )^2 ]
   Q*_theta(S, A) = r + gamma * argmax_{A_{t+1}} Q_theta(S_{t+1}, A_{t+1})
   ...we adopt the following three strategies to improve the training in our work:
   fixed Q-targets (Van Hasselt, Guez, and Silver 2016), double DQN (Hasselt 2010),
   and dueling DQN (Wang et al. 2016). Fixed Q-targets and double DQN are used to reduce policy variances
   and to solve the problem of chasing tails... Dueling DQNs separate the Q-value into state value
   and the advantage of each action."
- Hyperparameters (Exhibit 1, Page 7):
  - alpha_critic = 0.0001
  - Optimizer = Adam
  - Batch Size = 64
  - gamma = 0.3
  - Memory Size = 5,000
  - tau = 1,000
- WHY (Decoupled Combinatorial Matrix):
  Allows DQN to be paired with:
  1. Discrete Action Space: Direct Q-values for positions {-1, 0, 1}.
  2. Continuous Action Space: Discretized fine-resolution action mesh / binning in [-1, 1]
     (e.g., 21 bins), mapping continuous positions to optimal discrete bin approximations
     and executing continuous target orders.
"""

import random
from collections import deque
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading_drl.config import RLTrainingConfig, ModelConfig
from trading_drl.action_spaces import BaseActionSpace, DiscreteActionSpace, ContinuousActionSpace, ActionSpaceType
from trading_drl.models.networks import DuelingQNetwork


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    WHERE:
    - Section: "RL Algorithms -> DQN" (Page 5), Exhibit 1 (Page 7: Memory Size = 5,000).
    
    WHY:
    - Breaks temporal correlation among sequential financial transitions,
      satisfying the i.i.d. assumption necessary for stable SGD updates.
    """

    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, float(reward), next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.int64),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Production-grade Combinatorial Double Dueling DQN with Fixed Q-Targets.
    """

    def __init__(
        self,
        input_dim: int,
        action_space: BaseActionSpace,
        train_config: Optional[RLTrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        WHERE:
        - Section: "RL Algorithms -> DQN" (Page 5), Exhibit 1 (Page 7).
        """
        self.input_dim = input_dim
        self.action_space = action_space
        self.train_cfg = train_config or RLTrainingConfig()
        self.model_cfg = model_config or ModelConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine number of Q-network output actions
        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            self.num_actions = len(self.action_space.positions)  # typically 3: {-1, 0, 1}
        else:
            self.num_actions = self.action_space.n_bins  # e.g., 21 bins for continuous mesh

        # Online Network Q_theta
        self.q_network = DuelingQNetwork(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            config=self.model_cfg,
        ).to(self.device)

        # Target Network Q_target
        # WHERE: Page 5: "Fixed Q-targets (Van Hasselt et al. 2016)... uses a separate network to produce target values"
        self.target_network = DuelingQNetwork(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            config=self.model_cfg,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Adam Optimizer
        # WHERE: Exhibit 1: alpha_critic = 0.0001, Optimizer = Adam
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.train_cfg.dqn_lr)
        self.loss_fn = nn.MSELoss()

        # Experience Replay
        # WHERE: Exhibit 1: Memory Size = 5,000
        self.memory = ReplayBuffer(capacity=self.train_cfg.dqn_memory_size)

        # Exploration Parameters
        self.epsilon = self.train_cfg.epsilon_start
        self.epsilon_end = self.train_cfg.epsilon_end
        self.epsilon_decay = self.train_cfg.epsilon_decay

        self.step_count = 0
        self.tau = self.train_cfg.dqn_target_update_tau  # tau = 1,000 steps

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[Union[int, float], int]:
        """
        Selects trading action using epsilon-greedy policy during training or greedy argmax during eval.
        
        Returns:
            position: Floating point portfolio target order in [-1.0, 1.0].
            raw_action_idx: Integer action index used for replay buffer storage.
        """
        # Epsilon-greedy exploration
        if not evaluate and random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                q_vals = self.q_network(state_tensor)
                action_idx = int(torch.argmax(q_vals, dim=-1).item())

        # Map action index to target position
        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            position = self.action_space.to_position(action_idx)
        else:
            position = self.action_space.bin_index_to_position(action_idx)

        return position, action_idx

    def update(self) -> Optional[float]:
        """
        Performs one Double DQN optimization step over a minibatch from experience replay.
        
        WHERE:
        - Section: "RL Algorithms -> DQN", Equation (5) (Page 5).
        - Citations: Hasselt (2010), Van Hasselt et al. (2016).
        
        Formula:
        a* = argmax_{a'} Q_online(s', a')
        Target = r + gamma * Q_target(s', a*) * (1 - done)
        Loss = MSE(Q_online(s, a), Target)
        
        WHY:
        - Double DQN decouples action selection from action evaluation.
          Standard Q-learning suffers from maximization bias (chasing tails) because max_a Q(s', a)
          tends to systematically overestimate values in noisy environments.
          Double DQN uses the online network to select the best action and the target network
          to evaluate its value, substantially reducing overestimation.
        """
        if len(self.memory) < self.train_cfg.dqn_batch_size:
            return None

        self.q_network.train()
        states, actions, rewards, next_states, dones = self.memory.sample(self.train_cfg.dqn_batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current predicted Q(s, a)
        curr_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 1. Action selection using online network: a* = argmax_a Q_online(s', a)
            next_online_q = self.q_network(next_states)
            next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

            # 2. Action evaluation using target network: Q_target(s', a*)
            next_target_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)

            # 3. Target calculation (Equation 5 with discount gamma=0.3)
            gamma = self.train_cfg.gamma
            target_q = rewards + (1.0 - dones) * gamma * next_target_q

        # Compute TD error loss
        loss = self.loss_fn(curr_q, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Step count & target network synchronization
        # WHERE: Exhibit 1: tau = 1,000 steps
        self.step_count += 1
        if self.step_count % self.tau == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay exploration epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
