"""
Policy Gradients (PG / REINFORCE) for Combinatorial Discrete and Continuous Trading.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "RL Algorithms -> PG", Equations (2) and (6) (Page 5-6).
  "The PG aims to maximize the expected cumulative rewards by optimizing the policy directly...
   J(theta) = E [ sum_{t=0}^{T-1} R_{t+1} | pi_theta ]
   grad_theta J(theta) = sum_{t=0}^{T-1} grad_theta log pi_theta(A_t | S_t) * G_t
   where G_t is defined in Equation 2: G_t = sum_{k=t+1}^T gamma^{k-t-1} * R_k...
   Compared with DQN, PG learns a policy directly and can output a probability distribution over actions.
   This is useful if we want to design stochastic policies or work with continuous action spaces."
- Hyperparameters (Exhibit 1, Page 7):
  - alpha_actor = 0.0001
  - Optimizer = Adam
  - gamma = 0.3
  - bp = 0.0020
- WHY (Decoupled Combinatorial Matrix):
  Supports both:
  1. Discrete Action Space: Categorical distribution over {-1, 0, 1}.
  2. Continuous Action Space: Continuous Gaussian policy with mean bounded in [-1, 1] via Tanh,
     enabling direct analytical score-function policy gradient optimization without discretization.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading_drl.config import RLTrainingConfig, ModelConfig
from trading_drl.action_spaces import BaseActionSpace, ActionSpaceType
from trading_drl.models.networks import (
    PolicyGradientDiscreteNetwork,
    PolicyGradientContinuousNetwork,
)


class PolicyGradientsAgent:
    """
    Combinatorial Policy Gradients (REINFORCE) supporting both Discrete and Continuous action spaces.
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
        - Section: "RL Algorithms -> PG" (Page 5-6), Exhibit 1 (Page 7).
        """
        self.input_dim = input_dim
        self.action_space = action_space
        self.train_cfg = train_config or RLTrainingConfig()
        self.model_cfg = model_config or ModelConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate appropriate policy network based on action space type
        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            num_actions = len(self.action_space.positions)  # 3: {-1, 0, 1}
            self.policy_net = PolicyGradientDiscreteNetwork(
                input_dim=self.input_dim,
                num_actions=num_actions,
                config=self.model_cfg,
            ).to(self.device)
        else:
            self.policy_net = PolicyGradientContinuousNetwork(
                input_dim=self.input_dim,
                config=self.model_cfg,
            ).to(self.device)

        # Adam Optimizer
        # WHERE: Exhibit 1: alpha_actor = 0.0001, Optimizer = Adam
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.train_cfg.pg_actor_lr)

        # Trajectory episodic storage
        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[Union[int, float], Any]:
        """
        Samples an action from the stochastic policy pi_theta(A|S).
        
        Args:
            state: Array of shape (60, input_dim).
            evaluate: If True, disables dropout (.eval()), selects deterministic action, and skips autograd.
            
        Returns:
            position: Target portfolio position in [-1.0, 1.0].
            raw_action: Sampled discrete action index or continuous float.
            
        WHY (Bug 2 Fix):
        - Enforces `self.policy_net.eval()` and `torch.no_grad()` during evaluation/test-time rollouts,
          disabling LSTM dropout (p=0.20) and guaranteeing deterministic, noise-free predictions.
          Restores `self.policy_net.train()` during training iterations.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if evaluate:
            self.policy_net.eval()
            with torch.no_grad():
                if self.action_space.space_type == ActionSpaceType.DISCRETE:
                    dist = self.policy_net(state_tensor)
                    action_idx = int(torch.argmax(dist.probs, dim=-1).item())
                    position = self.action_space.to_position(action_idx)
                    return position, action_idx
                else:
                    dist = self.policy_net(state_tensor)
                    action_val = dist.loc
                    clamped_action = torch.clamp(action_val, -1.0, 1.0)
                    position = float(clamped_action.item())
                    return position, position

        # Training mode: dropout active for regularization
        self.policy_net.train()
        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            dist = self.policy_net(state_tensor)
            action_tensor = dist.sample()
            action_idx = int(action_tensor.item())
            log_prob = dist.log_prob(action_tensor)
            self.saved_log_probs.append(log_prob)
            position = self.action_space.to_position(action_idx)
            return position, action_idx
        else:
            dist = self.policy_net(state_tensor)
            action_val = dist.sample()
            log_prob = dist.log_prob(action_val)
            clamped_action = torch.clamp(action_val, -1.0, 1.0)
            self.saved_log_probs.append(log_prob.sum())
            position = float(clamped_action.item())
            return position, position

    def record_reward(self, reward: float):
        """Records immediate step reward R_{t+1}."""
        self.rewards.append(float(reward))

    def update(self) -> Optional[float]:
        """
        Performs Monte Carlo Policy Gradient update at end of trajectory.
        
        WHERE:
        - Section: "RL Algorithms -> PG", Equation (6) (Page 5).
          grad_theta J(theta) = sum_{t=0}^{T-1} grad_theta log pi_theta(A_t | S_t) * G_t
        - Equation (2), Page 4:
          G_t = sum_{k=t+1}^T gamma^{k-t-1} * R_k
        
        WHY:
        - Discounted returns-to-go G_t attribute credit only to future rewards following an action,
          reducing the variance of the gradient estimator compared to total episode returns.
        - Baseline standardization (z-score normalization of G_t) further stabilizes training
          without introducing bias into the gradient expectation.
        """
        if len(self.rewards) == 0 or len(self.saved_log_probs) == 0:
            return None

        gamma = self.train_cfg.gamma
        returns_to_go = []
        discounted_return = 0.0

        # Calculate discounted return-to-go backward from T to 0 (Equation 2)
        for r in reversed(self.rewards):
            discounted_return = r + gamma * discounted_return
            returns_to_go.insert(0, discounted_return)

        returns = torch.tensor(returns_to_go, dtype=torch.float32, device=self.device)

        # Variance reduction via standardization
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss: - sum_t (log pi(a_t | s_t) * G_t)
        policy_loss = []
        for log_prob, g_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * g_t)

        loss = torch.stack(policy_loss).sum()

        # Gradient ascent step (minimizing negative expected reward)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Reset trajectory storage
        self.saved_log_probs.clear()
        self.rewards.clear()

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
