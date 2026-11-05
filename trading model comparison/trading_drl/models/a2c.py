"""
Advantage Actor-Critic (A2C) for Combinatorial Discrete and Continuous Trading.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "RL Algorithms -> A2C", Equations (7), (8), (9) (Page 6).
  "The A2C is proposed to solve the training problem of PG by updating the policy in real time.
   It consists of two models: One is an actor network that outputs the policy, and the other
   is a critic network that measures how good the chosen action is in the given state...
   J(theta) = E [ log pi(A|S, theta) * A_adv(S, A) ]   (Equation 7)
   A_adv(S_t, A_t) = R_{t+1} + gamma * V(S_{t+1} | w) - V(S_t | w)   (Equation 8)
   J(w) = ( R_{t+1} + gamma * V(S_{t+1} | w) - V(S_t | w) )^2   (Equation 9)
   ...we adopt the synchronous approach and execute agents in parallel on multiple environments."
- Hyperparameters (Exhibit 1, Page 7):
  - alpha_critic = 0.001
  - alpha_actor = 0.0001
  - Optimizer = Adam
  - Batch Size = 128
  - gamma = 0.3
  - bp = 0.0020
- WHY (Decoupled Combinatorial Matrix):
  Supports both:
  1. Discrete Action Space: Discrete Categorical Actor + Critic V(s).
  2. Continuous Action Space: Continuous Gaussian Actor (bounded via Tanh in [-1, 1]) + Critic V(s).
  Provides lower variance updates than Monte Carlo PG via 1-step bootstrap temporal difference advantage.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading_drl.config import RLTrainingConfig, ModelConfig
from trading_drl.action_spaces import BaseActionSpace, ActionSpaceType
from trading_drl.models.networks import (
    DiscreteActorCriticNetwork,
    ContinuousActorCriticNetwork,
)


class A2CTransitionBuffer:
    """
    Rolling mini-batch transition buffer for synchronous A2C updates.
    
    WHERE: Exhibit 1: Batch Size = 128 (Page 7).
    """

    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.states: List[np.ndarray] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []

    def push(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

    def is_ready(self) -> bool:
        return len(self.rewards) >= self.batch_size

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()


class A2CAgent:
    """
    Combinatorial Synchronous Advantage Actor-Critic (A2C) supporting both Discrete and Continuous actions.
    """

    def __init__(
        self,
        input_dim: int,
        action_space: BaseActionSpace,
        train_config: Optional[RLTrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
        entropy_coef: float = 0.01,
    ):
        """
        WHERE:
        - Section: "RL Algorithms -> A2C" (Page 6), Exhibit 1 (Page 7).
        """
        self.input_dim = input_dim
        self.action_space = action_space
        self.train_cfg = train_config or RLTrainingConfig()
        self.model_cfg = model_config or ModelConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy_coef = entropy_coef

        # Instantiate discrete or continuous network
        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            num_actions = len(self.action_space.positions)  # 3: {-1, 0, 1}
            self.network = DiscreteActorCriticNetwork(
                input_dim=self.input_dim,
                num_actions=num_actions,
                config=self.model_cfg,
            ).to(self.device)
        else:
            self.network = ContinuousActorCriticNetwork(
                input_dim=self.input_dim,
                config=self.model_cfg,
            ).to(self.device)

        # Dual Adam optimizers with distinct learning rates
        # WHERE: Exhibit 1 (Page 7): alpha_critic = 0.001, alpha_actor = 0.0001
        actor_params = [p for n, p in self.network.named_parameters() if "critic" not in n]
        critic_params = [p for n, p in self.network.named_parameters() if "critic" in n]

        self.actor_optimizer = optim.Adam(actor_params, lr=self.train_cfg.a2c_actor_lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=self.train_cfg.a2c_critic_lr)

        # Mini-batch transition buffer (Batch Size = 128)
        self.buffer = A2CTransitionBuffer(batch_size=self.train_cfg.a2c_batch_size)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[Union[int, float], Dict[str, Any]]:
        """
        Selects action using actor distribution and queries critic value V(s).
        
        Returns:
            position: Target position in [-1.0, 1.0].
            step_data: Dictionary containing log_prob, value, entropy, raw_action for updating.
            
        WHY (Bug 2 Fix):
        - Enforces `self.network.eval()` and `torch.no_grad()` during evaluation/test-time rollouts,
          disabling LSTM dropout (p=0.20) and guaranteeing deterministic, noise-free predictions.
          Restores `self.network.train()` during training iterations.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if evaluate:
            self.network.eval()
            with torch.no_grad():
                dist, value = self.network(state_tensor)
                if self.action_space.space_type == ActionSpaceType.DISCRETE:
                    action_idx = int(torch.argmax(dist.probs, dim=-1).item())
                    position = self.action_space.to_position(action_idx)
                    raw_action = action_idx
                else:
                    action_tensor = dist.loc
                    clamped = torch.clamp(action_tensor, -1.0, 1.0)
                    position = float(clamped.item())
                    raw_action = position

                step_data = {
                    "log_prob": torch.tensor(0.0, device=self.device),
                    "value": value.squeeze(),
                    "entropy": torch.tensor(0.0, device=self.device),
                    "raw_action": raw_action,
                }
                return position, step_data

        # Training mode: dropout active for regularization
        self.network.train()
        dist, value = self.network(state_tensor)

        if self.action_space.space_type == ActionSpaceType.DISCRETE:
            action_tensor = dist.sample()
            action_idx = int(action_tensor.item())
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()
            position = self.action_space.to_position(action_idx)
            raw_action = action_idx
        else:
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).sum()
            entropy = dist.entropy().sum()
            clamped = torch.clamp(action_tensor, -1.0, 1.0)
            position = float(clamped.item())
            raw_action = position

        step_data = {
            "log_prob": log_prob,
            "value": value.squeeze(),
            "entropy": entropy,
            "raw_action": raw_action,
        }
        return position, step_data

    def record_transition(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        step_data: Dict[str, Any],
    ):
        """Stores transition step in rolling buffer."""
        self.buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=step_data["log_prob"],
            value=step_data["value"],
            entropy=step_data["entropy"],
        )

    def update(self, force: bool = False) -> Optional[Dict[str, float]]:
        """
        Performs synchronous temporal difference actor-critic gradient step.
        
        WHERE:
        - Section: "RL Algorithms -> A2C", Equations (7), (8), (9) (Page 6).
        - Equation (8): A_adv(S_t, A_t) = R_{t+1} + gamma * V(S_{t+1}|w) - V(S_t|w)
        - Equation (9): J(w) = ( R_{t+1} + gamma * V(S_{t+1}|w) - V(S_t|w) )^2
        - Equation (7): J(theta) = E [ log pi(A|S) * A_adv(S, A) ]
        
        WHY:
        - Temporal difference advantage evaluates immediate reward + discounted next state value
          against expected state baseline, yielding a low-variance gradient compared to pure Monte Carlo PG.
        - Critic updates minimize TD error MSE, while actor updates perform policy gradient ascent.
        """
        if not self.buffer.is_ready() and not (force and len(self.buffer.rewards) > 0):
            return None

        gamma = self.train_cfg.gamma

        # Compute next state values V(s_{t+1}) for TD target
        next_states_tensor = torch.tensor(
            np.array(self.buffer.next_states),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            _, next_values = self.network(next_states_tensor)
            next_values = next_values.squeeze(-1)

        rewards_tensor = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)

        # 1-step TD Target: R_{t+1} + gamma * V(S_{t+1}) * (1 - done)
        target_values = rewards_tensor + gamma * next_values * (1.0 - dones_tensor)

        current_values = torch.stack(self.buffer.values)
        log_probs = torch.stack(self.buffer.log_probs)
        entropies = torch.stack(self.buffer.entropies)

        # Advantage function (Equation 8): A_adv = Target - V(S_t)
        advantages = (target_values - current_values).detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Critic loss (Equation 9): MSE between Target and V(s)
        critic_loss = nn.functional.mse_loss(current_values, target_values)

        # Actor loss (Equation 7 with entropy bonus): - (log_prob * A_adv + entropy_coef * H)
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropies.mean()

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        loss_info = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "mean_advantage": float(advantages.mean().item()),
        }

        self.buffer.clear()
        return loss_info

    def save(self, filepath: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
