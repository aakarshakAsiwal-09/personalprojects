"""
Neural Network Architectures for Deep Reinforcement Learning in Trading.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Training Schemes for RL" (Page 6-7).
  "We use two-layer LSTM networks with 64 and 32 units in all models, and leaky rectifying linear
   units (Leaky-ReLU) (Maas, Hannun, and Ng 2013) are used as activation functions...
   In addition, we use dropout (Hinton et al. 2016), a regularization method commonly employed
   in deep learning... Dueling DQNs separate the Q-value into state value and the advantage
   of each action (Wang et al. 2016)."
- WHY:
  1. Temporal Modeling: LSTMs effectively capture persistent momentum and long-range dependencies
     over the 60-step lookback window without vanishing gradients.
  2. Overfitting Mitigation: Financial data has notoriously low signal-to-noise ratios.
     Constraining the architecture to a compact (64, 32) footprint with Dropout (0.2)
     prevents memorization of spurious historical price regimes.
  3. Dueling Decomposition: Decoupling state value V(s) from action advantages A(s, a) enables
     the network to evaluate market states (bull/bear/neutral) independently of specific trades,
     accelerating value convergence.
  4. Combinatorial Matrix Heads: Provides dedicated, decoupled distribution heads for both
     Discrete action spaces (Categorical) and Continuous action spaces (Tanh-Gaussian).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from trading_drl.config import ModelConfig


class LSTMBackbone(nn.Module):
    """
    Two-layer LSTM feature extractor with LeakyReLU and Dropout.
    
    WHERE:
    - Section: "Training Schemes for RL" (Page 6-7):
      "We use two-layer LSTM networks with 64 and 32 units in all models, and leaky rectifying
       linear units (Leaky-ReLU) (Maas, Hannun, and Ng 2013) are used as activation functions."
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout_rate: float = 0.2,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.leaky_slope = leaky_slope

        # First LSTM layer: input_dim -> 64
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            batch_first=True,
        )
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Second LSTM layer: 64 -> 32
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim_1,
            hidden_size=hidden_dim_2,
            batch_first=True,
        )
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2-layer LSTM backbone.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length=60, input_dim).
            
        Returns:
            Latent representation of shape (batch_size, hidden_dim_2=32).
        """
        # Ensure 3D tensor
        if x.dim() == 2:
            x = x.unsqueeze(0)

        out1, _ = self.lstm1(x)
        out1 = self.act1(out1)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.act2(out2)
        out2 = self.dropout2(out2)

        # Take final temporal step's hidden state
        # out2 shape: (batch_size, seq_len, 32) -> take out2[:, -1, :]
        features = out2[:, -1, :]
        return features


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network with 2-layer LSTM backbone.
    
    WHERE:
    - Section: "RL Algorithms -> DQN" (Page 5).
    - Citation: Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning".
    
    Formula:
    Q(s, a) = V(s) + [ A(s, a) - (1 / |A|) * sum_{a'} A(s, a') ]
    
    WHY:
    - In trading, identifying general market state value V(s) is often more informative
      than evaluating subtle differences among actions during quiet periods.
      Subtracting the mean advantage guarantees identifiability so V(s) and A(s, a)
      converge uniquely without drift.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int = 3,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        cfg = config or ModelConfig()
        self.num_actions = num_actions

        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim_1=cfg.lstm_hidden_dim_1,
            hidden_dim_2=cfg.lstm_hidden_dim_2,
            dropout_rate=cfg.dropout_rate,
            leaky_slope=cfg.leaky_relu_slope,
        )

        # State Value Stream V(s): 32 -> 16 -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, 1),
        )

        # Advantage Stream A(s, a): 32 -> 16 -> num_actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates Q(s, a) for all actions via dueling aggregation.
        
        Returns:
            Tensor of shape (batch_size, num_actions).
        """
        features = self.backbone(x)
        val = self.value_stream(features)  # (batch_size, 1)
        adv = self.advantage_stream(features)  # (batch_size, num_actions)

        # Identifiability constraint: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_values


class DiscreteActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for Discrete Action Spaces {-1, 0, 1}.
    
    WHERE:
    - Section: "RL Algorithms -> A2C" (Page 6).
    - Equation: (7), (8), (9).
    
    WHY:
    - Simultaneously predicts categorical policy distribution pi(a|s) and scalar value V(s).
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int = 3,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        cfg = config or ModelConfig()
        self.num_actions = num_actions

        # Shared LSTM backbone
        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim_1=cfg.lstm_hidden_dim_1,
            hidden_dim_2=cfg.lstm_hidden_dim_2,
            dropout_rate=cfg.dropout_rate,
            leaky_slope=cfg.leaky_relu_slope,
        )

        # Actor head: 32 -> num_actions (logits)
        self.actor_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, num_actions),
        )

        # Critic head: 32 -> 1 (scalar V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Returns:
            dist: torch.distributions.Categorical over discrete actions.
            value: scalar baseline V(s) of shape (batch_size, 1).
        """
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        dist = Categorical(logits=logits)
        return dist, value


class ContinuousActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for Continuous Action Spaces [-1.0, 1.0].
    
    WHERE:
    - Section: "RL Algorithms -> A2C" (Page 6).
      "The A2C is most useful if we are interested in continuous action spaces because
       we reduce the policy variance by using the advantage function and update the policy in real time."
    
    WHY:
    - Outputs Gaussian distribution parameter mean mu(s) through Tanh to strictly bound
      portfolio positions within [-1.0, 1.0], accompanied by a learnable log standard deviation.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ModelConfig] = None,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        cfg = config or ModelConfig()

        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim_1=cfg.lstm_hidden_dim_1,
            hidden_dim_2=cfg.lstm_hidden_dim_2,
            dropout_rate=cfg.dropout_rate,
            leaky_slope=cfg.leaky_relu_slope,
        )

        # Actor mean head: 32 -> 16 -> 1 with Tanh activation
        self.actor_mean = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, 1),
            nn.Tanh(),  # Strictly forces mu in [-1.0, 1.0]
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(1) * init_log_std)

        # Critic head: 32 -> 1 (scalar V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Returns:
            dist: torch.distributions.Normal over continuous target position.
            value: scalar baseline V(s) of shape (batch_size, 1).
        """
        features = self.backbone(x)
        mu = self.actor_mean(features)
        # Clamped standard deviation to prevent explosion / collapse
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        dist = Normal(loc=mu, scale=std)
        value = self.critic_head(features)
        return dist, value


class PolicyGradientDiscreteNetwork(nn.Module):
    """
    Policy Network for Discrete Policy Gradients (REINFORCE).
    
    WHERE:
    - Section: "RL Algorithms -> PG", Equation (6) (Page 5).
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int = 3,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        cfg = config or ModelConfig()
        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim_1=cfg.lstm_hidden_dim_1,
            hidden_dim_2=cfg.lstm_hidden_dim_2,
            dropout_rate=cfg.dropout_rate,
            leaky_slope=cfg.leaky_relu_slope,
        )
        self.logits_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, num_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        features = self.backbone(x)
        logits = self.logits_head(features)
        return Categorical(logits=logits)


class PolicyGradientContinuousNetwork(nn.Module):
    """
    Policy Network for Continuous Policy Gradients (Continuous REINFORCE).
    
    WHERE:
    - Section: "RL Algorithms -> PG", Equation (6) (Page 5-6).
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ModelConfig] = None,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        cfg = config or ModelConfig()
        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim_1=cfg.lstm_hidden_dim_1,
            hidden_dim_2=cfg.lstm_hidden_dim_2,
            dropout_rate=cfg.dropout_rate,
            leaky_slope=cfg.leaky_relu_slope,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim_2, 16),
            nn.LeakyReLU(negative_slope=cfg.leaky_relu_slope),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1) * init_log_std)

    def forward(self, x: torch.Tensor) -> Normal:
        features = self.backbone(x)
        mu = self.mean_head(features)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        return Normal(loc=mu, scale=std)
