"""
Configuration Parameters and Hyperparameters.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Training Schemes for RL", Exhibit 1 ("Values of Hyperparameters for Different RL Algorithms"),
  and Section "State Space" / "Reward function".
- WHY: Centralizes all hyperparameter constants, network shapes, transaction cost rates,
  feature lookbacks, and training specifications into immutable/dataclass structures
  to guarantee reproducible quantitative experiments across all 6 algorithm-action permutations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class StateSpaceConfig:
    """
    Configuration for state space representation.
    
    WHERE:
    - Section: "Methodology -> State space" (Page 4).
    - Equation: (3) for MACD normalization.
    
    WHY:
    - Defines the lookback window (60 observations) and the parameters for multi-scale
      returns, MACD indicators, and RSI to guarantee that the agent receives rich
      temporal representations of momentum and mean-reversion signals.
    """
    # Number of past observations fed to LSTM (Page 4: "we take the past 60 observations of each feature to form a single state")
    sequence_length: int = 60

    # Horizons for volatility-adjusted returns (in trading days: 1M=21, 2M=42, 3M=63, 1Y=252)
    return_horizons: Tuple[int, ...] = (21, 42, 63, 252)

    # 60-day span for Exponentially Weighted Moving Average (EWMA) standard deviation
    volatility_ewma_span: int = 60

    # MACD short and long timescales (S in {8, 16, 32}, L in {24, 48, 96}) (Equation 3 & 12, Page 4 & 6)
    macd_timescales: Tuple[Tuple[int, int], ...] = (
        (8, 24),
        (16, 48),
        (32, 96),
    )

    # Rolling window for normalizing price standard deviation in MACD (Equation 3: std(p_{t-63:t}))
    macd_price_std_window: int = 63

    # Rolling window for normalizing MACD indicator (Equation 3: std(q_{t-252:t}))
    macd_signal_std_window: int = 252

    # RSI lookback window (Page 4: "We include this indicator with a look-back window of 30 days")
    rsi_window: int = 30


@dataclass
class RewardConfig:
    """
    Configuration for reward function with volatility scaling and transaction frictions.
    
    WHERE:
    - Section: "Methodology -> Reward function", Equation (4) (Page 5).
    - Section: Exhibit 1 (Page 7).
    
    WHY:
    - Volatility scaling normalizes risk across contracts and regimes so the agent is reward-driven
      by skill rather than market volatility fluctuations.
    - Transaction cost rate (bp) acts as a turnover regularizer, preventing excessive churn.
    """
    # Annualized target volatility (typically 15% in trend-following literature, e.g. Moskowitz et al. 2012)
    target_volatility_annual: float = 0.15

    # Trading days per year for annualization
    annualization_factor: int = 252

    # Multiplier per contract (Page 5: "mu is a fixed number per contract at each trade, and we set it to 1")
    mu: float = 1.0

    # Transaction cost rate in basis points (1 bp = 0.0001). Default from Exhibit 1: bp = 0.0020 (20 bps)
    basis_point_cost: float = 0.0020

    # 60-day EWMA volatility estimation window for returns
    volatility_ewma_span: int = 60

    # Profit calculation mode: 'additive' (r_t = p_t - p_{t-1}) or 'multiplicative' (r_t = (p_t - p_{t-1})/p_{t-1})
    profit_mode: str = "additive"

    @property
    def daily_target_volatility(self) -> float:
        """Daily target volatility sigma_tgt = annual_sigma / sqrt(252)."""
        return self.target_volatility_annual / (self.annualization_factor ** 0.5)


@dataclass
class ModelConfig:
    """
    Configuration for LSTM Neural Network Architecture.
    
    WHERE:
    - Section: "Training Schemes for RL" (Page 6-7).
    - Exhibit 1: Values of Hyperparameters for Different RL Algorithms.
    
    WHY:
    - 2-layer LSTM with (64, 32) units and LeakyReLU activations provides adequate capacity
      to capture temporal dependencies while mitigating overfitting on noisy financial data.
    """
    lstm_hidden_dim_1: int = 64
    lstm_hidden_dim_2: int = 32
    leaky_relu_slope: float = 0.01
    dropout_rate: float = 0.2
    
    # Continuous action discretization bins for continuous DQN mode (e.g. 21 bins in [-1, 1])
    dqn_continuous_bins: int = 21


@dataclass
class RLTrainingConfig:
    """
    Hyperparameters for RL Algorithms matching Exhibit 1.
    
    WHERE:
    - Exhibit 1: Values of Hyperparameters for Different RL Algorithms (Page 7).
    - Section: "Procedures for Controlling Overfitting" (Page 7).
    
    WHY:
    - Exact parameter matching with Exhibit 1 (DQN lr=1e-4, batch=64, gamma=0.3, memory=5000, tau=1000;
      PG lr=1e-4, gamma=0.3; A2C critic_lr=1e-3, actor_lr=1e-4, batch=128, gamma=0.3).
    - Early stopping patience of 20 epochs on 10% cross-validation set.
    """
    # Discount factor gamma (Page 7, Exhibit 1: gamma = 0.3 across all models)
    gamma: float = 0.3

    # Learning rates
    dqn_lr: float = 0.0001
    pg_actor_lr: float = 0.0001
    a2c_critic_lr: float = 0.001
    a2c_actor_lr: float = 0.0001

    # Batch sizes
    dqn_batch_size: int = 64
    a2c_batch_size: int = 128

    # DQN Experience Replay and Target Network
    dqn_memory_size: int = 5000
    dqn_target_update_tau: int = 1000  # Number of steps between target network syncs
    dqn_update_freq: int = 4           # Train Q-network every k environment steps (standard DQN setting)

    # Exploration for DQN
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Early stopping and validation (Bug 4 Fix)
    validation_split: float = 0.20     # Expanded from 10% to 20% (minimum 63-126 days) for statistical stability
    min_validation_steps: int = 63     # At least 1 quarter (63 days) to compute reliable risk-adjusted metrics
    early_stopping_patience: int = 20  # 20 epochs early stopping (Page 7)
    max_epochs: int = 100

    # Rolling walk-forward evaluation scheme (Page 6: 5-year out-of-sample evaluation, Bug 5 Fix)
    train_window_years: int = 5
    test_window_years: int = 5
    walk_forward_expanding: bool = True
