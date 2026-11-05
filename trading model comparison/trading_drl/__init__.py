"""
Deep Reinforcement Learning for Trading Framework.
Based on Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".
Journal of Financial Data Science, Spring 2020.
"""

from trading_drl.config import (
    StateSpaceConfig,
    RewardConfig,
    ModelConfig,
    RLTrainingConfig,
)
from trading_drl.action_spaces import (
    ActionSpaceType,
    BaseActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
    create_action_space,
)
from trading_drl.data_pipeline import DataPipeline
from trading_drl.environment import TradingEnvironment
from trading_drl.baselines import (
    LongOnlyStrategy,
    SignRStrategy,
    MACDSignalStrategy,
)
from trading_drl.models.networks import (
    LSTMBackbone,
    DuelingQNetwork,
    DiscreteActorCriticNetwork,
    ContinuousActorCriticNetwork,
    PolicyGradientDiscreteNetwork,
    PolicyGradientContinuousNetwork,
)
from trading_drl.models.dqn import DQNAgent
from trading_drl.models.pg import PolicyGradientsAgent
from trading_drl.models.a2c import A2CAgent
from trading_drl.seed import set_global_seed
from trading_drl.train import ModelTrainer, TimeSeriesSplitter, WalkForwardSplitter
from trading_drl.evaluate import PortfolioEvaluator

__version__ = "1.0.0"

__all__ = [
    "set_global_seed",
    "StateSpaceConfig",
    "RewardConfig",
    "ModelConfig",
    "RLTrainingConfig",
    "ActionSpaceType",
    "BaseActionSpace",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "create_action_space",
    "DataPipeline",
    "TradingEnvironment",
    "LongOnlyStrategy",
    "SignRStrategy",
    "MACDSignalStrategy",
    "LSTMBackbone",
    "DuelingQNetwork",
    "DiscreteActorCriticNetwork",
    "ContinuousActorCriticNetwork",
    "PolicyGradientDiscreteNetwork",
    "PolicyGradientContinuousNetwork",
    "DQNAgent",
    "PolicyGradientsAgent",
    "A2CAgent",
    "ModelTrainer",
    "TimeSeriesSplitter",
    "WalkForwardSplitter",
    "PortfolioEvaluator",
]
