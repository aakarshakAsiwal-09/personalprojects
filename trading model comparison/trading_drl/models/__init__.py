"""
Combinatorial Deep Reinforcement Learning Models Package.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".
"""

from trading_drl.models.networks import (
    LSTMBackbone,
    DuelingQNetwork,
    DiscreteActorCriticNetwork,
    ContinuousActorCriticNetwork,
    PolicyGradientDiscreteNetwork,
    PolicyGradientContinuousNetwork,
)
from trading_drl.models.dqn import DQNAgent, ReplayBuffer
from trading_drl.models.pg import PolicyGradientsAgent
from trading_drl.models.a2c import A2CAgent

__all__ = [
    "LSTMBackbone",
    "DuelingQNetwork",
    "DiscreteActorCriticNetwork",
    "ContinuousActorCriticNetwork",
    "PolicyGradientDiscreteNetwork",
    "PolicyGradientContinuousNetwork",
    "DQNAgent",
    "ReplayBuffer",
    "PolicyGradientsAgent",
    "A2CAgent",
]
