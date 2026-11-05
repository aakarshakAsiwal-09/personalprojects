"""
Modular Action Space Interface and Combinatorial Adapters.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Methodology -> Action space" (Page 4-5).
  "We study both discrete and continuous action spaces. For discrete action spaces,
   a simple action set of {-1, 0, 1} is used, and each value represents the position directly...
   This representation of action space is also known as target orders (Huang 2018)...
   The design of continuous action spaces is very similar to the discrete case in which we still
   output trade positions but allow actions to be any value between -1 and 1 (A_t in [-1, 1])."
- WHY: In the baseline paper, algorithms were rigid (DQN/PG were discrete, A2C was continuous).
  This module decouples action space representation from algorithm implementations,
  allowing an exhaustive combinatorial matrix:
  1. DQN + Discrete Action Space ({-1, 0, 1})
  2. DQN + Continuous Action Space (Discretized mesh / binning in [-1, 1])
  3. PG + Discrete Action Space (Categorical distribution)
  4. PG + Continuous Action Space (Continuous Gaussian / Tanh distribution in [-1, 1])
  5. A2C + Discrete Action Space (Actor Categorical + Critic V(s))
  6. A2C + Continuous Action Space (Actor Gaussian + Critic V(s))
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Tuple
import numpy as np


class ActionSpaceType(str, Enum):
    """
    Action space enumeration for combinatorial matrix experiments.
    """
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class BaseActionSpace(ABC):
    """
    Abstract Base Class for Trading Action Spaces.
    
    WHERE:
    - Section: "Methodology -> Action space" (Page 4).
    
    WHY:
    - Establishes a uniform contract for converting between agent output representations
      and target portfolio positions A_t.
    """

    @property
    @abstractmethod
    def space_type(self) -> ActionSpaceType:
        """Returns whether the action space is DISCRETE or CONTINUOUS."""
        pass

    @abstractmethod
    def sample(self) -> Union[int, float]:
        """Generates a random exploration action."""
        pass

    @abstractmethod
    def to_position(self, raw_action: Union[int, float, np.ndarray]) -> float:
        """
        Converts raw network or policy output into a financial target order position A_t in [-1.0, 1.0].
        """
        pass

    @abstractmethod
    def to_raw_action(self, position: float) -> Union[int, float]:
        """
        Converts a continuous portfolio position back to raw action format.
        """
        pass

    def clip_position(self, position: float) -> float:
        """Ensures target order stays strictly bounded in [-1.0, 1.0]."""
        return float(np.clip(position, -1.0, 1.0))


class DiscreteActionSpace(BaseActionSpace):
    """
    Discrete Action Space: {-1, 0, 1}.
    
    WHERE:
    - Section: "Methodology -> Action space" (Page 4).
      "For discrete action spaces, a simple action set of {-1, 0, 1} is used, and each value
       represents the position directly (i.e., -1 corresponds to a maximally short position,
       0 to no holdings, and 1 to a maximally long position)."
    
    WHY:
    - Direct position representation ("target orders", Huang 2018) avoids action-path dependencies.
    - Simplifies policy distribution into a 3-way categorical decision:
      index 0 -> short (-1.0), index 1 -> flat (0.0), index 2 -> long (+1.0).
    """

    def __init__(self, positions: Tuple[float, ...] = (-1.0, 0.0, 1.0)):
        """
        Initialize discrete positions. Default: (-1.0, 0.0, 1.0).
        """
        self._positions = list(positions)
        self._n_actions = len(self._positions)

    @property
    def space_type(self) -> ActionSpaceType:
        return ActionSpaceType.DISCRETE

    @property
    def n_actions(self) -> int:
        """Number of discrete action choices (default: 3)."""
        return self._n_actions

    @property
    def positions(self) -> List[float]:
        """List of target positions corresponding to discrete indices."""
        return self._positions

    def sample(self) -> int:
        """Sample uniform random discrete index."""
        return int(np.random.randint(0, self._n_actions))

    def to_position(self, raw_action: Union[int, float, np.ndarray]) -> float:
        """
        Maps discrete action index {0, 1, 2} or direct position {-1.0, 0.0, 1.0} to target position.
        
        WHERE:
        - Section: "Methodology -> Action space" (Page 4).
        """
        arr = np.asarray(raw_action)
        # If passed as a float that directly matches a target position
        if arr.dtype.kind == "f" and float(arr.item()) in self._positions:
            return float(arr.item())
        # Otherwise treat as integer action index {0, 1, ..., n-1}
        idx = int(arr.item())
        idx = int(np.clip(idx, 0, self._n_actions - 1))
        return self._positions[idx]

    def to_raw_action(self, position: float) -> int:
        """
        Finds the nearest discrete action index to a requested position.
        """
        pos = self.clip_position(position)
        differences = [abs(p - pos) for p in self._positions]
        return int(np.argmin(differences))


class ContinuousActionSpace(BaseActionSpace):
    """
    Continuous Action Space: A_t in [-1.0, 1.0].
    
    WHERE:
    - Section: "Methodology -> Action space" (Page 5).
      "The design of continuous action spaces is very similar to the discrete case in which
       we still output trade positions but allow actions to be any value between -1 and 1
       (A_t in [-1, 1])."
    
    WHY:
    - Allows fine-grained risk allocation and smooth position scaling rather than
      binary all-in / all-out bets, critical during consolidating or volatile market regimes.
    - For discrete-based architectures (like DQN), provides a fine-resolution discretization mesh
      (e.g., 21 bins from -1 to 1) so DQN can be evaluated in continuous action mode.
    """

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        discretization_bins: int = 21,
    ):
        """
        Args:
            low: Minimum position boundary (default -1.0).
            high: Maximum position boundary (default 1.0).
            discretization_bins: Number of bins for DQN binning adapter.
        """
        self.low = float(low)
        self.high = float(high)
        self.discretization_bins = discretization_bins
        # Equidistant mesh for DQN continuous adaptation
        self._mesh = np.linspace(self.low, self.high, self.discretization_bins, dtype=np.float32)

    @property
    def space_type(self) -> ActionSpaceType:
        return ActionSpaceType.CONTINUOUS

    @property
    def mesh(self) -> np.ndarray:
        """Discretization mesh for continuous action binning."""
        return self._mesh

    @property
    def n_bins(self) -> int:
        """Number of discretization bins."""
        return self.discretization_bins

    def sample(self) -> float:
        """Sample uniform random continuous position in [low, high]."""
        return float(np.random.uniform(self.low, self.high))

    def to_position(self, raw_action: Union[int, float, np.ndarray]) -> float:
        """
        Converts continuous raw output to position in [low, high].
        
        WHERE:
        - Section: "Methodology -> Action space" (Page 5).
        
        WHY (Bug 1 Fix):
        - Eliminates the previous dtype collision bug where whole numbers (e.g. 0, 0.0, 1.0)
          were misrouted into discrete bin indices (0 -> bin 0 -> -1.0, 1.0 -> bin 1 -> -0.9).
          Continuous positions in [-1.0, 1.0] are strictly preserved and clipped to [low, high].
          If conversion from a discrete DQN bin index to a continuous position is required,
          the caller explicitly uses `bin_index_to_position(bin_idx)`.
        """
        arr = np.asarray(raw_action)
        val = float(arr.item())
        return self.clip_position(val)

    def to_raw_action(self, position: float) -> float:
        """
        Clips continuous position to valid range [low, high].
        """
        return self.clip_position(float(position))

    def position_to_bin_index(self, position: float) -> int:
        """
        Finds the closest bin index in the discretization mesh.
        Used by DQN replay buffers when collecting continuous actions.
        """
        pos = self.clip_position(position)
        return int(np.argmin(np.abs(self._mesh - pos)))

    def bin_index_to_position(self, bin_idx: int) -> float:
        """Converts bin index to continuous position value."""
        idx = int(np.clip(bin_idx, 0, self.n_bins - 1))
        return float(self._mesh[idx])


def create_action_space(
    action_type: Union[ActionSpaceType, str],
    discretization_bins: int = 21,
) -> BaseActionSpace:
    """
    Factory function to instantiate either discrete or continuous action space.
    
    WHERE:
    - Core Combinatorial Matrix architecture.
    
    WHY:
    - Enables clean dynamic switching across the 6 experimental permutations.
    """
    action_type = ActionSpaceType(str(action_type).lower())
    if action_type == ActionSpaceType.DISCRETE:
        return DiscreteActionSpace()
    elif action_type == ActionSpaceType.CONTINUOUS:
        return ContinuousActionSpace(discretization_bins=discretization_bins)
    else:
        raise ValueError(f"Unsupported action space type: {action_type}")
