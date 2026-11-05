"""
Trading Environment with Volatility Scaling and Transaction Frictions.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Methodology -> Reward function", Equation (4) (Page 5):
  "The reward R_t at time t is:
   R_t = mu * [ A_{t-1} * (sigma_tgt / sigma_{t-1}) * r_t
               - bp * p_{t-1} * | (sigma_tgt / sigma_{t-1}) * A_{t-1} - (sigma_tgt / sigma_{t-2}) * A_{t-2} | ]
   where sigma_tgt is the volatility target and sigma_{t-1} is an ex ante volatility estimate
   calculated using an exponentially weighted moving standard deviation with a 60-day window on r_t."
- WHY:
  1. Risk-neutral expected wealth equivalence: As established in Equation (1) & (2),
     under linear utility, maximizing expected cumulative rewards is equivalent to maximizing wealth.
  2. Volatility scaling: Scales position size inversely with market volatility,
     ensuring that the agent's rewards are driven by predictive alpha rather than varying volatility regimes.
  3. Proportional transaction costs: Explicitly penalizes portfolio turnover using basis points (bp)
     and traded value p_{t-1}, training the agent to maintain convictions and avoid whipsawing in noisy markets.
  4. Combinatorial Action Space support: Accepts both DiscreteActionSpace ({-1, 0, 1}) and
     ContinuousActionSpace ([-1, 1]) seamlessly via the unified BaseActionSpace abstraction.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

from trading_drl.config import RewardConfig
from trading_drl.action_spaces import BaseActionSpace, DiscreteActionSpace, ContinuousActionSpace, ActionSpaceType


class TradingEnvironment:
    """
    Simulated Financial Market Environment executing Equation (4) dynamics.
    
    Conforms to standard RL step / reset semantics.
    """

    def __init__(
        self,
        states: np.ndarray,
        prices: np.ndarray,
        daily_diff: np.ndarray,
        ewma_vols: np.ndarray,
        action_space: BaseActionSpace,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Args:
            states: Sequential feature array of shape (N, 60, num_features).
            prices: Close price array of shape (N,).
            daily_diff: Price change array r_t = p_t - p_{t-1} of shape (N,).
            ewma_vols: Ex-ante 60-day EWMA volatility array sigma_t of shape (N,).
            action_space: Modular BaseActionSpace (Discrete or Continuous).
            reward_config: RewardConfig instance configuring sigma_tgt, bp, mu.
        """
        assert len(states) == len(prices) == len(daily_diff) == len(ewma_vols), (
            f"Dimension mismatch: states={len(states)}, prices={len(prices)}, "
            f"daily_diff={len(daily_diff)}, ewma_vols={len(ewma_vols)}"
        )

        self.states = states
        self.prices = prices
        self.daily_diff = daily_diff
        self.ewma_vols = ewma_vols
        self.action_space = action_space
        self.reward_cfg = reward_config or RewardConfig()

        self.num_steps = len(states)
        self.current_step = 0

        # State tracking for Equation (4):
        # A_{t-1} (current position), A_{t-2} (previous position)
        self.prev_action = 0.0       # A_{t-1}
        self.prev_prev_action = 0.0  # A_{t-2}
        
        # Volatility-scaled positions: (sigma_tgt / sigma) * A
        self.prev_scaled_pos = 0.0
        self.prev_prev_scaled_pos = 0.0

        # Performance bookkeeping
        self.cumulative_reward = 0.0
        self.total_turnover = 0.0
        self.total_costs = 0.0
        self.history: Dict[str, list] = {
            "rewards": [],
            "positions": [],
            "raw_returns": [],
            "costs": [],
            "scaled_positions": [],
            "turnovers": [],
        }

    def reset(self, start_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to starting state.
        
        Args:
            start_idx: Step index to begin episode (default 0).
            
        Returns:
            Initial state S_0 and info dict.
        """
        self.current_step = start_idx
        self.prev_action = 0.0
        self.prev_prev_action = 0.0
        self.prev_scaled_pos = 0.0
        self.prev_prev_scaled_pos = 0.0

        self.cumulative_reward = 0.0
        self.total_turnover = 0.0
        self.total_costs = 0.0

        self.history = {
            "rewards": [],
            "positions": [],
            "raw_returns": [],
            "costs": [],
            "scaled_positions": [],
            "turnovers": [],
        }

        info = {
            "step": self.current_step,
            "position": self.prev_action,
        }
        return self.states[self.current_step], info

    def step(self, raw_action: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advances the environment by one trading step applying Equation (4).
        
        WHERE:
        - Section: "Methodology -> Reward function", Equation (4) (Page 5):
          R_t = mu * [ A_{t-1} * (sigma_tgt / sigma_{t-1}) * r_t
                      - bp * p_{t-1} * | (sigma_tgt / sigma_{t-1}) * A_{t-1} - (sigma_tgt / sigma_{t-2}) * A_{t-2} | ]
        
        WHY:
        - In target order mechanics, the action chosen at time t-1 (A_{t-1}) dictates exposure
          to the price movement realized at time t (r_t = p_t - p_{t-1}).
          Meanwhile, transitioning from A_{t-2} to A_{t-1} incurs a transaction cost
          proportional to the basis point cost bp and the traded asset value p_{t-1}.
        """
        if self.current_step >= self.num_steps - 1:
            # Reached end of time series
            truncated = True
            terminated = True
            reward = 0.0
            return self.states[self.current_step], reward, terminated, truncated, {"end_of_data": True}

        # Convert raw action (discrete index or continuous float) to target position A_t in [-1, 1]
        target_pos = self.action_space.to_position(raw_action)

        # Retrieve market data at time t:
        # p_{t-1} is price from preceding step, r_t is daily price difference, sigma_{t-1} is ex-ante volatility
        p_prev = float(self.prices[self.current_step])
        r_t = float(self.daily_diff[self.current_step + 1])
        sigma_prev = float(self.ewma_vols[self.current_step])
        sigma_prev = max(sigma_prev, 1e-6)

        sigma_tgt = self.reward_cfg.daily_target_volatility
        vol_scalar = sigma_tgt / sigma_prev

        # Scaled position at t-1: (sigma_tgt / sigma_{t-1}) * A_{t-1}
        scaled_pos_t_minus_1 = vol_scalar * target_pos
        scaled_pos_t_minus_2 = self.prev_scaled_pos

        # Absolute turnover between t-2 and t-1
        # | (sigma_tgt / sigma_{t-1}) * A_{t-1} - (sigma_tgt / sigma_{t-2}) * A_{t-2} |
        turnover = abs(scaled_pos_t_minus_1 - scaled_pos_t_minus_2)

        # 1. Gross trading return: A_{t-1} * (sigma_tgt / sigma_{t-1}) * r_t
        gross_return = scaled_pos_t_minus_1 * r_t

        # 2. Transaction cost penalty: bp * p_{t-1} * turnover
        # WHERE: Page 5: "bp is the cost rate and 1 bp = 0.0001... our transaction cost term
        # also includes a price term p_{t-1}."
        bp = self.reward_cfg.basis_point_cost
        cost = bp * p_prev * turnover

        # 3. Net reward R_t (Equation 4)
        mu = self.reward_cfg.mu
        reward = mu * (gross_return - cost)

        # Update historical state variables
        self.prev_prev_action = self.prev_action
        self.prev_action = target_pos
        self.prev_prev_scaled_pos = self.prev_scaled_pos
        self.prev_scaled_pos = scaled_pos_t_minus_1

        self.cumulative_reward += reward
        self.total_turnover += turnover
        self.total_costs += cost

        # Log history
        self.history["rewards"].append(reward)
        self.history["positions"].append(target_pos)
        self.history["raw_returns"].append(gross_return)
        self.history["costs"].append(cost)
        self.history["scaled_positions"].append(scaled_pos_t_minus_1)
        self.history["turnovers"].append(turnover)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1
        truncated = False

        next_state = self.states[self.current_step]

        info = {
            "step": self.current_step,
            "target_position": target_pos,
            "scaled_position": scaled_pos_t_minus_1,
            "gross_return": gross_return,
            "cost": cost,
            "turnover": turnover,
            "cumulative_reward": self.cumulative_reward,
        }

        return next_state, float(reward), terminated, truncated, info

    def get_summary_metrics(self) -> Dict[str, float]:
        """Calculates cumulative trajectory metrics from episode history."""
        rewards = np.array(self.history["rewards"], dtype=np.float64)
        if len(rewards) == 0:
            return {}

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards) + 1e-8)
        annualized_return = mean_reward * 252.0
        annualized_vol = std_reward * np.sqrt(252.0)
        sharpe = annualized_return / annualized_vol

        return {
            "cumulative_reward": self.cumulative_reward,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe,
            "total_turnover": self.total_turnover,
            "total_costs": self.total_costs,
            "mean_turnover": float(np.mean(self.history["turnovers"])),
        }
