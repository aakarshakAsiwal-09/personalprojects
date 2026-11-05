"""
Training Engine, Overfitting Controls, and Rolling Walk-Forward Splitting.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE:
  1. Section "Procedures for Controlling Overfitting" (Page 7):
     "we use 10% of any given training data as a separate cross-validation set to optimize
      hyperparameters. We monitor the validation performance and use early stopping with 20 epochs
      to help optimal model selection. Any hyperparameter optimization is done on the cross-validation
      set, leaving the test set for the final test performance evaluation. This reduces the leakage
      of the test data, another cause of overfitting... expanding the dataset to allow training on
      a cross section of futures contracts, not just on a single contract, greatly helps to reduce
      the problem in our case."
  2. Section "Description of Dataset" (Page 6):
     "We retrain our model at every five years, using all data available up to that point to optimize
      the parameters. Model parameters are then fixed for the next five years to produce out-of-sample results."
- WHY:
  Financial markets suffer from non-stationarity, regime shifts, and lookahead biases.
  A strict 10% cross-validation set with 20-epoch early stopping prevents parameter memorization.
  Walk-forward out-of-sample evaluation guarantees that reported metrics reflect realistic live execution.
  This module handles all 6 algorithm-action permutations seamlessly.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import copy
import os
import numpy as np
import pandas as pd
import torch

from trading_drl.config import RLTrainingConfig, RewardConfig
from trading_drl.action_spaces import BaseActionSpace, ActionSpaceType
from trading_drl.environment import TradingEnvironment
from trading_drl.models.dqn import DQNAgent
from trading_drl.models.pg import PolicyGradientsAgent
from trading_drl.models.a2c import A2CAgent


class TimeSeriesSplitter:
    """
    Splits time series into Train, Validation, and Out-Of-Sample Test partitions
    with statistical sample-size safeguards.
    
    WHERE:
    - Section: "Procedures for Controlling Overfitting" (Page 7).
    
    WHY (Bug 4 Fix):
    - Enforces a statistically meaningful validation set (default 20% / minimum 63 trading days)
      to prevent validation metrics from degenerating into small-sample noise.
    """

    def __init__(
        self,
        validation_ratio: float = 0.20,
        test_ratio: float = 0.25,
        min_val_steps: int = 63,
    ):
        self.val_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.min_val_steps = min_val_steps

    def split_arrays(
        self,
        states: np.ndarray,
        prices: np.ndarray,
        diffs: np.ndarray,
        vols: np.ndarray,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        n = len(states)
        n_test = max(int(n * self.test_ratio), 20)
        n_train_val = n - n_test
        n_val = max(int(n_train_val * self.val_ratio), min(self.min_val_steps, n_train_val // 3))
        n_train = n_train_val - n_val

        # Train: [0 : n_train]
        train_data = (
            states[:n_train],
            prices[:n_train],
            diffs[:n_train],
            vols[:n_train],
        )

        # Validation: [n_train : n_train + n_val]
        val_data = (
            states[n_train:n_train_val],
            prices[n_train:n_train_val],
            diffs[n_train:n_train_val],
            vols[n_train:n_train_val],
        )

        # Test: [n_train_val : n]
        test_data = (
            states[n_train_val:],
            prices[n_train_val:],
            diffs[n_train_val:],
            vols[n_train_val:],
        )

        return train_data, val_data, test_data


class WalkForwardSplitter:
    """
    Generates rolling and expanding walk-forward train/val/test folds
    matching Zhang et al. (2020) Section 'Description of Dataset' (Page 6).
    
    WHERE:
    - Section: "Description of Dataset" (Page 6):
      "We retrain our model at every five years, using all data available up to that point
       to optimize the parameters. Model parameters are then fixed for the next five years
       to produce out-of-sample results."
    
    WHY (Bug 5 Fix):
    - Operationalizes the 5-year rolling / expanding retraining scheme defined in the paper.
    - Yields multi-fold out-of-sample test slices that are concatenated to measure authentic,
      leakage-free walk-forward alpha.
    """

    def __init__(
        self,
        train_window_years: int = 5,
        test_window_years: int = 5,
        validation_ratio: float = 0.20,
        expanding: bool = True,
        steps_per_year: int = 252,
        min_val_steps: int = 63,
    ):
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.val_ratio = validation_ratio
        self.expanding = expanding
        self.steps_per_year = steps_per_year
        self.min_val_steps = min_val_steps

    def get_folds(
        self,
        states: np.ndarray,
        prices: np.ndarray,
        diffs: np.ndarray,
        vols: np.ndarray,
    ) -> List[Dict[str, Any]]:
        n = len(states)
        train_len = self.train_window_years * self.steps_per_year
        test_len = self.test_window_years * self.steps_per_year

        # If data is shorter than full 10-year block (train + test), adapt proportionally
        if n < train_len + test_len:
            test_len = max(int(n * 0.25), self.steps_per_year // 2)
            train_len = n - test_len

        folds = []
        current_test_start = train_len

        while current_test_start < n:
            current_test_end = min(current_test_start + test_len, n)
            if current_test_end - current_test_start < max(20, self.steps_per_year // 4):
                break

            if self.expanding:
                fold_train_start = 0
            else:
                fold_train_start = max(0, current_test_start - train_len)

            fold_train_val_end = current_test_start
            total_train_val = fold_train_val_end - fold_train_start

            val_steps = max(int(total_train_val * self.val_ratio), self.min_val_steps)
            val_steps = min(val_steps, int(total_train_val * 0.35))
            fold_train_end = fold_train_val_end - val_steps

            train_slice = (
                states[fold_train_start:fold_train_end],
                prices[fold_train_start:fold_train_end],
                diffs[fold_train_start:fold_train_end],
                vols[fold_train_start:fold_train_end],
            )
            val_slice = (
                states[fold_train_end:fold_train_val_end],
                prices[fold_train_end:fold_train_val_end],
                diffs[fold_train_end:fold_train_val_end],
                vols[fold_train_end:fold_train_val_end],
            )
            test_slice = (
                states[current_test_start:current_test_end],
                prices[current_test_start:current_test_end],
                diffs[current_test_start:current_test_end],
                vols[current_test_start:current_test_end],
            )

            folds.append({
                "fold_idx": len(folds),
                "train": train_slice,
                "val": val_slice,
                "test": test_slice,
                "train_indices": (fold_train_start, fold_train_end),
                "val_indices": (fold_train_end, fold_train_val_end),
                "test_indices": (current_test_start, current_test_end),
            })

            current_test_start += test_len

        if not folds:
            splitter = TimeSeriesSplitter(validation_ratio=self.val_ratio, min_val_steps=self.min_val_steps)
            tr, va, te = splitter.split_arrays(states, prices, diffs, vols)
            folds.append({
                "fold_idx": 0,
                "train": tr,
                "val": va,
                "test": te,
                "train_indices": (0, len(tr[0])),
                "val_indices": (len(tr[0]), len(tr[0]) + len(va[0])),
                "test_indices": (len(tr[0]) + len(va[0]), n),
            })

        return folds


class ModelTrainer:
    """
    Unified Trainer implementing 10% cross-validation and 20-epoch early stopping.
    
    WHERE:
    - Section: "Procedures for Controlling Overfitting" (Page 7).
    - Exhibit 1: Values of Hyperparameters for Different RL Algorithms (Page 7).
    """

    def __init__(
        self,
        agent: Union[DQNAgent, PolicyGradientsAgent, A2CAgent],
        train_env: TradingEnvironment,
        val_env: TradingEnvironment,
        config: Optional[RLTrainingConfig] = None,
        save_dir: str = "checkpoints",
    ):
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.cfg = config or RLTrainingConfig()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_metric = -float("inf")
        self.best_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
        self.training_history: List[Dict[str, float]] = []

    @staticmethod
    def compute_validation_score(val_metrics: Dict[str, float]) -> float:
        """
        Computes a regularized, sample-stable validation score.
        
        WHERE:
        - Section: "Procedures for Controlling Overfitting" (Page 7).
        
        WHY (Bug 4 Fix):
        - Raw Sharpe ratio over short validation windows has high standard error and blows up
          spuriously if standard deviation happens to be tiny.
        - Clamping extreme ratios and incorporating directional return stability prevents model
          selection from overfitting to validation noise.
        """
        val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
        ann_return = val_metrics.get("annualized_return", 0.0)
        cum_reward = val_metrics.get("cumulative_reward", 0.0)

        if np.isnan(val_sharpe) or np.isinf(val_sharpe):
            return cum_reward

        clamped_sharpe = float(np.clip(val_sharpe, -4.0, 4.0))
        score = clamped_sharpe + 0.05 * np.sign(ann_return) * np.log1p(abs(ann_return))
        return score

    def evaluate_validation(self) -> Dict[str, float]:
        """
        Runs deterministic validation rollout to evaluate out-of-sample skill.
        """
        state, _ = self.val_env.reset()
        done = False

        while not done:
            if isinstance(self.agent, DQNAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            elif isinstance(self.agent, PolicyGradientsAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            elif isinstance(self.agent, A2CAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            else:
                position = 0.0

            next_state, _, terminated, truncated, _ = self.val_env.step(position)
            done = terminated or truncated
            state = next_state

        metrics = self.val_env.get_summary_metrics()
        return metrics

    def train_epoch(self) -> float:
        """
        Runs one full training pass across the training environment.
        
        WHY (Bug 3 Fix):
        - When an episode completes (done=True), PG agent immediately runs `self.agent.update()`
          for that trajectory and resets episodic buffers. Trajectories are never conflated
          or discounted across episode boundaries.
        """
        state, _ = self.train_env.reset()
        done = False
        total_epoch_reward = 0.0

        step_idx = 0
        while not done:
            if isinstance(self.agent, DQNAgent):
                position, raw_action_idx = self.agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, _ = self.train_env.step(position)
                done = terminated or truncated
                self.agent.memory.push(state, raw_action_idx, reward, next_state, done)
                if step_idx % getattr(self.cfg, "dqn_update_freq", 4) == 0:
                    self.agent.update()
                step_idx += 1

            elif isinstance(self.agent, PolicyGradientsAgent):
                position, _ = self.agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, _ = self.train_env.step(position)
                done = terminated or truncated
                self.agent.record_reward(reward)
                # Episode-boundary update for REINFORCE (Bug 3 Fix)
                if done:
                    self.agent.update()

            elif isinstance(self.agent, A2CAgent):
                position, step_data = self.agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, _ = self.train_env.step(position)
                done = terminated or truncated
                self.agent.record_transition(state, position, reward, next_state, done, step_data)
                self.agent.update(force=done)

            total_epoch_reward += reward
            state = next_state

        # Safety flush if any trajectory transitions were recorded before termination
        if isinstance(self.agent, PolicyGradientsAgent) and len(self.agent.rewards) > 0:
            self.agent.update()
        elif isinstance(self.agent, A2CAgent) and len(self.agent.buffer.rewards) > 0:
            self.agent.update(force=True)

        return total_epoch_reward

    def train(self, max_epochs: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Executes training loop with 20-epoch early stopping.
        
        WHERE:
        - Page 7: "We monitor the validation performance and use early stopping with 20 epochs
          to help optimal model selection."
        
        WHY:
        - Terminates training when generalization to the validation set ceases, preventing overfitting.
        """
        epochs = max_epochs or self.cfg.max_epochs
        patience = self.cfg.early_stopping_patience  # 20 epochs
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_reward = self.train_epoch()
            val_metrics = self.evaluate_validation()
            val_sharpe = val_metrics.get("sharpe_ratio", -999.0)
            val_reward = val_metrics.get("cumulative_reward", -999.0)

            # Robust optimization criterion (Bug 4 Fix)
            val_criterion = self.compute_validation_score(val_metrics)

            record = {
                "epoch": epoch,
                "train_reward": train_reward,
                "val_reward": val_reward,
                "val_sharpe": val_sharpe,
                "val_score": val_criterion,
            }
            self.training_history.append(record)
            print(f"      Epoch {epoch:2d}/{epochs:2d} | Train R: {train_reward:.3f} | Val Score: {val_criterion:.3f} | Val Sharpe: {val_sharpe:.3f}", flush=True)

            if val_criterion > self.best_val_metric:
                self.best_val_metric = val_criterion
                patience_counter = 0
                self.agent.save(self.best_checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # Early stopping triggered
                break

        # Reload best model parameters for final evaluation
        if os.path.exists(self.best_checkpoint_path):
            self.agent.load(self.best_checkpoint_path)

        return self.training_history

    def backtest(self, test_env: TradingEnvironment) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluates the trained model on out-of-sample test environment.
        
        Returns:
            returns: Daily net returns R_t.
            turnovers: Daily turnovers.
            metrics: Full performance dictionary.
        """
        state, _ = test_env.reset()
        done = False

        while not done:
            if isinstance(self.agent, DQNAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            elif isinstance(self.agent, PolicyGradientsAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            elif isinstance(self.agent, A2CAgent):
                position, _ = self.agent.select_action(state, evaluate=True)
            else:
                position = 0.0

            next_state, _, terminated, truncated, _ = test_env.step(position)
            done = terminated or truncated
            state = next_state

        returns = np.array(test_env.history["rewards"], dtype=np.float64)
        turnovers = np.array(test_env.history["turnovers"], dtype=np.float64)
        metrics = test_env.get_summary_metrics()

        return returns, turnovers, metrics
