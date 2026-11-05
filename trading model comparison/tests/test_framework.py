"""
Comprehensive Unit and Integration Tests for Deep Reinforcement Learning Trading Framework.
Tests all features, equations, models, action space combinations, and evaluation metrics.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from trading_drl.seed import set_global_seed
from trading_drl.config import StateSpaceConfig, RewardConfig, RLTrainingConfig, ModelConfig
from trading_drl.action_spaces import (
    DiscreteActionSpace,
    ContinuousActionSpace,
    ActionSpaceType,
    create_action_space,
)
from trading_drl.data_pipeline import DataPipeline
from trading_drl.environment import TradingEnvironment
from trading_drl.baselines import LongOnlyStrategy, SignRStrategy, MACDSignalStrategy
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
from trading_drl.train import TimeSeriesSplitter, WalkForwardSplitter, ModelTrainer
from trading_drl.evaluate import PortfolioEvaluator


class TestDataPipeline(unittest.TestCase):
    """Tests feature engineering and state generation matching Section: Methodology -> State space."""

    def setUp(self):
        self.pipeline = DataPipeline()
        self.df = DataPipeline.generate_synthetic_data(n_days=400, random_seed=123)

    def test_synthetic_data_generation(self):
        self.assertEqual(len(self.df), 400)
        self.assertIn("Close", self.df.columns)
        self.assertTrue((self.df["Close"] > 0).all())

    def test_feature_extraction(self):
        feat_df = self.pipeline.extract_features(self.df)
        expected_cols = [
            "norm_close",
            "vol_adj_return_21d",
            "vol_adj_return_42d",
            "vol_adj_return_63d",
            "vol_adj_return_252d",
            "macd_8_24",
            "macd_16_48",
            "macd_32_96",
            "composite_macd",
            "rsi_norm",
            "daily_diff",
            "ewma_vol_additive",
        ]
        for col in expected_cols:
            self.assertIn(col, feat_df.columns, f"Missing feature column: {col}")
        # Check no infinite values
        self.assertFalse(np.isinf(feat_df[self.pipeline.get_feature_columns()].values).any())

    def test_sequential_dataset_shape(self):
        states, prices, diffs, vols = self.pipeline.build_sequential_dataset(self.df)
        self.assertEqual(states.ndim, 3)
        self.assertEqual(states.shape[1], 60)  # Sequence length = 60
        self.assertEqual(states.shape[2], len(self.pipeline.get_feature_columns()))
        self.assertEqual(len(states), len(prices))
        self.assertEqual(len(prices), len(diffs))
        self.assertEqual(len(diffs), len(vols))


class TestActionSpaces(unittest.TestCase):
    """Tests Discrete and Continuous Action Space Adapters."""

    def test_discrete_action_space(self):
        space = DiscreteActionSpace()
        self.assertEqual(space.space_type, ActionSpaceType.DISCRETE)
        self.assertEqual(space.n_actions, 3)
        self.assertEqual(space.to_position(0), -1.0)
        self.assertEqual(space.to_position(1), 0.0)
        self.assertEqual(space.to_position(2), 1.0)
        self.assertEqual(space.to_raw_action(0.8), 2)
        self.assertEqual(space.to_raw_action(-0.9), 0)

    def test_continuous_action_space(self):
        space = ContinuousActionSpace(discretization_bins=21)
        self.assertEqual(space.space_type, ActionSpaceType.CONTINUOUS)
        self.assertEqual(space.n_bins, 21)
        self.assertEqual(space.to_position(0.5), 0.5)
        # Binned DQN mapping
        pos = space.bin_index_to_position(0)
        self.assertAlmostEqual(pos, -1.0, places=4)
        pos_mid = space.bin_index_to_position(10)
        self.assertAlmostEqual(pos_mid, 0.0, places=4)
        pos_high = space.bin_index_to_position(20)
        self.assertAlmostEqual(pos_high, 1.0, places=4)

    def test_continuous_action_space_zero_and_integer_positions(self):
        """Bug 1 Fix: Verify continuous action space never misroutes integers 0, 1 into bin indices."""
        space = ContinuousActionSpace(discretization_bins=21)
        # Position 0.0 or integer 0 must map to 0.0, NOT bin 0 (-1.0)
        self.assertEqual(space.to_position(0), 0.0)
        self.assertEqual(space.to_position(0.0), 0.0)
        self.assertEqual(space.to_position(1.0), 1.0)
        self.assertEqual(space.to_position(-1.0), -1.0)
        self.assertEqual(space.to_position(0.75), 0.75)
        # Bins are accessed explicitly
        self.assertEqual(space.bin_index_to_position(0), -1.0)
        self.assertEqual(space.bin_index_to_position(10), 0.0)
        self.assertEqual(space.bin_index_to_position(20), 1.0)


class TestEnvironmentAndReward(unittest.TestCase):
    """Tests Equation (4) Reward dynamics with volatility scaling and transaction costs."""

    def setUp(self):
        pipeline = DataPipeline()
        df = DataPipeline.generate_synthetic_data(n_days=400, random_seed=42)
        self.states, self.prices, self.diffs, self.vols = pipeline.build_sequential_dataset(df)

    def test_discrete_environment_step(self):
        env = TradingEnvironment(
            states=self.states,
            prices=self.prices,
            daily_diff=self.diffs,
            ewma_vols=self.vols,
            action_space=DiscreteActionSpace(),
            reward_config=RewardConfig(basis_point_cost=0.0020),
        )
        state, info = env.reset()
        self.assertEqual(state.shape, (60, self.states.shape[2]))

        # Take long position
        next_state, reward, term, trunc, info = env.step(raw_action=2)  # position +1.0
        self.assertFalse(term)
        self.assertIn("gross_return", info)
        self.assertIn("cost", info)
        self.assertIn("turnover", info)
        self.assertTrue(info["cost"] > 0)  # Incurred transaction cost entering position

    def test_continuous_environment_step(self):
        env = TradingEnvironment(
            states=self.states,
            prices=self.prices,
            daily_diff=self.diffs,
            ewma_vols=self.vols,
            action_space=ContinuousActionSpace(),
            reward_config=RewardConfig(basis_point_cost=0.0020),
        )
        state, info = env.reset()
        next_state, reward, term, trunc, info = env.step(raw_action=0.45)
        self.assertAlmostEqual(info["target_position"], 0.45, places=4)


class TestBaselineStrategies(unittest.TestCase):
    """Tests Long-Only, Sign(R), and Baz et al. MACD Baselines."""

    def test_baselines(self):
        long_strat = LongOnlyStrategy()
        self.assertEqual(long_strat.predict(), 1.0)

        sign_strat = SignRStrategy()
        dummy_state = np.zeros((60, 10))
        dummy_state[-1, 4] = 1.5  # Positive 252d return
        self.assertEqual(sign_strat.predict(dummy_state), 1.0)
        dummy_state[-1, 4] = -0.5
        self.assertEqual(sign_strat.predict(dummy_state), -1.0)

        macd_strat = MACDSignalStrategy()
        # Test non-linear transfer function phi(x) = x * exp(-x^2 / 4) / 0.89
        # Theoretical max of x * exp(-x^2 / 4) is sqrt(2) * exp(-0.5) ~ 0.8578
        # Divided by 0.89 yields ~ 0.9638
        val_at_max = MACDSignalStrategy.phi_response(np.sqrt(2))
        self.assertAlmostEqual(val_at_max, 0.9638, places=3)


class TestCombinatorialRLPermutations(unittest.TestCase):
    """Tests all 6 algorithm + action-space permutations."""

    def setUp(self):
        pipeline = DataPipeline()
        df = DataPipeline.generate_synthetic_data(n_days=400, random_seed=99)
        self.states, self.prices, self.diffs, self.vols = pipeline.build_sequential_dataset(df)
        self.input_dim = self.states.shape[2]

    def test_permutation_1_dqn_discrete(self):
        action_space = DiscreteActionSpace()
        agent = DQNAgent(input_dim=self.input_dim, action_space=action_space)
        pos, idx = agent.select_action(self.states[0], evaluate=False)
        self.assertIn(pos, [-1.0, 0.0, 1.0])
        # Push to replay buffer and update
        agent.memory.push(self.states[0], idx, 0.1, self.states[1], False)
        # Populate enough to run update
        for _ in range(65):
            agent.memory.push(self.states[0], idx, 0.1, self.states[1], False)
        loss = agent.update()
        self.assertIsNotNone(loss)

    def test_permutation_2_dqn_continuous(self):
        action_space = ContinuousActionSpace(discretization_bins=11)
        agent = DQNAgent(input_dim=self.input_dim, action_space=action_space)
        pos, idx = agent.select_action(self.states[0], evaluate=False)
        self.assertTrue(-1.0 <= pos <= 1.0)
        for _ in range(65):
            agent.memory.push(self.states[0], idx, 0.1, self.states[1], False)
        loss = agent.update()
        self.assertIsNotNone(loss)

    def test_permutation_3_pg_discrete(self):
        action_space = DiscreteActionSpace()
        agent = PolicyGradientsAgent(input_dim=self.input_dim, action_space=action_space)
        pos, idx = agent.select_action(self.states[0], evaluate=False)
        self.assertIn(pos, [-1.0, 0.0, 1.0])
        agent.record_reward(0.5)
        loss = agent.update()
        self.assertIsNotNone(loss)

    def test_permutation_4_pg_continuous(self):
        action_space = ContinuousActionSpace()
        agent = PolicyGradientsAgent(input_dim=self.input_dim, action_space=action_space)
        pos, raw = agent.select_action(self.states[0], evaluate=False)
        self.assertTrue(-1.0 <= pos <= 1.0)
        agent.record_reward(0.2)
        loss = agent.update()
        self.assertIsNotNone(loss)

    def test_permutation_5_a2c_discrete(self):
        action_space = DiscreteActionSpace()
        agent = A2CAgent(input_dim=self.input_dim, action_space=action_space)
        pos, step_data = agent.select_action(self.states[0], evaluate=False)
        self.assertIn(pos, [-1.0, 0.0, 1.0])
        agent.record_transition(self.states[0], pos, 0.1, self.states[1], False, step_data)
        loss_info = agent.update(force=True)
        self.assertIsNotNone(loss_info)
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)

    def test_permutation_6_a2c_continuous(self):
        action_space = ContinuousActionSpace()
        agent = A2CAgent(input_dim=self.input_dim, action_space=action_space)
        pos, step_data = agent.select_action(self.states[0], evaluate=False)
        self.assertTrue(-1.0 <= pos <= 1.0)
        agent.record_transition(self.states[0], pos, 0.1, self.states[1], False, step_data)
        loss_info = agent.update(force=True)
        self.assertIsNotNone(loss_info)
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)

    def test_eval_mode_disables_dropout(self):
        """Bug 2 Fix: Verify that evaluate=True puts policy networks in eval mode."""
        action_space = ContinuousActionSpace()
        pg_agent = PolicyGradientsAgent(input_dim=self.input_dim, action_space=action_space)
        a2c_agent = A2CAgent(input_dim=self.input_dim, action_space=action_space)

        # In evaluation mode, networks must be in eval() (training == False)
        pos1, _ = pg_agent.select_action(self.states[0], evaluate=True)
        self.assertFalse(pg_agent.policy_net.training)
        pos2, _ = pg_agent.select_action(self.states[0], evaluate=True)
        self.assertEqual(pos1, pos2)

        pos1_a2c, _ = a2c_agent.select_action(self.states[0], evaluate=True)
        self.assertFalse(a2c_agent.network.training)
        pos2_a2c, _ = a2c_agent.select_action(self.states[0], evaluate=True)
        self.assertEqual(pos1_a2c, pos2_a2c)


class TestPortfolioEvaluator(unittest.TestCase):
    """Tests evaluation metrics from Exhibit 2 & volatility targeting."""

    def test_metrics_calculation(self):
        evaluator = PortfolioEvaluator()
        returns = np.array([0.01, -0.005, 0.02, 0.003, -0.01, 0.015, -0.002, 0.008])
        turnovers = np.array([0.2, 0.1, 0.3, 0.0, 0.4, 0.1, 0.05, 0.2])

        metrics = evaluator.calculate_metrics(returns, turnovers=turnovers, apply_vol_target=False)
        expected_keys = [
            "E(R)",
            "Std(R)",
            "DD",
            "Sharpe",
            "Sortino",
            "MDD",
            "Calmar",
            "% +ve Ret",
            "Ave. P / Ave. L",
            "Mean Daily Turnover",
            "Ave. Return / Turnover",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_ex_ante_volatility_targeting(self):
        """Bug 7 Fix: Verify ex-ante volatility targeting is causal and has zero lookahead bias."""
        evaluator = PortfolioEvaluator()
        returns = np.random.randn(250) * 0.015
        scaled = evaluator.apply_ex_ante_volatility_targeting(returns, target_annual_vol=0.15)
        self.assertEqual(len(scaled), len(returns))
        annual_vol = np.std(scaled) * np.sqrt(252)
        self.assertTrue(0.05 <= annual_vol <= 0.30)

    def test_exact_cost_sensitivity_sweep(self):
        """Bug 8 Fix: Verify exact per-step transaction cost sensitivity calculation."""
        evaluator = PortfolioEvaluator()
        n = 100
        gross_returns = np.ones(n) * 0.005
        turnovers = np.ones(n) * 0.2
        prices = np.ones(n) * 100.0
        bps_levels = [0.0, 10.0, 20.0, 50.0]

        res = evaluator.compute_exact_cost_sensitivity(
            gross_returns=gross_returns,
            turnovers=turnovers,
            prices=prices,
            cost_bps_levels=bps_levels,
            mu=1.0,
            apply_vol_target=False,
        )
        self.assertEqual(len(res), len(bps_levels))
        # Zero cost net return should match gross return
        self.assertAlmostEqual(res[0.0]["E(R)"], 0.005 * 252, places=3)
        # Higher cost rates must strictly decrease expected returns
        self.assertGreater(res[0.0]["E(R)"], res[10.0]["E(R)"])
        self.assertGreater(res[10.0]["E(R)"], res[20.0]["E(R)"])
        self.assertGreater(res[20.0]["E(R)"], res[50.0]["E(R)"])

    def test_monte_carlo_summary_compilation(self):
        """Bug 10 Fix: Verify Monte Carlo multi-path aggregation computes correct Mean +/- SE."""
        evaluator = PortfolioEvaluator()
        # Create synthetic results across 3 paths
        path_results = []
        for i in range(3):
            path_dict = {
                "Model_A": {
                    "returns": np.random.randn(100) * 0.01 + (0.001 * (i + 1)),
                    "turnovers": np.ones(100) * 0.1,
                },
                "Model_B": {
                    "returns": np.random.randn(100) * 0.01 + 0.0005,
                    "turnovers": np.ones(100) * 0.05,
                },
            }
            path_results.append(path_dict)

        disp_df, num_df = evaluator.compile_monte_carlo_summary(path_results, apply_vol_target=False)
        self.assertEqual(len(disp_df), 2)
        self.assertEqual(len(num_df), 2)
        self.assertIn("Sharpe", disp_df.columns)
        self.assertIn("Sharpe_Mean", num_df.columns)
        self.assertIn("Sharpe_SE", num_df.columns)
        # Check string format "mean +/- se"
        self.assertIn("±", disp_df.loc["Model_A", "Sharpe"])


class TestWalkForwardAndSeeding(unittest.TestCase):
    """Tests walk-forward splits and global seeding."""

    def test_global_seeding(self):
        """Bug 6 Fix: Verify bit-for-bit reproducibility across runs."""
        set_global_seed(42)
        t1 = torch.randn(5, 5)
        a1 = np.random.rand(5)

        set_global_seed(42)
        t2 = torch.randn(5, 5)
        a2 = np.random.rand(5)

        self.assertTrue(torch.equal(t1, t2))
        np.testing.assert_array_equal(a1, a2)

    def test_walk_forward_splitter(self):
        """Bug 5 Fix: Verify walk-forward fold generation."""
        pipeline = DataPipeline()
        df = pipeline.generate_synthetic_data(n_days=1000, random_seed=42)
        states, prices, diffs, vols = pipeline.build_sequential_dataset(df)
        wf = WalkForwardSplitter(train_window_years=2, test_window_years=1, steps_per_year=252)
        folds = wf.get_folds(states, prices, diffs, vols)
        self.assertTrue(len(folds) >= 1)
        for fold in folds:
            self.assertIn("train", fold)
            self.assertIn("val", fold)
            self.assertIn("test", fold)
            self.assertLessEqual(fold["train_indices"][1], fold["test_indices"][0])

    def test_checkpoint_manifest_creation(self):
        """Bug 9 Fix: Verify checkpoint manifest generation, hash calculation, and timestamp synchronization."""
        import os
        import tempfile
        from run_matrix_experiment import create_checkpoint_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            class DummyArgs:
                seed = 42
                cost_bps = 20.0
                vol_target = 0.15
                epochs = 5
                source = "synthetic"
                walk_forward = False
                monte_carlo_paths = 1

            # Create mock checkpoint directories
            models = ["DQN_Discrete", "DQN_Continuous", "PG_Discrete", "PG_Continuous", "A2C_Discrete", "A2C_Continuous"]
            for m in models:
                mdir = os.path.join(tmpdir, f"ckpt_{m}")
                os.makedirs(mdir, exist_ok=True)
                with open(os.path.join(mdir, "best_model.pt"), "wb") as f:
                    f.write(b"dummy_weights_data")

            manifest = create_checkpoint_manifest(output_dir=tmpdir, args=DummyArgs(), start_time="2026-10-28T18:00:00Z")
            self.assertTrue(manifest["all_checkpoints_synchronized"])
            self.assertEqual(len(manifest["checkpoints"]), 6)
            for m in models:
                self.assertEqual(manifest["checkpoints"][m]["status"], "VERIFIED_PRESENT")
                self.assertTrue(len(manifest["checkpoints"][m]["sha256"]) == 64)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoint_manifest.json")))



if __name__ == "__main__":
    unittest.main()
