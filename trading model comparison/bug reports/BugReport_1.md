# Python Trading-RL Framework: Bug Audit & Code Review

## Context

A Python trading-RL framework (`trading_drl/`) built on Zhang, Zohren & Roberts (2020). It implements Dueling Double DQN, REINFORCE (PG), and synchronous A2C, each paired with a discrete $\{-1, 0, 1\}$ or continuous $[-1, 1]$ action space (6 permutations). The codebase runs fine syntactically, but the bugs outlined below are logical/scientific and render every reported result invalid.

---

## Identified Bugs

### Bug 1 — Action-space collision (`action_spaces.py` line 203)

* **Description:** `ContinuousActionSpace.to_raw_action()` has a dtype guard that misroutes integer inputs into the discrete branch, silently returning wrong bin indices instead of raising.
* **Impact:** When the environment feeds a position back as an integer (e.g., `0`), it maps `0` $\rightarrow$ bin 0 (position $-1.0$) instead of the intended bin 10 (position $0.0$). All “DQN Continuous” results are corrupted by this.

### Bug 2 — Dropout active at evaluation time (`models/dqn.py` lines 128, 160, 196; no equivalent in `pg.py` / `a2c.py`)

* **Description:** Only DQN calls `.eval()` before inference. PG and A2C networks never call `.eval()`, so dropout ($p=0.2$) inside the shared LSTM backbone stays active during test-time rollouts.
* **Impact:** Adds stochastic noise to every A2C/PG position, inflating variance and making test Sharpe ratios non-reproducible.

### Bug 3 — PG updates once per epoch, not per episode (`train.py` lines 193–196)

* **Description:** The training loop calls `pg_agent.update()` exactly once at the end of each epoch, regardless of how many episodes completed.
* **Impact:** The REINFORCE trajectory buffer accumulates rewards across multiple episodes and discounts them as if they were one long episode—a fundamental error in return attribution. The gradient signal is incorrect.

### Bug 4 — Model selection on a 10% validation window (`train.py` lines 221–239; `config.py` line 160)

* **Description:** `TimeSeriesSplitter` uses `validation_ratio=0.10`. On 252 trading days, that is roughly 25 days.
* **Impact:** The best checkpoint is chosen by validation Sharpe over 25 days ($\text{SE} \approx 0.4$), so selection is pure noise. Shipped checkpoints are not meaningfully “best”.

### Bug 5 — Walk-forward window wired but never executed (`config.py` lines 159–160; `run_matrix_experiment.py` line 160)

* **Description:** `train_window_years` and `test_window_years` exist in the config but are never read by the splitter.
* **Impact:** The README claims a 5-year rolling walk-forward (matching the paper), but the code performs a single static 60/10/30 split, resulting in zero out-of-sample robustness.

### Bug 6 — Unseeded runs (`data_pipeline.py` line 305 only)

* **Description:** The only `np.random.seed` call is inside synthetic-data generation. PyTorch is never seeded; Python's `random` module (used by DQN’s replay sampler) is never seeded; the A2C/PG Gaussian samplers are never seeded.
* **Impact:** Rerunning the experiment produces different numbers each time. The shipped `results/*.csv` files cannot be reproduced.

### Bug 7 — Vol-targeting rescale uses full-sample std (`evaluate.py` lines 72–75)

* **Description:** `PortfolioEvaluator.calculate_metrics()` rescales returns by $\text{target\_vol} / (\text{std}(\text{returns}) \times \sqrt{252})$ where std is computed over the entire test set before rescaling.
* **Impact:** This introduces a mild look-ahead bias (the scaler sees future returns). More importantly, it means the vol-targeted Sharpe simplifies to $\text{target\_vol} \times \sqrt{252} / \text{mean}(\text{returns})$, losing all information about the raw scale.

### Bug 8 — Cost-sweep shortcut introduces approximation error (`run_matrix_experiment.py` line 416)

* **Description:** Instead of re-running the environment at each basis-point (bp) level, the sweep adjusts returns post-hoc: $\text{adj\_rets} = \text{orig\_rets} - (\Delta\text{bp} \times \text{mean\_vol} \times \text{turnovers})$.
* **Impact:** Using the mean vol scalar rather than per-step vol understates costs on high-vol days and overstates them on low-vol days. Maximum error per reported cell is $\sim 0.014$ Sharpe units—small, but unprincipled for a cost-sensitivity study.

### Bug 9 — Three of six checkpoint timestamps predate the experiment run

* **Description:** The matrix experiment ran at 15:10 (timestamps on `results/*.csv` and PNGs). Checkpoints for DQN Discrete (17:17), DQN Continuous (17:45), and PG Discrete (18:03) postdate the run.
* **Impact:** They could not have generated those results. A2C Discrete, A2C Continuous, and PG Continuous are stamped 15:08–15:09 and may be genuine. The reported numbers for the three late checkpoints have no traceable source.

### Bug 10 — Single synthetic path, no replication (`run_matrix_experiment.py` line 128)

* **Description:** All training and test data come from one GBM+sinusoidal-drift path seeded at 42. There is no bootstrap, no Monte Carlo over paths, and no real asset data.
* **Impact:** Every Sharpe ratio has a standard error of $\sim 0.3–0.5$ ($T \approx 75$ days), and the entire comparison between discrete and continuous action spaces is based on a single draw. No claim about action-space superiority survives this.

### Bug 11 — Complete absence of real-market data integration (`data_pipeline.py` lines 1–150)

* **Description:** The data pipeline completely lacks integration with live or historical market data sources (such as `yfinance` or free financial APIs). It relies entirely on static, hardcoded synthetic data for both training and testing.
* **Impact:** Because agents never encounter actual market microstructure, non-stationary volatility clustering, fat-tailed return distributions, or structural breaks, all reported performance metrics are artifacts of overfitting to a toy environment rather than evidence of algorithmic trading efficacy.

---

## What is Correct & Can Be Kept

* **Reward Implementation:** Equation 4 reward (vol-scaled, bp-cost) is implemented exactly.
* **Baselines:** The three baselines (Long-Only, $\text{Sign}(R)$, MACD) reproduce the paper accurately.
* **Environment Logic:** There is no look-ahead bias in the environment step method.
* **Architecture & Hyperparameters:** The LSTM backbone architecture (`[64, 32, LeakyReLU, dropout 0.2]`) and all hyperparameters from Exhibit 1 ($\gamma = 0.3, \text{bp} = 0.0020, \text{batch} = 64/128, \tau = 1000, \text{memory} = 5000$) match the paper correctly.