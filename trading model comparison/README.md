# Deep Reinforcement Learning for Trading Framework

A production-grade, modular, and extensible Python quantitative trading framework inspired by the landmark research paper:
> **"Deep Reinforcement Learning for Trading"**  
> *Zihao Zhang, Stefan Zohren, and Stephen Roberts*  
> **Journal of Financial Data Science, Spring 2020** (Oxford-Man Institute of Quantitative Finance, University of Oxford).

---

## 1. Architectural Highlights & Experimental Matrix

Unlike the baseline paper—which rigidly mapped specific RL algorithms to fixed action spaces (DQN and PG to discrete $\{-1, 0, 1\}$, and A2C to continuous $[-1, 1]$)—this framework is **fully combinatorial and decoupled**. Any algorithm can be configured to operate on **any action space**:

| Algorithm | Action Space Type | Policy / Q Representation | Target Order Output |
| :--- | :--- | :--- | :--- |
| **DQN** | **Discrete** | Dueling Q-Network ($3$ heads) | $\{-1.0, 0.0, 1.0\}$ |
| **DQN** | **Continuous** | Binned Mesh / Discretized Q-Network ($K=21$ bins) | $[-1.0, 1.0]$ |
| **Policy Gradients (PG)** | **Discrete** | Categorical Logits Distribution | $\{-1.0, 0.0, 1.0\}$ |
| **Policy Gradients (PG)** | **Continuous** | Tanh-Gaussian Normal Distribution | $[-1.0, 1.0]$ |
| **Advantage Actor-Critic (A2C)** | **Discrete** | Categorical Actor + TD Critic $V(s)$ | $\{-1.0, 0.0, 1.0\}$ |
| **Advantage Actor-Critic (A2C)** | **Continuous** | Tanh-Gaussian Actor + TD Critic $V(s)$ | $[-1.0, 1.0]$ |

In addition, the framework provides exact implementations of the paper's three classical benchmarks:
1. **Long-Only**: Unconditional long exposure ($A_t = 1.0$).
2. **Sign(R) (TSMOM)**: 12-Month Time-Series Momentum ($A_t = \text{sign}(r_{t-252:t})$, Eq. 10; Moskowitz et al. 2012).
3. **MACD Signal**: Baz et al. (2015) non-linear transfer function over multi-scale composite MACD ($A_t = \phi(\text{MACD}_t)$, Eq. 11–12).

---

## 2. Research Paper Methodology & Mathematical Mapping

Every major class and function in this repository is thoroughly annotated with explicit **WHERE** and **WHY** commentary citing the research paper:

### 2.1 State Space Representation (Section: Methodology -> State Space, Page 4)
Each state $S_t$ consists of the **past 60 observations** across normalized feature channels:
1. **Normalized Close Prices**: 60-day rolling z-score $(p_t - \mu_{60}) / \sigma_{60}$.
2. **Volatility-Adjusted Returns**: Multi-horizon returns over 1M (21d), 2M (42d), 3M (63d), and 1Y (252d), scaled by daily EWMA volatility $\sigma_t \sqrt{\text{horizon}}$ where $\sigma_t$ has a 60-day span.
3. **Normalized MACD (Baz et al. 2015, Eq. 3 & 12)**:
   $$\text{MACD}_t(S, L) = \frac{q_t}{\text{std}(q_{t-252:t})}, \quad q_t = \frac{m(S) - m(L)}{\text{std}(p_{t-63:t})}$$
   Evaluated across short/long pairs $(S, L) \in \{(8, 24), (16, 48), (32, 96)\}$ and combined into composite MACD.
4. **Relative Strength Index (RSI, Wilder 1978)**: 30-day lookback window, zero-centered to $[-1.0, 1.0]$.

### 2.2 Volatility Scaling & Reward Function with Frictions (Equation 4, Page 5)
$$R_t = \mu \left[ A_{t-1} \frac{\sigma_{\text{tgt}}}{\sigma_{t-1}} r_t - \text{bp} \cdot p_{t-1} \left| \frac{\sigma_{\text{tgt}}}{\sigma_{t-1}} A_{t-1} - \frac{\sigma_{\text{tgt}}}{\sigma_{t-2}} A_{t-2} \right| \right]$$
- $\sigma_{\text{tgt}}$: Daily volatility target ($\sigma_{\text{annual}} / \sqrt{252}$).
- $\sigma_{t-1}$: Ex-ante 60-day EWMA volatility of daily price changes $r_t = p_t - p_{t-1}$.
- $\text{bp}$: Transaction cost rate in basis points ($1\text{ bp} = 0.0001$, default $20\text{ bps} = 0.0020$).
- $p_{t-1}$: Preceding asset price reflecting true traded capital volume.
- $\mu = 1.0$: Fixed contract scaling multiplier.

### 2.3 Neural Network Architectures (Section: Training Schemes for RL, Page 6–7)
- **Backbone**: Two-layer LSTM networks ($64$ units in layer 1, $32$ units in layer 2).
- **Activations**: Leaky-ReLU non-linearities (Maas et al. 2013).
- **Regularization**: Dropout ($p=0.2$) to mitigate overfitting on financial noise.
- **DQN Architecture**:
  - Dueling streams: State value $V(s)$ and Action Advantage $A(s, a)$.
  - Double DQN target evaluation to eliminate maximization bias (Hasselt 2010).
  - Fixed Q-targets with target sync parameter $\tau = 1,000$ steps.
  - Replay memory capacity $5,000$, mini-batch size $64$.
- **Policy Gradients (PG)**:
  - Trajectory sampling with discounted returns-to-go $G_t$ ($\gamma = 0.3$).
  - Policy gradient ascent updating parameters $\theta$ (Eq. 6).
- **Advantage Actor-Critic (A2C)**:
  - Synchronous 1-step temporal difference advantage: $A(S_t, A_t) = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$.
  - Critic TD MSE loss optimization ($\alpha_{\text{critic}} = 0.001$).
  - Actor policy gradient optimization ($\alpha_{\text{actor}} = 0.0001$, batch size $128$).

### 2.4 Overfitting Controls & Out-of-Sample Evaluation (Page 7)
- **10% Cross-Validation Split**: Strict chronological partitioning for validation.
- **Early Stopping**: 20 epochs without validation improvement triggers optimal model rollback.
- **5-Year Rolling Window Evaluation**: Walk-forward training and out-of-sample testing.
- **Portfolio-Level Volatility Targeting (Exhibit 2, Page 8)**: Standardizes all strategies to a uniform annualized volatility (e.g., $15\%$) for fair cross-model comparison.

---

## 3. Directory Structure

```
├── run_matrix_experiment.py     # Master executable running the 6 permutations + 3 baselines
├── trading_drl/
│   ├── __init__.py              # Framework exports and version
│   ├── config.py                # Hyperparameters, StateSpaceConfig, RewardConfig (Exhibit 1)
│   ├── action_spaces.py         # Modular Discrete & Continuous Action Space adapters
│   ├── data_pipeline.py         # Ingestion, Feature extraction (Eq 3), sequential builder
│   ├── environment.py           # MDP Trading Environment & Reward function (Eq 4)
│   ├── baselines.py             # Long-Only, Sign(R) (Eq 10), Baz et al. MACD (Eq 11-12)
│   ├── evaluate.py              # 9 metrics from Exhibit 2, Portfolio Eq 13, Vol Targeting
│   ├── train.py                 # 10% CV, Early Stopping (20 epochs), Chronological Splitter
│   └── models/
│       ├── __init__.py          # Export models
│       ├── networks.py          # 2-layer LSTM (64, 32), LeakyReLU, Dueling, Actor/Critic
│       ├── dqn.py               # Double Dueling DQN with Fixed Targets (Discrete & Cont.)
│       ├── pg.py                # Policy Gradients with Returns-to-Go (Discrete & Cont.)
│       └── a2c.py               # Synchronous A2C with TD Advantage (Discrete & Cont.)
├── tests/
│   └── test_framework.py        # 16 unit tests covering all components and permutations
├── results/                     # Output folder for CSV metrics tables and PNG figures
└── README.md                    # Detailed documentation and research guide
```

---

## 4. Quickstart Guide

### 4.1 Running the Unit Test Suite
To verify all 16 tests covering feature formulas, reward equations, action adapters, and all 6 permutations:
```bash
python -m unittest tests/test_framework.py
```

### 4.2 Running the Full Combinatorial Matrix Experiment
To execute all 6 algorithm-action permutations alongside the 3 classical baselines on synthetic regime-switching price series:
```bash
python run_matrix_experiment.py --epochs 15 --cost-bps 20.0 --save-plots --output-dir results
```

### 4.3 Running on Real Market Data via yfinance
To run the full matrix on real financial contracts (e.g., S&P 500 E-mini futures `ES=F`, Crude Oil `CL=F`, or 10-Year T-Note `ZN=F`):
```bash
python run_matrix_experiment.py --ticker ES=F --epochs 20 --cost-bps 20.0 --output-dir results_sp500
```

---

## 5. Summary Performance Metrics (Exhibit 2 Format)

The evaluator automatically computes the complete suite of performance metrics from Exhibit 2:
- **E(R)**: Annualized Expected Trade Return ($\mu_R \times 252$)
- **Std(R)**: Annualized Standard Deviation ($\sigma_R \times \sqrt{252}$)
- **DD**: Downside Risk / Semi-Deviation of negative returns
- **Sharpe**: Annualized Sharpe Ratio ($E(R) / \text{Std}(R)$)
- **Sortino**: Downside-risk adjusted return ($E(R) / DD$)
- **MDD**: Maximum Peak-to-Trough Drawdown
- **Calmar**: Calmar Ratio ($E(R) / \text{MDD}$)
- **% +ve Ret**: Percentage of trading days with positive return
- **Ave. P / Ave. L**: Win/Loss ratio (average profit divided by average loss)
- **Mean Daily Turnover**: Average daily absolute position change
- **Ave. Return / Turnover**: Trading return generated per unit of transaction turnover (Exhibit 4)
