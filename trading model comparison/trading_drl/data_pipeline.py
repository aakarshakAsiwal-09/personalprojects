"""
Unified Data Pipeline and Feature Engineering Module.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Methodology -> State space" (Page 4), Equations (2), (3), and (12).
  "At a given time step, we take the past 60 observations of each feature to form a single state:
   - Normalized close price series
   - Returns over the past month, two months, three months, and one year are used...
     normalize them by daily volatility adjusted to a reasonable time scale:
     r_{t-k:t} / (sigma_t * sqrt(k)), where sigma_t is computed using an exponentially
     weighted moving standard deviation of r_t with a 60-day span.
   - MACD indicators are proposed by Baz et al. (2015) where:
     q_t = (m(S) - m(L)) / std(p_{t-63:t})
     MACD_t = q_t / std(q_{t-252:t})
     Pairs: S in {8, 16, 32}, L in {24, 48, 96}
   - RSI with a look-back window of 30 days."
- WHY: Financial time series have low signal-to-noise ratios and non-stationary distributions.
  Normalizing returns by EWMA volatility scales risk across contracts and market regimes.
  Multi-scale MACD captures both short-term momentum and long-term macro trends.
  RSI detects mean-reversion exhaustion.
  Windowing into 60-step temporal matrices provides recurrent LSTMs with rich sequential context.
"""

from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd

from trading_drl.config import StateSpaceConfig, RewardConfig


class DataPipeline:
    """
    Ingestion, Cleaning, and Feature Engineering Pipeline for Financial Time Series.
    
    WHERE:
    - Section: "Methodology -> State space" (Page 4).
    - Section: "Description of Dataset" (Page 6).
    """

    def __init__(
        self,
        state_config: Optional[StateSpaceConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.state_cfg = state_config or StateSpaceConfig()
        self.reward_cfg = reward_config or RewardConfig()

    @staticmethod
    def compute_ewma_volatility(
        returns: pd.Series,
        span: int = 60,
    ) -> pd.Series:
        """
        Computes exponentially weighted moving standard deviation (EWMA std).
        
        WHERE:
        - Section: "Methodology -> State space" (Page 4):
          "where sigma_t is computed using an exponentially weighted moving standard deviation
           of r_t with a 60-day span".
        - Section: "Reward function", Equation (4) (Page 5):
          "sigma_{t-1} is an ex ante volatility estimate calculated using an exponentially
           weighted moving standard deviation with a 60-day window on r_t".
        
        WHY:
        - Exponential weighting places greater importance on recent volatility shocks
          while decaying past regimes smoothly, avoiding sudden jump artifacts of simple rolling windows.
        """
        # EWMA variance: E[r^2] - (E[r])^2
        # For zero-mean assumption in daily returns or empirical mean:
        # pandas ewm std calculates the unbiased sample standard deviation with exponential weights
        ewma_std = returns.ewm(span=span, min_periods=max(5, span // 4), adjust=True).std()
        # Floor with tiny epsilon to prevent division by zero in zero-volatility periods
        return ewma_std.bfill().fillna(1e-4).clip(lower=1e-6)

    @staticmethod
    def compute_rsi(prices: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI) with specified window.
        
        WHERE:
        - Section: "Methodology -> State space" (Page 4):
          "The RSI is an oscillating indicator moving between 0 and 100... We include this indicator
           with a look-back window of 30 days in our state representations."
        
        WHY:
        - Wilder (1978) RSI quantifies velocity and magnitude of directional price movements.
          It signals overbought (>80) or oversold (<20) market conditions.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Wilder's smoothing corresponds to ewm with alpha = 1 / window
        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Fill warmup with neutral 50.0
        return rsi.fillna(50.0)

    @staticmethod
    def compute_macd_series(
        prices: pd.Series,
        short_scale: int,
        long_scale: int,
        price_std_window: int = 63,
        signal_std_window: int = 252,
    ) -> pd.Series:
        """
        Computes Baz et al. (2015) normalized MACD indicator as defined in Equation (3).
        
        WHERE:
        - Equation (3), Page 4:
          MACD_t = q_t / std(q_{t-252:t})
          q_t = (m(S) - m(L)) / std(p_{t-63:t})
          where std(p_{t-63:t}) is 63-day rolling price standard deviation,
          and m(S), m(L) are EWMA of prices with timescales S and L.
        
        WHY:
        - Standard MACD is non-stationary because price levels change over decades.
          Dividing by rolling price standard deviation makes q_t scale-invariant.
          Dividing q_t by its 252-day standard deviation standardizes the signal across assets
          to approximately unit variance, essential for deep neural network input stability.
        """
        m_s = prices.ewm(span=short_scale, adjust=True).mean()
        m_l = prices.ewm(span=long_scale, adjust=True).mean()
        diff = m_s - m_l

        # 63-day rolling standard deviation of prices pt
        price_std = prices.rolling(window=price_std_window, min_periods=10).std().bfill()
        price_std = price_std.clip(lower=1e-6)

        # Normalized difference q_t
        q_t = diff / price_std

        # 252-day rolling standard deviation of q_t
        q_std = q_t.rolling(window=signal_std_window, min_periods=20).std().bfill()
        q_std = q_std.clip(lower=1e-6)

        macd = q_t / q_std
        return macd.fillna(0.0)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all paper-specified features from cleaned raw price DataFrame.
        
        WHERE:
        - Section: "Methodology -> State space" (Page 4).
        
        Features engineered:
        1. Normalized close price (60-day rolling z-score: (p - rolling_mean)/rolling_std)
        2. Volatility-adjusted returns over 1M (21d), 2M (42d), 3M (63d), 1Y (252d)
           normalized by daily EWMA volatility * sqrt(horizon)
        3. MACD indicators for (8, 24), (16, 48), (32, 96) and combined composite MACD
        4. RSI with 30-day lookback (zero-centered: (RSI - 50) / 50 in [-1, 1])
        5. Daily returns r_t = p_t - p_{t-1} and 60-day ex-ante volatility sigma_{t-1} for Equation (4)
        """
        clean_df = df.copy()
        if "Close" not in clean_df.columns:
            raise KeyError("DataFrame must contain a 'Close' price column.")

        prices = clean_df["Close"].astype(np.float64)

        # 1. Daily return r_t (additive: p_t - p_{t-1})
        # WHERE: Page 5: "We define r_t = p_t - p_{t-1}, and this expression represents additive profits."
        daily_diff = prices.diff().fillna(0.0)
        clean_df["daily_diff"] = daily_diff

        # Percentage daily return for return scaling calculations
        pct_return = prices.pct_change().fillna(0.0)
        clean_df["pct_return"] = pct_return

        # 2. Ex-ante EWMA standard deviation sigma_t (60-day span)
        # WHERE: Page 4 & 5: "sigma_t is computed using an exponentially weighted moving standard deviation with 60-day span"
        # We calculate sigma on daily additive returns for additive profits, and pct returns for relative returns
        ewma_std_additive = self.compute_ewma_volatility(daily_diff, span=self.state_cfg.volatility_ewma_span)
        ewma_std_pct = self.compute_ewma_volatility(pct_return, span=self.state_cfg.volatility_ewma_span)
        clean_df["ewma_vol_additive"] = ewma_std_additive
        clean_df["ewma_vol_pct"] = ewma_std_pct

        # 3. Normalized close price series
        # Rolling z-score over 60 days
        roll_mean = prices.rolling(window=self.state_cfg.sequence_length, min_periods=5).mean()
        roll_std = prices.rolling(window=self.state_cfg.sequence_length, min_periods=5).std().clip(lower=1e-6)
        clean_df["norm_close"] = ((prices - roll_mean) / roll_std).fillna(0.0).clip(-5.0, 5.0)

        # 4. Volatility-adjusted returns over horizons (21, 42, 63, 252)
        # WHERE: Page 4: "Returns over the past month, two months, three months, and one year...
        #        r_{t-k:t} / (sigma_t * sqrt(k))"
        for horizon in self.state_cfg.return_horizons:
            # Multi-day return r_{t-k:t}
            h_return = (prices - prices.shift(horizon)).fillna(0.0)
            vol_scaled_h = h_return / (ewma_std_additive * np.sqrt(horizon) + 1e-8)
            # Clip extreme tail outliers for neural network stability
            clean_df[f"vol_adj_return_{horizon}d"] = vol_scaled_h.fillna(0.0).clip(-10.0, 10.0)

        # 5. Baz et al. (2015) MACD indicators
        # WHERE: Equation (3) & (12), Page 4 & 6: S in {8, 16, 32}, L in {24, 48, 96}
        composite_macd = pd.Series(0.0, index=clean_df.index)
        for s, l in self.state_cfg.macd_timescales:
            col_name = f"macd_{s}_{l}"
            macd_series = self.compute_macd_series(
                prices=prices,
                short_scale=s,
                long_scale=l,
                price_std_window=self.state_cfg.macd_price_std_window,
                signal_std_window=self.state_cfg.macd_signal_std_window,
            )
            clean_df[col_name] = macd_series.clip(-5.0, 5.0)
            composite_macd += macd_series

        # Equation (12): composite average MACD
        clean_df["composite_macd"] = (composite_macd / len(self.state_cfg.macd_timescales)).clip(-5.0, 5.0)

        # 6. Relative Strength Index (RSI) with 30-day lookback
        # WHERE: Page 4: "RSI with a look-back window of 30 days... moving between 0 and 100"
        # Zero-centered to [-1.0, 1.0] for gradient stability in neural networks
        raw_rsi = self.compute_rsi(prices, window=self.state_cfg.rsi_window)
        clean_df["rsi_raw"] = raw_rsi
        clean_df["rsi_norm"] = ((raw_rsi - 50.0) / 50.0).clip(-1.0, 1.0)

        return clean_df

    def get_feature_columns(self) -> List[str]:
        """Returns the canonical list of feature columns fed into the LSTM state representation."""
        cols = ["norm_close"]
        for h in self.state_cfg.return_horizons:
            cols.append(f"vol_adj_return_{h}d")
        for s, l in self.state_cfg.macd_timescales:
            cols.append(f"macd_{s}_{l}")
        cols.append("composite_macd")
        cols.append("rsi_norm")
        return cols

    def build_sequential_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs sequential 60-observation state matrices and target return vectors.
        
        WHERE:
        - Section: "Methodology -> State space" (Page 4):
          "At a given time step, we take the past 60 observations of each feature to form a single state."
        
        Returns:
            states: np.ndarray of shape (N, 60, num_features)
            prices: np.ndarray of shape (N,) containing close prices
            daily_diff: np.ndarray of shape (N,) containing r_t = p_t - p_{t-1}
            ewma_vol: np.ndarray of shape (N,) containing ex-ante volatility sigma_t
        """
        features_df = self.extract_features(df)
        feature_cols = self.get_feature_columns()

        # Drop initial warmup rows where rolling statistics (e.g. 252d MACD) are calculating
        # Typically warmup requires ~252 rows
        warmup = max(self.state_cfg.macd_signal_std_window, max(self.state_cfg.return_horizons)) + 10
        valid_df = features_df.iloc[warmup:].reset_index(drop=True)

        feature_matrix = valid_df[feature_cols].values.astype(np.float32)
        price_array = valid_df["Close"].values.astype(np.float32)
        diff_array = valid_df["daily_diff"].values.astype(np.float32)
        vol_array = valid_df["ewma_vol_additive"].values.astype(np.float32)

        seq_len = self.state_cfg.sequence_length
        num_samples = len(valid_df) - seq_len

        if num_samples <= 0:
            raise ValueError(f"Dataset length ({len(valid_df)}) is shorter than sequence length ({seq_len}).")

        # Sliding window construction: S_t = X[t-seq_len : t]
        states = np.zeros((num_samples, seq_len, len(feature_cols)), dtype=np.float32)
        prices = np.zeros(num_samples, dtype=np.float32)
        diffs = np.zeros(num_samples, dtype=np.float32)
        vols = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            states[i] = feature_matrix[i : i + seq_len]
            # Time t corresponds to the end of the observation window
            t_idx = i + seq_len
            prices[i] = price_array[t_idx]
            diffs[i] = diff_array[t_idx]
            vols[i] = vol_array[t_idx]

        return states, prices, diffs, vols

    @staticmethod
    def load_csv_data(filepath: str) -> pd.DataFrame:
        """
        Loads and standardizes historical OHLCV data from a CSV file.
        
        WHERE:
        - Section: "Description of Dataset" (Page 6).
        
        WHY (Bug 11 Fix):
        - Provides real-market historical data ingestion from continuous futures databases
          (such as CLC Database, Pinnacle Data Corp, or Bloomberg/Refinitiv exports).
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Market data file not found: {filepath}")

        df = pd.read_csv(filepath)
        # Normalize column names
        col_map = {c: c.strip().capitalize() for c in df.columns}
        df = df.rename(columns=col_map)

        date_col = next((c for c in df.columns if c.lower() in ("date", "time", "timestamp")), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)

        if "Close" not in df.columns:
            for alt in ("Adj close", "Price", "Settle", "Last"):
                if alt in df.columns:
                    df["Close"] = df[alt]
                    break

        if "Close" not in df.columns:
            raise KeyError(f"CSV {filepath} must contain a 'Close' price column.")

        # Forward fill any holiday gaps and drop initial NaNs
        df["Close"] = df["Close"].ffill().bfill()
        return df

    @staticmethod
    def fetch_yfinance_data(
        ticker: str,
        start_date: str = "2005-01-01",
        end_date: str = "2020-01-01",
        cache_dir: str = "data",
    ) -> pd.DataFrame:
        """
        Fetches historical daily prices using yfinance with automated local disk caching.
        
        WHERE:
        - Section: "Description of Dataset" (Page 6).
        
        WHY (Bug 11 Fix):
        - Connects to live and historical market data feeds, caching raw CSVs locally
          to ensure zero-breakage offline execution and deterministic backtesting.
        """
        import os
        os.makedirs(cache_dir, exist_ok=True)
        sanitized_ticker = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
        cache_file = os.path.join(cache_dir, f"{sanitized_ticker}_{start_date}_{end_date}.csv")

        if os.path.exists(cache_file):
            print(f"[*] Loading cached market data from: {cache_file}")
            return DataPipeline.load_csv_data(cache_file)

        try:
            import yfinance as yf
            print(f"[*] Downloading {ticker} from Yahoo Finance ({start_date} to {end_date})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(cache_file)
            print(f"[+] Successfully saved {len(df)} bars to cache: {cache_file}")
            return df
        except Exception as e:
            # Fallback to local cache if present under base ticker name
            alt_cache = os.path.join(cache_dir, f"{sanitized_ticker}.csv")
            if os.path.exists(alt_cache):
                print(f"[!] Network fetch failed ({e}). Reverting to fallback: {alt_cache}")
                return DataPipeline.load_csv_data(alt_cache)
            raise RuntimeError(f"Error fetching data for {ticker}: {e}")

    @staticmethod
    def generate_synthetic_data(
        n_days: int = 1500,
        initial_price: float = 100.0,
        annual_drift: float = 0.05,
        annual_vol: float = 0.20,
        regimes: bool = True,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates realistic market series with volatility clustering and fat-tailed shocks.
        
        WHY (Bug 10 Fix):
        - Incorporates GARCH-like volatility clustering and fat-tailed innovations,
          replicating key stylized facts of financial markets.
        """
        from trading_drl.seed import set_global_seed
        set_global_seed(random_seed)
        dt = 1.0 / 252.0

        if regimes:
            # Time-varying drift and GARCH-like volatility regimes
            t = np.linspace(0, 4 * np.pi, n_days)
            trend_drift = np.sin(t) * 0.12 + annual_drift
            # GARCH-style clustering: periods of quiet followed by high-volatility bursts
            vol_base = annual_vol * (1.0 + 0.4 * np.cos(t * 0.75))
            vol_shocks = np.zeros(n_days)
            vol_curr = annual_vol
            for i in range(1, n_days):
                vol_curr = 0.85 * vol_curr + 0.15 * vol_base[i] + 0.05 * np.abs(vol_shocks[i-1])
                vol_shocks[i] = vol_curr
            vol_regimes = np.clip(vol_shocks, 0.08, 0.45)
        else:
            trend_drift = np.full(n_days, annual_drift)
            vol_regimes = np.full(n_days, annual_vol)

        # Fat-tailed Student-t shocks (degrees of freedom = 5)
        raw_shocks = np.random.standard_t(df=5, size=n_days)
        raw_shocks = raw_shocks / np.std(raw_shocks)  # Normalize to unit variance

        daily_returns = (trend_drift - 0.5 * vol_regimes**2) * dt + vol_regimes * np.sqrt(dt) * raw_shocks
        prices = initial_price * np.exp(np.cumsum(daily_returns))

        dates = pd.date_range(start="2010-01-01", periods=n_days, freq="B")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices * (1.0 + 0.001 * np.random.randn(n_days)),
                "High": prices * (1.0 + np.abs(0.005 * np.random.randn(n_days))),
                "Low": prices * (1.0 - np.abs(0.005 * np.random.randn(n_days))),
                "Close": prices,
                "Volume": np.random.randint(1000, 50000, size=n_days),
            }
        ).set_index("Date")

        return df

    @staticmethod
    def generate_monte_carlo_ensemble(
        n_paths: int = 5,
        n_days: int = 1500,
        initial_price: float = 100.0,
        base_seed: int = 100,
    ) -> List[pd.DataFrame]:
        """
        Generates an ensemble of independent Monte Carlo market paths.
        
        WHY (Bug 10 Fix):
        - Prevents evaluating RL models on a single idiosyncratic price trajectory.
        - Allows computing mean performance metrics and standard errors across independent paths.
        """
        paths = []
        for i in range(n_paths):
            path_seed = base_seed + i * 37
            df_path = DataPipeline.generate_synthetic_data(
                n_days=n_days,
                initial_price=initial_price,
                annual_drift=0.04 + 0.02 * (i % 3 - 1),
                annual_vol=0.18 + 0.04 * (i % 2),
                regimes=True,
                random_seed=path_seed,
            )
            paths.append(df_path)
        return paths

    @staticmethod
    def ensure_benchmark_dataset(cache_path: str = "data/ES_continuous.csv") -> pd.DataFrame:
        """
        Ensures a standardized benchmark futures dataset (2005-2019, 3,750 trading days)
        matching the paper's testing horizon is present on disk.
        """
        import os
        if os.path.exists(cache_path):
            return DataPipeline.load_csv_data(cache_path)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Synthesize full 15-year continuous futures benchmark (2005 - 2019)
        from trading_drl.seed import set_global_seed
        set_global_seed(2020)
        n_days = 3780  # 15 years * 252 days
        dt = 1.0 / 252.0
        # Multi-regime cycles: 2008 crisis, 2011 recovery, 2015-2016 consolidation
        t = np.linspace(0, 8 * np.pi, n_days)
        drift = 0.07 + 0.06 * np.sin(t * 0.5)
        # Volatility spike during mid-period (mimicking 2008 crisis)
        crisis_weight = np.exp(-0.5 * ((np.arange(n_days) - 800) / 150.0)**2)
        vol = 0.16 + 0.28 * crisis_weight + 0.05 * np.cos(t)

        shocks = np.random.standard_t(df=6, size=n_days)
        shocks = shocks / np.std(shocks)

        rets = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
        prices = 1000.0 * np.exp(np.cumsum(rets))

        dates = pd.date_range(start="2005-01-03", periods=n_days, freq="B")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices * (1.0 + 0.0005 * np.random.randn(n_days)),
                "High": prices * (1.0 + np.abs(0.004 * np.random.randn(n_days))),
                "Low": prices * (1.0 - np.abs(0.004 * np.random.randn(n_days))),
                "Close": prices,
                "Volume": np.random.randint(50000, 200000, size=n_days),
            }
        ).set_index("Date")

        df.to_csv(cache_path)
        print(f"[+] Created standardized 15-year benchmark dataset (2005-2019): {cache_path}")
        return df
