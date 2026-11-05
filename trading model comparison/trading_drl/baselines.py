"""
Classical Benchmark and Baseline Trading Strategies.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Baseline Algorithms", Equations (10), (11), (12) (Page 6).
  1. Long only
  2. Sign(R) (Moskowitz, Ooi, and Pedersen 2012; Lim et al. 2019):
     A_t = sign(r_{t-252:t})   (Equation 10)
  3. MACD signal (Baz et al. 2015):
     A_t = phi(MACD_t)
     phi(MACD) = (MACD * exp(-MACD^2 / 4)) / 0.89   (Equation 11)
     where MACD_t = (1/K) * sum_k MACD_t(S_k, L_k)   (Equation 12)
- WHY:
  These three strategies represent the cornerstone benchmarks in quantitative finance:
  - Long Only captures the unconditional asset drift.
  - Sign(R) represents pure 12-month Time-Series Momentum (TSMOM), the standard academic factor.
  - Baz et al. MACD signal provides a continuous, non-linear multi-timescale trend-following benchmark.
  Providing these exact benchmarks ensures rigorous out-of-sample comparison against the 6 RL permutations.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd


class BaseBaselineStrategy:
    """Abstract base class for benchmark trading strategies."""

    def predict(self, state: np.ndarray, info: Optional[dict] = None) -> float:
        """
        Generates target position A_t in [-1.0, 1.0].
        """
        raise NotImplementedError


class LongOnlyStrategy(BaseBaselineStrategy):
    """
    Long Only Benchmark: Always holds maximum long position (+1.0).
    
    WHERE:
    - Section: "Baseline Algorithms" (Page 6).
    
    WHY:
    - Baseline representing unconditional buy-and-hold equity/commodity market risk premium.
    """

    def predict(self, state: Optional[np.ndarray] = None, info: Optional[dict] = None) -> float:
        return 1.0


class SignRStrategy(BaseBaselineStrategy):
    """
    Time Series Momentum (TSMOM) Sign(R) Benchmark.
    
    WHERE:
    - Section: "Baseline Algorithms", Equation (10) (Page 6):
      A_t = sign(r_{t-252:t})
    - Citations: Moskowitz, Ooi, and Pedersen (2012); Lim, Zohren, and Roberts (2019).
    
    WHY:
    - Academic standard for 12-month trend following. If past 1-year cumulative return
      is positive, the position is +1; if negative, -1; if zero, 0.
    """

    def predict(self, state: np.ndarray, info: Optional[dict] = None) -> float:
        """
        Extracts 252-day return from state or info and returns sign.
        State format: (60, num_features). Feature index 4 is vol_adj_return_252d.
        """
        if info is not None and "return_252d" in info:
            val = info["return_252d"]
        else:
            # Last observation in state sequence: feature column 4 (vol_adj_return_252d)
            val = state[-1, 4]

        if val > 0:
            return 1.0
        elif val < 0:
            return -1.0
        return 0.0


class MACDSignalStrategy(BaseBaselineStrategy):
    """
    Baz et al. (2015) Multi-Scale MACD Benchmark with Non-linear Sizing Function.
    
    WHERE:
    - Section: "Baseline Algorithms", Equations (11) and (12) (Page 6):
      A_t = phi(MACD_t)
      phi(MACD) = (MACD * exp(-MACD^2 / 4)) / 0.89
      MACD_t = sum_k MACD_t(S_k, L_k)
    
    WHY:
    - Baz et al. non-linear response function phi(x) scales positions smoothly with trend strength.
      At MACD = sqrt(2) (~1.414), phi achieves its maximum of exactly 1.0.
      For extreme values (potential overextended spikes), phi gently dampens exposure,
      mitigating whipsaw risk at trend turning points.
    """

    @staticmethod
    def phi_response(macd: float) -> float:
        """
        Non-linear response transfer function from Equation (11).
        phi(MACD) = (MACD * exp(-MACD^2 / 4)) / 0.89
        """
        return float((macd * np.exp(-(macd**2) / 4.0)) / 0.89)

    def predict(self, state: np.ndarray, info: Optional[dict] = None) -> float:
        """
        Calculates position based on composite MACD feature in state.
        Feature column 8 is composite_macd.
        """
        if info is not None and "composite_macd" in info:
            macd_val = float(info["composite_macd"])
        else:
            # Last observation in state sequence: feature column 8 is composite_macd
            macd_val = float(state[-1, 8])

        pos = self.phi_response(macd_val)
        return float(np.clip(pos, -1.0, 1.0))
