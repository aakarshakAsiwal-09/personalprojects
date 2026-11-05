"""
Portfolio Evaluation, Performance Metrics, and Volatility Targeting Module.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Section "Experimental Results" (Page 7-8), Exhibit 2, Exhibit 4, and Exhibit B1.
  Metrics specified (Page 7-8):
  1. E(R): annualized expected trade returns
  2. std(R): annualized standard deviation of trade returns
  3. Downside deviation (DD): annualized standard deviation of negative trade returns
  4. Sharpe: annualized Sharpe ratio (E(R) / std(R))
  5. Sortino: downside risk-adjusted return (E(R) / DD)
  6. MDD: maximum peak-to-trough drawdown
  7. Calmar: ratio of expected annual return to MDD (E(R) / MDD)
  8. % +ve Returns: fraction of days with positive return
  9. Ave. P / Ave. L: ratio of average gain on winning days to average loss on losing days
  10. Turnover & Return per Turnover: average turnover and return per unit turnover (Exhibit 4)
  11. Portfolio Volatility Targeting:
      "an additional layer of portfolio-level volatility scaling is applied for each model.
       This brings the volatility of different methods to the same target, so we can directly
       compare metrics such as expected and cumulative trade returns." (Page 8)
  12. Equation (13):
      R^{port}_t = (1 / N) * sum_{i=1}^N R_t^i   (Equal-weighted portfolio across contracts)
- WHY:
  Raw backtest metrics can be misleading when strategy risk profiles diverge.
  Annualizing with sqrt(252), isolating downside semivariance (Sortino), and scaling to a common
  portfolio volatility target guarantees mathematically rigorous, fair comparisons across all
  combinatorial algorithms (DQN, PG, A2C in Discrete vs. Continuous modes) and classical baselines.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd


class PortfolioEvaluator:
    """
    Evaluator computing the full suite of paper-defined metrics and portfolio volatility targeting.
    """

    def __init__(self, annualization_factor: int = 252):
        self.ann_factor = annualization_factor
        self.sqrt_ann = np.sqrt(annualization_factor)

    @classmethod
    def apply_ex_ante_volatility_targeting(
        cls,
        returns: np.ndarray,
        target_annual_vol: float = 0.15,
        lookback: int = 60,
        annualization_factor: int = 252,
        max_leverage: float = 4.0,
    ) -> np.ndarray:
        """
        Applies causal ex-ante portfolio-level volatility scaling (Harvey et al. 2018; Zhang et al. 2020).
        
        WHERE:
        - Section: "Experimental Results -> Exhibit 2" (Page 8).
        
        WHY (Bug 7 Fix):
        - Eliminates look-ahead bias and prevents the volatility-targeted Sharpe from collapsing
          into a trivial rescale of mean returns.
        - Position scale factor S_t = (sigma_tgt / sqrt(252)) / sigma_{t-1} is computed strictly
          using ex-ante returns up to t-1 via 60-day EWMA standard deviation.
        """
        r = np.asarray(returns, dtype=np.float64)
        if len(r) < 2:
            return r

        target_daily_vol = target_annual_vol / np.sqrt(annualization_factor)
        r_series = pd.Series(r)

        # Compute 60-day EWMA std strictly shifted by 1 (no future lookahead)
        ewm_std = r_series.ewm(span=lookback, min_periods=max(5, lookback // 4)).std()
        ex_ante_vol = ewm_std.shift(1).bfill().fillna(target_daily_vol).values
        ex_ante_vol = np.clip(ex_ante_vol, 1e-6, None)

        scales = np.clip(target_daily_vol / ex_ante_vol, 0.1, max_leverage)
        return r * scales

    def calculate_metrics(
        self,
        returns: Union[np.ndarray, pd.Series, List[float]],
        turnovers: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        apply_vol_target: bool = False,
        target_annual_vol: float = 0.15,
        ex_ante: bool = True,
    ) -> Dict[str, float]:
        """
        Calculates all 9 primary metrics from Exhibit 2, plus turnover metrics.
        
        Args:
            returns: Array of daily net trade returns R_t (Equation 4).
            turnovers: Optional array of daily turnovers.
            apply_vol_target: If True, applies portfolio-level volatility targeting.
            target_annual_vol: Target annualized volatility (default 0.15).
            ex_ante: If True (default), applies strictly causal ex-ante volatility scaling (Bug 7 Fix).
            
        Returns:
            Dictionary containing all metrics.
        """
        r = np.asarray(returns, dtype=np.float64)
        if len(r) == 0:
            return {}

        # Causal Portfolio-level Volatility Targeting (Bug 7 Fix)
        if apply_vol_target:
            if ex_ante:
                r = self.apply_ex_ante_volatility_targeting(
                    r,
                    target_annual_vol=target_annual_vol,
                    annualization_factor=self.ann_factor,
                )
            else:
                raw_std = float(np.std(r, ddof=1)) if len(r) > 1 else 1e-4
                raw_annual_vol = raw_std * self.sqrt_ann
                if raw_annual_vol > 1e-6:
                    vol_scale = target_annual_vol / raw_annual_vol
                    r = r * vol_scale

        # 1. E(R): Annualized expected trade return
        # WHERE: Page 8: "1. E(R): annualized expected trade returns"
        mean_ret = float(np.mean(r))
        e_r = mean_ret * self.ann_factor

        # 2. std(R): Annualized standard deviation of trade returns
        # WHERE: Page 8: "2. std(R): annualized standard deviation of trade returns"
        std_ret = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        std_r = std_ret * self.sqrt_ann

        # 3. Downside deviation (DD): Semi-standard deviation of negative returns
        # WHERE: Page 8: "3. Downside deviation (DD): annualized standard deviation of trade returns that are negative"
        neg_returns = r[r < 0.0]
        if len(neg_returns) > 0:
            # Root mean square of negative deviations from zero
            downside_variance = np.mean(neg_returns**2)
            dd = float(np.sqrt(downside_variance) * self.sqrt_ann)
        else:
            dd = 1e-6

        # 4. Sharpe Ratio: E(R) / std(R)
        # WHERE: Page 8: "4. Sharpe: annualized Sharpe ratio (E(R)/std(R))"
        sharpe = e_r / (std_r + 1e-8)

        # 5. Sortino Ratio: E(R) / DD
        # WHERE: Page 8: "5. Sortino: a variant of the Sharpe ratio that uses downside deviation as risk measures"
        sortino = e_r / (dd + 1e-8)

        # 6. Maximum Drawdown (MDD)
        # WHERE: Page 8: "6. MDD: maximum drawdown shows the maximum observed loss from any peak of a portfolio to the trough"
        cum_wealth = np.cumsum(r)
        running_max = np.maximum.accumulate(cum_wealth)
        drawdowns = running_max - cum_wealth
        mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 7. Calmar Ratio: E(R) / MDD
        # WHERE: Page 8: "7. Calmar: the Calmar ratio compares the expected annual rate of return with MDD"
        calmar = e_r / (mdd + 1e-8)

        # 8. % +ve Returns: Fraction of positive returns
        # WHERE: Page 8: "8. % +ve Returns: percentage of positive trade returns"
        pos_returns = r[r > 0.0]
        pct_positive = float(len(pos_returns) / len(r))

        # 9. Ave. P / Ave. L: Ratio of average gain to average loss
        # WHERE: Page 8: "9. Ave. P / Ave. L: the ratio between positive and negative trade returns"
        avg_gain = float(np.mean(pos_returns)) if len(pos_returns) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(neg_returns))) if len(neg_returns) > 0 else 1e-6
        ave_p_ave_l = avg_gain / avg_loss

        metrics = {
            "E(R)": round(e_r, 4),
            "Std(R)": round(std_r, 4),
            "DD": round(dd, 4),
            "Sharpe": round(sharpe, 4),
            "Sortino": round(sortino, 4),
            "MDD": round(mdd, 4),
            "Calmar": round(calmar, 4),
            "% +ve Ret": round(pct_positive, 4),
            "Ave. P / Ave. L": round(ave_p_ave_l, 4),
            "Cumulative Return": round(float(cum_wealth[-1]), 4) if len(cum_wealth) > 0 else 0.0,
        }

        # 10. Turnover metrics (Exhibit 4)
        if turnovers is not None:
            t_arr = np.asarray(turnovers, dtype=np.float64)
            mean_turnover = float(np.mean(t_arr))
            annual_turnover = mean_turnover * self.ann_factor
            ret_per_turnover = e_r / (annual_turnover + 1e-8)
            metrics["Mean Daily Turnover"] = round(mean_turnover, 4)
            metrics["Annualized Turnover"] = round(annual_turnover, 4)
            metrics["Ave. Return / Turnover"] = round(ret_per_turnover, 4)

        return metrics

    @staticmethod
    def aggregate_equal_weight_portfolio(
        contract_returns: List[np.ndarray],
    ) -> np.ndarray:
        """
        Computes equal-weighted portfolio trade return across N contracts as defined in Equation (13).
        
        WHERE:
        - Equation (13), Page 7:
          R^{port}_t = (1 / N) * sum_{i=1}^N R_t^i
        
        WHY:
        - Diversifies idiosyncratic contract noise while evaluating strategy systematic alpha across assets.
        """
        min_len = min(len(r) for r in contract_returns)
        aligned = [r[:min_len] for r in contract_returns]
        matrix = np.array(aligned)  # (N, min_len)
        return np.mean(matrix, axis=0)

    def compile_matrix_comparison(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        apply_vol_target: bool = False,
    ) -> pd.DataFrame:
        """
        Compiles performance metrics dictionary into a styled pandas DataFrame table
        matching Exhibit 2 in the paper.
        """
        records = []
        for name, data in results_dict.items():
            returns = data.get("returns", [])
            turnovers = data.get("turnovers", None)
            m = self.calculate_metrics(returns, turnovers=turnovers, apply_vol_target=apply_vol_target)
            m["Model"] = name
            records.append(m)

        df = pd.DataFrame(records)
        cols = [
            "Model",
            "E(R)",
            "Std(R)",
            "DD",
            "Sharpe",
            "Sortino",
            "MDD",
            "Calmar",
            "% +ve Ret",
            "Ave. P / Ave. L",
        ]
        if "Mean Daily Turnover" in df.columns:
            cols.extend(["Mean Daily Turnover", "Ave. Return / Turnover"])

        # Reorder columns
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].set_index("Model")
        return df

    def compute_exact_cost_sensitivity(
        self,
        gross_returns: np.ndarray,
        turnovers: np.ndarray,
        prices: np.ndarray,
        cost_bps_levels: List[float],
        mu: float = 1.0,
        apply_vol_target: bool = False,
    ) -> Dict[float, Dict[str, float]]:
        """
        Calculates exact per-step transaction cost sensitivity across varying basis point rates.
        
        WHERE:
        - Section: Exhibit 5 (Page 9): "Performance under different transaction cost rates."
        - Equation (4) (Page 5).
        
        WHY (Bug 8 Fix):
        - Eliminates the previous approximation shortcut that subtracted (Delta bp * mean_vol * turnovers).
        - Recomputes exact per-step penalties: Cost_t = bp * p_{t-1} * Turnover_t.
        - Preserves authentic cost dynamics across volatile vs calm periods.
        """
        gross_r = np.asarray(gross_returns, dtype=np.float64)
        t_overs = np.asarray(turnovers, dtype=np.float64)
        px = np.asarray(prices, dtype=np.float64)
        results: Dict[float, Dict[str, float]] = {}

        for bp in cost_bps_levels:
            rate = bp * 0.0001
            exact_cost = rate * px * t_overs
            exact_net = mu * (gross_r - exact_cost)
            metrics = self.calculate_metrics(exact_net, turnovers=t_overs, apply_vol_target=apply_vol_target)
            results[bp] = metrics

        return results

    def compile_monte_carlo_summary(
        self,
        path_results_list: List[Dict[str, Dict[str, Any]]],
        apply_vol_target: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compiles mean performance metrics and standard errors across independent Monte Carlo paths.
        
        WHERE:
        - Section: "Description of Dataset" and "Experimental Results" (Pages 6-9).
        
        WHY (Bug 10 Fix):
        - Eliminates single-path evaluation bias by aggregating results across M independent market trajectories.
        - Reports empirical Mean +/- Standard Error for each metric, enabling statistically rigorous
          comparisons between discrete and continuous action spaces.
          
        Returns:
            display_df: Formatted DataFrame with "Mean +/- SE" strings for human readability.
            numeric_df: DataFrame with raw Mean and SE floats for plotting and downstream analysis.
        """
        n_paths = len(path_results_list)
        if n_paths == 0:
            return pd.DataFrame(), pd.DataFrame()

        model_metrics: Dict[str, Dict[str, List[float]]] = {}
        metric_keys = ["E(R)", "Std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve Ret", "Mean Daily Turnover"]

        for path_res in path_results_list:
            comp_df = self.compile_matrix_comparison(path_res, apply_vol_target=apply_vol_target)
            for model_name in comp_df.index:
                if model_name not in model_metrics:
                    model_metrics[model_name] = {k: [] for k in metric_keys}
                for k in metric_keys:
                    if k in comp_df.columns:
                        model_metrics[model_name][k].append(float(comp_df.loc[model_name, k]))

        display_rows = []
        numeric_rows = []

        for model_name, metrics in model_metrics.items():
            disp_entry = {"Model": model_name}
            num_entry = {"Model": model_name}
            for k in metric_keys:
                vals = np.array(metrics[k], dtype=np.float64)
                if len(vals) > 0:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    se_val = std_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

                    disp_entry[k] = f"{mean_val:.2f} ± {se_val:.2f}"
                    num_entry[f"{k}_Mean"] = mean_val
                    num_entry[f"{k}_SE"] = se_val
                else:
                    disp_entry[k] = "N/A"
            display_rows.append(disp_entry)
            numeric_rows.append(num_entry)

        display_df = pd.DataFrame(display_rows).set_index("Model")
        numeric_df = pd.DataFrame(numeric_rows).set_index("Model")
        return display_df, numeric_df

