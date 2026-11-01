from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.errors import FinSightError

TRADING_DAYS = 252


@dataclass(frozen=True)
class Estimates:
    returns: pd.DataFrame
    expected_returns: pd.Series
    sample_covariance: pd.DataFrame
    covariance: pd.DataFrame
    ridge: float


def estimate(prices: pd.DataFrame) -> Estimates:
    values = prices.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise FinSightError("At least three aligned observations of two positive, finite asset-price series are required.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise FinSightError("Returns contain invalid or overflowing values.")
    expected = returns.mean() * TRADING_DAYS
    covariance = returns.cov(ddof=1) * TRADING_DAYS
    raw = covariance.to_numpy()
    if not np.isfinite(raw).all() or not np.isfinite(expected.to_numpy()).all():
        raise FinSightError("Return estimates are not finite. Check the price magnitudes.")
    # Small scale-aware diagonal ridge; same adjusted covariance is used in solving and metrics.
    ridge = max(float(np.trace(raw) / len(raw)), 1e-6) * 1e-8
    regularized = (raw + raw.T) / 2 + np.eye(len(raw)) * ridge
    return Estimates(returns, expected, covariance, pd.DataFrame(regularized, index=prices.columns, columns=prices.columns), ridge)


def portfolio_metrics(weights, expected, covariance, risk_free_rate: float) -> dict[str, float]:
    weights, expected, covariance = map(lambda x: np.asarray(x, dtype=float), (weights, expected, covariance))
    if weights.ndim != 1 or expected.shape != weights.shape or covariance.shape != (len(weights), len(weights)):
        raise FinSightError("Portfolio dimensions do not match.")
    if not all(np.isfinite(x).all() for x in (weights, expected, covariance)) or not np.isfinite(risk_free_rate):
        raise FinSightError("Portfolio inputs must be finite.")
    annual_return = float(expected @ weights)
    variance = float(weights @ covariance @ weights)
    if variance <= 0:
        raise FinSightError("Portfolio volatility is zero or invalid; Sharpe ratio is undefined.")
    volatility = float(np.sqrt(variance))
    return {"expected_return": annual_return, "volatility": volatility, "sharpe_ratio": (annual_return - risk_free_rate) / volatility}
