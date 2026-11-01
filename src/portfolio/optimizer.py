from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from src.errors import FinSightError
from .metrics import Estimates, portfolio_metrics


@dataclass(frozen=True)
class Portfolio:
    strategy: str
    weights: pd.Series
    metrics: dict[str, float]
    solver: str
    status: str


def solve_portfolio(estimates: Estimates, risk_free_rate: float, strategy: str, solver: str | None = None) -> Portfolio:
    if not np.isfinite(risk_free_rate) or not -1 < risk_free_rate <= 10:
        raise FinSightError("Annual risk-free rate must be finite, greater than -100%, and no more than 1000%.")
    if strategy not in {"Minimum variance", "Maximum Sharpe"}:
        raise FinSightError("Unknown portfolio strategy.")
    available = cp.installed_solvers()
    solver = solver or next((name for name in ("CLARABEL", "OSQP", "SCS") if name in available), None)
    if solver not in available or solver not in {"CLARABEL", "OSQP", "SCS"}:
        raise FinSightError("No supported solver is available. Install CLARABEL, OSQP, or SCS and retry.")
    mean = estimates.expected_returns.to_numpy(dtype=float)
    covariance = estimates.covariance.to_numpy(dtype=float)
    if covariance.shape != (len(mean), len(mean)) or not np.isfinite(covariance).all() or not np.isfinite(mean).all():
        raise FinSightError("Invalid portfolio estimates.")
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise FinSightError("Covariance must be symmetric and positive semidefinite.")
    variable = cp.Variable(len(mean))
    constraints = [variable >= 0]
    if strategy == "Minimum variance":
        constraints += [cp.sum(variable) == 1]
    else:
        excess = mean - risk_free_rate
        if float(excess.max()) <= 1e-10:
            raise FinSightError("Maximum Sharpe requires at least one asset with positive estimated excess return at the selected risk-free rate. Minimum variance remains available.")
        # y = w / ((mu-rf)^T w). Min y^T Sigma y, excess^T y = 1.
        constraints += [excess @ variable == 1]
    problem = cp.Problem(cp.Minimize(cp.quad_form(variable, cp.psd_wrap(covariance))), constraints)
    try:
        problem.solve(solver=solver)
    except Exception as exc:
        raise FinSightError("Portfolio solver failed. Check data conditioning or install another supported solver; no fallback allocation was substituted.") from exc
    if problem.status != cp.OPTIMAL or variable.value is None:
        raise FinSightError("The solver did not converge to an optimal solution. Try better-conditioned data or another solver.")
    values = np.asarray(variable.value, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or values.min() < -1e-7 or values.sum() <= 0:
        raise FinSightError("Solver output failed finite/nonnegative weight validation.")
    if strategy == "Minimum variance" and abs(values.sum() - 1) > 1e-6:
        raise FinSightError("Solver output failed the fully-invested constraint.")
    if strategy == "Maximum Sharpe" and abs(float((mean - risk_free_rate) @ values) - 1) > 1e-5:
        raise FinSightError("Solver output failed the maximum-Sharpe transformation constraint.")
    # Clip only tiny solver roundoff, then normalize. This is not a fallback allocation.
    weights = np.clip(values, 0, None)
    weights /= weights.sum()
    if abs(weights.sum() - 1) > 1e-8:
        raise FinSightError("Normalized weights do not sum to one.")
    metrics = portfolio_metrics(weights, mean, covariance, risk_free_rate)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FinSightError("Portfolio metrics are not finite.")
    return Portfolio(strategy, pd.Series(weights, index=estimates.expected_returns.index, name="Weight"), metrics, solver, problem.status)
