from dataclasses import replace
from io import StringIO

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src.errors import FinSightError
from src.portfolio.analysis import analyze
from src.portfolio.metrics import Estimates, estimate, portfolio_metrics
from src.portfolio.optimizer import solve_portfolio
from src.portfolio.price_loader import load_prices


def price_frame():
    index = pd.bdate_range('2025-01-01', periods=65)
    n = np.arange(65)
    return pd.DataFrame({'Date': index.strftime('%Y-%m-%d'), 'A': 100 + n + np.sin(n), 'B': 80 + n * .4 + np.cos(n)})


def as_bytes(frame):
    return frame.to_csv(index=False).encode()


def test_price_load_and_annualization():
    data = load_prices(as_bytes(price_frame()), 'prices.csv', 'test source', True)
    stats = estimate(data.prices)
    returns = data.prices.pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_frame_equal(stats.returns, returns)
    pd.testing.assert_series_equal(stats.expected_returns, returns.mean() * 252)
    pd.testing.assert_frame_equal(stats.sample_covariance, returns.cov(ddof=1) * 252)
    assert stats.ridge > 0
    np.testing.assert_allclose(stats.covariance, stats.sample_covariance + np.eye(2) * stats.ridge)


@pytest.mark.parametrize('case', ['duplicate_date', 'invalid_date', 'unsorted', 'missing', 'nonnumeric', 'zero', 'negative', 'infinite', 'short', 'one_asset', 'missing_provenance'])
def test_price_validation(case):
    frame = price_frame()
    provenance = 'test'
    if case == 'duplicate_date': frame.loc[1, 'Date'] = frame.loc[0, 'Date']
    if case == 'invalid_date': frame.loc[1, 'Date'] = '2025-02-30'
    if case == 'unsorted': frame = frame.iloc[::-1]
    if case == 'missing': frame.loc[1, 'A'] = np.nan
    if case == 'nonnumeric':
        frame['A'] = frame['A'].astype(str)
        frame.loc[1, 'A'] = 'unknown'
    if case == 'zero': frame.loc[1, 'A'] = 0
    if case == 'negative': frame.loc[1, 'A'] = -1
    if case == 'infinite': frame.loc[1, 'A'] = np.inf
    if case == 'short': frame = frame.iloc[:60]
    if case == 'one_asset': frame = frame.drop(columns='B')
    if case == 'missing_provenance': provenance = ''
    with pytest.raises(FinSightError):
        load_prices(as_bytes(frame), 'prices.csv', provenance)


def test_duplicate_asset_headers():
    data = as_bytes(price_frame()).replace(b'Date,A,B', b'Date,A,A')
    with pytest.raises(FinSightError, match='Duplicate'):
        load_prices(data, 'prices.csv', 'test')


def analytic_estimates():
    mean = pd.Series([.12, .18, .08], index=['A', 'B', 'C'])
    covariance = pd.DataFrame(np.diag([.04, .09, .01]), index=mean.index, columns=mean.index)
    return Estimates(pd.DataFrame(), mean, covariance, covariance, 0)


def test_minimum_variance_analytic_solution():
    stats = analytic_estimates()
    portfolio = solve_portfolio(stats, .03, 'Minimum variance')
    expected = 1 / np.diag(stats.covariance)
    expected /= expected.sum()
    np.testing.assert_allclose(portfolio.weights, expected, atol=1e-5)
    assert portfolio.weights.min() >= 0
    assert portfolio.weights.sum() == pytest.approx(1)


def test_maximum_sharpe_analytic_and_grid_comparison():
    stats = analytic_estimates()
    portfolio = solve_portfolio(stats, .03, 'Maximum Sharpe')
    expected = np.linalg.solve(stats.covariance, stats.expected_returns - .03)
    expected /= expected.sum()
    np.testing.assert_allclose(portfolio.weights, expected, atol=2e-4)
    best_grid = -np.inf
    for a in np.linspace(0, 1, 31):
        for b in np.linspace(0, 1 - a, 31):
            metrics = portfolio_metrics([a, b, 1 - a - b], stats.expected_returns, stats.covariance, .03)
            best_grid = max(best_grid, metrics['sharpe_ratio'])
    assert portfolio.metrics['sharpe_ratio'] >= best_grid - 1e-5


def test_max_sharpe_no_positive_excess_min_variance_still_works():
    stats = analytic_estimates()
    with pytest.raises(FinSightError, match='positive estimated excess'):
        solve_portfolio(stats, .5, 'Maximum Sharpe')
    assert solve_portfolio(stats, .5, 'Minimum variance').metrics['sharpe_ratio'] < 0


@pytest.mark.parametrize('rate', [np.nan, np.inf, -1, 11])
def test_invalid_risk_free_rate(rate):
    with pytest.raises(FinSightError, match='risk-free'):
        solve_portfolio(analytic_estimates(), rate, 'Minimum variance')


def test_missing_solver_and_solver_exception(monkeypatch):
    monkeypatch.setattr(cp, 'installed_solvers', lambda: [])
    with pytest.raises(FinSightError, match='No supported solver'):
        solve_portfolio(analytic_estimates(), .03, 'Minimum variance')
    monkeypatch.setattr(cp, 'installed_solvers', lambda: ['CLARABEL'])
    def fail(*args, **kwargs):
        raise cp.error.SolverError('synthetic failure')
    monkeypatch.setattr(cp.Problem, 'solve', fail)
    with pytest.raises(FinSightError, match='no fallback'):
        solve_portfolio(analytic_estimates(), .03, 'Minimum variance')


def test_nonconvergence(monkeypatch):
    def fail(self, **kwargs):
        self._status = cp.INFEASIBLE
    monkeypatch.setattr(cp.Problem, 'solve', fail)
    with pytest.raises(FinSightError, match='converge'):
        solve_portfolio(analytic_estimates(), .03, 'Minimum variance')


def test_reject_bad_covariance():
    stats = analytic_estimates()
    with pytest.raises(FinSightError, match='positive semidefinite'):
        solve_portfolio(replace(stats, covariance=-stats.covariance), .03, 'Minimum variance')


def test_singular_correlated_prices_regularized():
    prices = price_frame().drop(columns='Date')
    prices['B'] = prices['A'] * 2
    stats = estimate(prices)
    assert np.linalg.eigvalsh(stats.covariance).min() > 0
    weights = solve_portfolio(stats, .03, 'Minimum variance').weights
    assert weights.sum() == pytest.approx(1)


def test_full_pipeline_outputs_and_unsupported_strategy(samples):
    data = (samples / 'synthetic_prices.csv').read_bytes()
    result = analyze(data, 'synthetic_prices.csv', 'SIMULATED PCG64 seed 2025', True, .04)
    assert len(result.portfolios) == 2 and not result.failures
    assert result.report['simulated'] and result.report['price_observations'] == 253
    assert set(result.charts) == {'rebased_prices.png', 'allocations.png', 'risk_return.png'}
    assert all(image.startswith(b'\x89PNG\r\n\x1a\n') and len(image) > 1000 for image in result.charts.values())
    unsupported = analyze(data, 'synthetic_prices.csv', 'SIMULATED', True, 1)
    assert len(unsupported.portfolios) == 1
    assert 'Maximum Sharpe' in unsupported.failures


@pytest.mark.parametrize('values', [[float('nan'), 0, 1], [-.1, .5, .6], [.2, .2, .2]])
def test_invalid_solver_weights_rejected(monkeypatch, values):
    def invalid(self, **kwargs):
        self.variables()[0]._value = np.array(values)
        self._status = cp.OPTIMAL
    monkeypatch.setattr(cp.Problem, 'solve', invalid)
    with pytest.raises(FinSightError, match='validation|constraint'):
        solve_portfolio(analytic_estimates(), .03, 'Minimum variance')


def test_maximum_sharpe_transformation_residual_rejected(monkeypatch):
    def invalid(self, **kwargs):
        self.variables()[0]._value = np.array([.2, .3, .5])
        self._status = cp.OPTIMAL
    monkeypatch.setattr(cp.Problem, 'solve', invalid)
    with pytest.raises(FinSightError, match='transformation constraint'):
        solve_portfolio(analytic_estimates(), .03, 'Maximum Sharpe')
