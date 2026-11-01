from dataclasses import dataclass

from src.errors import FinSightError
from src.reporting.charts import chart_images
from src.reporting.report_generator import allocation_csv, json_bytes, portfolio_markdown, portfolio_report
from .metrics import Estimates, estimate
from .optimizer import Portfolio, solve_portfolio
from .price_loader import PriceData, load_prices


@dataclass
class Analysis:
    data: PriceData
    estimates: Estimates
    portfolios: list[Portfolio]
    failures: dict[str, str]
    report: dict
    markdown: str
    structured: bytes
    allocation: bytes
    charts: dict[str, bytes]


def analyze(data: bytes, filename: str, provenance: str, simulated: bool, risk_free_rate: float) -> Analysis:
    prices = load_prices(data, filename, provenance, simulated)
    estimates = estimate(prices.prices)
    portfolios, failures = [], {}
    for strategy in ("Minimum variance", "Maximum Sharpe"):
        try:
            portfolios.append(solve_portfolio(estimates, risk_free_rate, strategy))
        except FinSightError as exc:
            failures[strategy] = str(exc)
    if not portfolios:
        raise FinSightError("No strategy could be solved. " + " ".join(f"{key}: {value}" for key, value in failures.items()))
    report = portfolio_report(prices, estimates, portfolios, risk_free_rate, failures)
    return Analysis(prices, estimates, portfolios, failures, report, portfolio_markdown(report), json_bytes(report), allocation_csv(portfolios), chart_images(prices, portfolios))
