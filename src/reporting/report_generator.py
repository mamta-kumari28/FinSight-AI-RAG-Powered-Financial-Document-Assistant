import json
from datetime import datetime, timezone

import pandas as pd

from src.portfolio.metrics import Estimates
from src.portfolio.optimizer import Portfolio
from src.portfolio.price_loader import PriceData
from src.rag.qa_chain import Answer

ASSUMPTIONS = [
    "Educational historical illustration; not a backtest, trading recommendation, or forecast.",
    "Use aligned daily adjusted closing prices; corporate-action adjustment and provenance are the user's responsibility.",
    "Arithmetic mean daily returns and sample covariance (ddof=1), annualized using 252 trading days.",
    "Long-only, fully invested portfolios; constant annual risk-free rate; no leverage or shorting.",
    "Transaction costs, taxes, slippage, liquidity, and rebalancing costs are excluded.",
    "Historical estimates and optimized weights are uncertain and sensitive to the sample; future returns are not guaranteed.",
    "A small diagonal covariance ridge is included in optimization and reported portfolio metrics.",
]


def portfolio_report(data: PriceData, estimates: Estimates, portfolios: list[Portfolio], risk_free_rate: float, failures: dict | None = None) -> dict:
    return {
        "project": "FinSight AI", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename": data.filename, "assets": list(data.prices.columns), "provenance": data.provenance,
        "simulated": data.simulated, "date_start": str(data.prices.index[0].date()), "date_end": str(data.prices.index[-1].date()),
        "price_observations": len(data.prices), "return_observations": len(estimates.returns),
        "annual_risk_free_rate": risk_free_rate, "trading_days_per_year": 252, "covariance_ridge": estimates.ridge,
        "estimates": {"expected_returns": estimates.expected_returns.to_dict(), "sample_covariance": estimates.sample_covariance.to_dict(), "regularized_covariance": estimates.covariance.to_dict()},
        "portfolios": [{"strategy": p.strategy, "weights": p.weights.to_dict(), "metrics": p.metrics, "solver": p.solver, "status": p.status} for p in portfolios],
        "unavailable_strategies": failures or {}, "assumptions": ASSUMPTIONS,
    }


def portfolio_markdown(report: dict) -> str:
    lines = ["# FinSight AI — Portfolio Research Report", "", f"Generated (UTC): {report['generated_at_utc']}",
             f"Data: {report['filename']}", f"Provenance: {report['provenance']}",
             f"Simulated data: {'YES — fictional prices' if report['simulated'] else 'No (user-declared)'}",
             f"Date window: {report['date_start']} to {report['date_end']}",
             f"Price observations: {report['price_observations']}; returns: {report['return_observations']}",
             f"Annual risk-free rate: {report['annual_risk_free_rate']:.2%}",
             f"Covariance diagonal ridge: {report['covariance_ridge']:.12g}", ""]
    for portfolio in report["portfolios"]:
        metrics = portfolio["metrics"]
        lines += [f"## {portfolio['strategy']}", f"Solver: {portfolio['solver']} ({portfolio['status']})",
                  f"Estimated annual return: {metrics['expected_return']:.2%}", f"Annual volatility: {metrics['volatility']:.2%}",
                  f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}", "", "| Asset | Weight |", "| --- | ---: |"]
        lines += [f"| {asset} | {weight:.4%} |" for asset, weight in portfolio["weights"].items()]
        lines += [""]
    for strategy, message in report["unavailable_strategies"].items():
        lines += [f"{strategy} unavailable: {message}", ""]
    lines += ["## Assumptions and limitations", *[f"- {item}" for item in report["assumptions"]]]
    return "\n".join(lines)


def allocation_csv(portfolios: list[Portfolio]) -> bytes:
    frame = pd.DataFrame({p.strategy: p.weights for p in portfolios})
    frame.index.name = "Asset"
    # Prefix potentially executable cells to avoid spreadsheet formula execution on opening.
    frame.index = ["'" + str(name) if str(name).startswith(("=", "+", "-", "@")) else name for name in frame.index]
    return frame.to_csv().encode("utf-8")


def json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def qa_report(answer: Answer, models: dict) -> tuple[str, dict]:
    report = {"project": "FinSight AI", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "question": answer.question, "answer": answer.text, "insufficient": answer.insufficient,
              "sources": answer.sources, "models": models,
              "retrieved_evidence": [{"id": f"E{i}", "source": e.source, "text": e.text, "cosine_distance": e.distance} for i, e in enumerate(answer.evidence, 1)]}
    markdown = "\n".join(["# FinSight AI — Document Q&A", "", f"Question: {answer.question}", "", answer.text, "", "## Sources",
                          *[f"- {s['id']}: {s['source']}" for s in answer.sources], "", "## Retrieved evidence",
                          *[f"### {item['id']} — {item['source']}\n\n{item['text']}\n" for item in report["retrieved_evidence"]],
                          "Educational research only. Citations indicate provenance; verify claims against original documents."])
    return markdown, report
