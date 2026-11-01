from io import BytesIO
from threading import RLock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.portfolio.price_loader import PriceData
from src.portfolio.optimizer import Portfolio

CHART_LOCK = RLock()
COLORS = ["#2563eb", "#0d9488", "#f59e0b", "#8b5cf6", "#ef4444"]


def chart_images(data: PriceData, portfolios: list[Portfolio]) -> dict[str, bytes]:
    """Render and close charts under a lock; Matplotlib's global state is not thread safe."""
    with CHART_LOCK, plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}):
        charts = {}
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        (data.prices / data.prices.iloc[0] * 100).plot(ax=ax, color=COLORS)
        ax.set(ylabel="Rebased price (first observation = 100)", xlabel="Date", title="Historical asset prices" + (" — SIMULATED" if data.simulated else ""))
        charts["rebased_prices.png"] = _png(fig)
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        import pandas as pd
        allocation = pd.DataFrame({p.strategy: p.weights for p in portfolios}) * 100
        allocation.plot.bar(ax=ax, color=COLORS, rot=0)
        ax.set(ylabel="Allocation (%)", xlabel="Asset", title="Long-only portfolio allocations")
        charts["allocations.png"] = _png(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for color, portfolio in zip(COLORS, portfolios):
            x, y = portfolio.metrics["volatility"] * 100, portfolio.metrics["expected_return"] * 100
            ax.scatter(x, y, s=130, color=color, label=portfolio.strategy)
            ax.annotate(portfolio.strategy, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.margins(x=0.35, y=0.35)
        ax.set(xlabel="Annualized historical volatility (%)", ylabel="Annualized estimated return (%)", title="Historical risk–return comparison")
        charts["risk_return.png"] = _png(fig)
        return charts


def _png(fig: Figure) -> bytes:
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)
    return stream.getvalue()
