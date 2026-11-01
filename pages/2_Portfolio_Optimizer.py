import streamlit as st
import pandas as pd

from src.config import ROOT
from src.errors import FinSightError
from src.portfolio.analysis import analyze
from src.state import fingerprint, sync_inputs

st.set_page_config(page_title="Portfolio Optimizer | FinSight AI", page_icon="📈", layout="wide")
st.title("📈 Portfolio Optimizer")
st.write("Compare long-only, fully invested minimum-variance and maximum-Sharpe portfolios from aligned historical daily prices. No Gemini API key is required.")
st.info("Historical research illustration, not a backtest or forecast. Estimates are uncertain; future returns are not guaranteed. Transaction costs, taxes, and slippage are excluded.")
with st.expander("Input requirements and sample download", expanded=True):
    st.write("Upload CSV with Date (YYYY-MM-DD), at least two asset columns, and at least 61 daily observations. Use adjusted closing prices. Prices must be positive, finite, aligned, and complete. Dates must be unique and increasing; invalid rows are rejected.")
    st.download_button("Download simulated daily prices", (ROOT / "data/samples/synthetic_prices.csv").read_bytes(), file_name="synthetic_prices.csv", mime="text/csv")
source = st.radio("Price source", ["Upload price CSV", "Simulated example"], horizontal=True)
data, filename = b"", ""
if source == "Simulated example":
    filename = "synthetic_prices.csv"
    data = (ROOT / "data/samples" / filename).read_bytes()
    provenance = "Deterministic simulated daily prices; NumPy PCG64 seed 2025; no real market data."
    simulated = True
    st.warning("SIMULATED DATA — fictional assets and prices, not actual market observations.")
else:
    upload = st.file_uploader("Daily adjusted closing price CSV (separate from document uploads)", type=["csv"])
    if upload:
        filename, data = upload.name, upload.getvalue()
    provenance = st.text_input("Data provenance / provider and adjustment method", placeholder="Provider, download date, split/dividend-adjusted closing prices")
    simulated = st.checkbox("This uploaded price series is simulated / synthetic")
rf_percent = st.number_input("Annual risk-free rate (%)", min_value=-99.0, max_value=1000.0, value=4.0, step=0.25)
risk_free_rate = rf_percent / 100
identity = fingerprint(data, filename, provenance, simulated, risk_free_rate)
sync_inputs(st.session_state, "portfolio_identity", identity, ("portfolio_result",))
if st.button("Optimize portfolios", disabled=not data or not provenance.strip(), type="primary"):
    st.session_state.pop("portfolio_result", None)
    try:
        with st.spinner("Validating observations, estimating risk, solving portfolios, and rendering charts…"):
            st.session_state["portfolio_result"] = analyze(data, filename, provenance, simulated, risk_free_rate)
    except FinSightError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Portfolio analysis could not complete. Check the input data, installed solvers, and local report/chart dependencies.")
if "portfolio_result" in st.session_state:
    result = st.session_state["portfolio_result"]
    st.success(f"Analysis complete: {len(result.data.prices)} price observations, {len(result.estimates.returns)} daily returns.")
    st.caption(f"{result.report['date_start']} to {result.report['date_end']} · Risk-free rate {risk_free_rate:.2%} · {result.data.provenance}")
    for strategy, message in result.failures.items():
        st.warning(f"{strategy} unavailable: {message}")
    columns = st.columns(len(result.portfolios))
    for column, portfolio in zip(columns, result.portfolios):
        with column:
            st.subheader(portfolio.strategy)
            st.metric("Estimated annual return", f"{portfolio.metrics['expected_return']:.2%}")
            st.metric("Annual volatility", f"{portfolio.metrics['volatility']:.2%}")
            st.metric("Sharpe ratio", f"{portfolio.metrics['sharpe_ratio']:.3f}")
            st.caption(f"{portfolio.solver} · {portfolio.status}")
    st.subheader("Allocation table")
    allocation = pd.DataFrame({p.strategy: p.weights for p in result.portfolios})
    st.dataframe(allocation.style.format("{:.2%}"), width="stretch")
    for filename, image in result.charts.items():
        st.image(image, caption=filename.replace("_", " ").removesuffix(".png"), width="stretch")
    st.subheader("Download research outputs")
    st.download_button("Allocation CSV", result.allocation, file_name="finsight_allocation.csv", mime="text/csv")
    st.download_button("Readable report (Markdown)", result.markdown, file_name="finsight_portfolio.md", mime="text/markdown")
    st.download_button("Structured report (JSON)", result.structured, file_name="finsight_portfolio.json", mime="application/json")
    for filename, image in result.charts.items():
        st.download_button(f"Chart image: {filename}", image, file_name=filename, mime="image/png")
    with st.expander("Calculation details and assumptions"):
        st.write("Daily simple returns; arithmetic historical expected returns × 252; sample covariance × 252. Both solvers use regularized covariance, including the diagonal ridge shown below.")
        st.caption(f"Covariance ridge: {result.estimates.ridge:.12g}")
        st.write("Maximum Sharpe minimizes yᵀΣy subject to (μ − r_f 1)ᵀy = 1 and y ≥ 0, then recovers w = y / sum(y). At least one positive estimated excess return is required.")
        for assumption in result.report["assumptions"]:
            st.write(assumption)
