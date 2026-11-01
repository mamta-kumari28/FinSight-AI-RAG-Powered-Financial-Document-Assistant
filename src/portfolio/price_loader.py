from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

from src.errors import FinSightError
from src.ingestion.validation import read_csv_text, validate_upload


@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame
    filename: str
    provenance: str
    simulated: bool


def load_prices(data: bytes, filename: str, provenance: str, simulated: bool = False, min_observations: int = 61) -> PriceData:
    filename, _ = validate_upload(data, filename, {".csv"})
    text, headers = read_csv_text(data)
    if len(headers) < 3 or headers[0] != "Date":
        raise FinSightError("Price CSV must start with Date followed by at least two distinct asset-price columns.")
    if min_observations < 3:
        raise FinSightError("At least three observations are needed to estimate covariance.")
    try:
        frame = pd.read_csv(StringIO(text), keep_default_na=False)
        frame.columns = headers
        if len(frame) < min_observations:
            raise FinSightError(f"Provide at least {min_observations} daily price observations (rows).")
        if len(frame) > 100_000 or len(headers) > 101:
            raise FinSightError("Limit price uploads to 100,000 rows and 100 assets.")
        date_strings = frame.pop("Date").astype(str)
        if not date_strings.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise FinSightError("Dates must be valid daily ISO dates in YYYY-MM-DD format.")
        dates = pd.to_datetime(date_strings, format="%Y-%m-%d", errors="raise")
        if dates.duplicated().any():
            raise FinSightError("Duplicate dates are not allowed.")
        if not dates.is_monotonic_increasing:
            raise FinSightError("Dates must be in strictly increasing chronological order. Sort the source CSV first.")
        prices = frame.apply(pd.to_numeric, errors="raise").astype(float)
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
            raise FinSightError("Every price must be finite, positive, and present; missing/infinite/nonpositive prices are not accepted.")
        prices.index = pd.DatetimeIndex(dates, name="Date")
        if not provenance.strip():
            raise FinSightError("Provide the data source/provenance for the report.")
        return PriceData(prices, filename, provenance.strip(), bool(simulated))
    except FinSightError:
        raise
    except Exception as exc:
        raise FinSightError("Unreadable price data. Check invalid dates, missing cells, and nonnumeric prices; no rows have been silently dropped.") from exc
