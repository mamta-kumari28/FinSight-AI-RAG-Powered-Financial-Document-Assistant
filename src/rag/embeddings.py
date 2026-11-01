import hashlib
import json

import numpy as np
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import Settings
from src.errors import FinSightError, provider_error


def model_catalog(settings: Settings) -> dict[str, list[str]]:
    """Ask the actual Gemini Developer API; metadata is not a successful inference."""
    if not settings.api_key:
        raise FinSightError("Configure GEMINI_API_KEY securely before contacting Gemini.")
    try:
        with genai.Client(api_key=settings.api_key, vertexai=False, http_options=types.HttpOptions(api_version="v1beta", timeout=30_000)) as client:
            return {
                model.name.removeprefix("models/"): list(model.supported_actions or [])
                for model in client.models.list()
                if model.name
            }
    except Exception as exc:
        raise provider_error(exc) from None


def validate_models(settings: Settings) -> dict[str, list[str]]:
    catalog = model_catalog(settings)
    for name, action in ((settings.chat_model, "generateContent"), (settings.embedding_model, "embedContent")):
        normalized = name.removeprefix("models/")
        if normalized not in catalog or action not in catalog[normalized]:
            raise FinSightError(f"Configured model {normalized!r} is not listed with {action} support for this key. Select an available model in Settings.")
    return catalog


class GeminiEmbeddings:
    """Paired retrieval tasks, identical model/dimensions, explicitly normalized vectors."""

    def __init__(self, settings: Settings):
        if not settings.configured:
            raise FinSightError("Configure the Gemini key and model names before indexing.")
        self.dimensions = settings.embedding_dimensions
        spec = {
            "provider": "gemini-developer-v1beta", "model": settings.embedding_model.removeprefix("models/"),
            "dimensions": self.dimensions, "tasks": ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
            "normalization": "l2", "version": 1,
        }
        self.signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        self.client = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model, google_api_key=settings.api_key,
            vertexai=False, output_dimensionality=self.dimensions,
            client_args={"http_options": {"api_version": "v1beta", "timeout": 30_000}},
        )

    def _normalize(self, values) -> list[list[float]]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.dimensions or not np.isfinite(array).all():
            raise FinSightError("Embedding dimensions or values changed. Remove the affected index and re-index with a compatible model.")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise FinSightError("Gemini returned an empty embedding. Retry indexing.")
        return (array / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._normalize(self.client.embed_documents(texts, batch_size=32, task_type="RETRIEVAL_DOCUMENT"))
        except FinSightError:
            raise
        except Exception as exc:
            raise provider_error(exc) from None

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._normalize([self.client.embed_query(text, task_type="RETRIEVAL_QUERY")])[0]
        except FinSightError:
            raise
        except Exception as exc:
            raise provider_error(exc) from None
