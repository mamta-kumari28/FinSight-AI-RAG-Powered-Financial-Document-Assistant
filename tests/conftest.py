"""Offline fixtures deliberately use a local deterministic test embedder, never a downloaded model."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption('--live', action='store_true', help='Opt in to real Gemini requests (quota/costs apply).')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--live'):
        for item in items:
            if 'live' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='Live Gemini checks require --live and secure credentials.'))


@pytest.fixture
def samples():
    return ROOT / 'data/samples'


class OfflineTestEmbeddings:
    """Hashed bag-of-words TEST DOUBLE; not Gemini, not used by the application."""
    signature = 'offline-test-embedder-v1'
    dimensions = 64

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text):
        vector = np.zeros(self.dimensions)
        for word in re.findall(r'\w+', text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimensions] += 1
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else np.ones(self.dimensions) / np.sqrt(self.dimensions)).tolist()

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vector(text)


@pytest.fixture
def embedder():
    return OfflineTestEmbeddings()
