"""Read configuration without logging or persisting credentials."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 1_000_000
MAX_TABLE_ROWS = 20_000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    chat_model: str = "gemini-flash-latest"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    storage_root: Path = ROOT / "data" / "chroma"

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip() and self.chat_model.strip() and self.embedding_model.strip())


def load_settings(secrets: Mapping | None = None) -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    secrets = secrets or {}

    def get(name: str, default: str = "") -> str:
        return str(os.getenv(name) or secrets.get(name) or default).strip()

    return Settings(
        api_key=get("GEMINI_API_KEY"),
        chat_model=get("GEMINI_CHAT_MODEL", "gemini-flash-latest"),
        embedding_model=get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
