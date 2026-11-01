"""Read current Gemini Developer API model metadata for the securely configured key."""
import json

from src.config import load_settings
from src.errors import FinSightError
from src.rag.embeddings import model_catalog, validate_models


def main():
    settings = load_settings()
    try:
        catalog = model_catalog(settings)
        print(json.dumps(catalog, indent=2))
        validate_models(settings)
        print('Configured models support the required actions. Run opt-in live tests to verify inference.')
    except FinSightError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
