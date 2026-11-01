"""Safe, actionable errors for application boundaries."""


class FinSightError(ValueError):
    """An expected error that is safe to display to the user."""


def provider_error(exc: Exception) -> FinSightError:
    """Classify a provider failure without reflecting URLs, headers, or key values."""
    text = str(exc).lower()
    if any(word in text for word in ("429", "quota", "resource_exhausted", "rate limit")):
        return FinSightError("Gemini quota or rate limit reached. Check billing/quota and try again later.")
    if any(word in text for word in ("401", "403", "api_key", "api key", "permission_denied", "unauthenticated")):
        return FinSightError("Gemini authentication or access failed. Check your secure API key and model permissions.")
    if any(word in text for word in ("404", "not_found", "not found", "not supported")):
        return FinSightError("The configured Gemini model is unavailable for this key/endpoint. Validate model names in Settings.")
    return FinSightError("Gemini could not complete the request. Check network access, model settings, and service status, then retry.")
