"""Exception hierarchy for the Lead Qualification Assistant.

A single base class (:class:`LeadQualError`) lets callers catch everything
this application raises, while the subclasses let them react to specific
failure modes (bad input vs. missing record vs. provider outage).
"""

from __future__ import annotations


class LeadQualError(Exception):
    """Base class for every error raised by this application."""


# --------------------------------------------------------------------------
# Input / validation
# --------------------------------------------------------------------------
class ValidationError(LeadQualError):
    """Raised when user-supplied input fails validation.

    Attributes:
        field: Name of the offending field, when known.
        value: The rejected value, when safe to echo back.
    """

    def __init__(self, message: str, field: str | None = None, value: object = None) -> None:
        self.field = field
        self.value = value
        if field:
            message = f"{field}: {message}"
        super().__init__(message)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
class DatabaseError(LeadQualError):
    """Raised when a SQLite operation fails."""


class RecordNotFoundError(DatabaseError):
    """Raised when a lookup by primary key returns nothing."""

    def __init__(self, entity: str, key: object) -> None:
        self.entity = entity
        self.key = key
        super().__init__(f"No {entity} found with id {key!r}.")


# --------------------------------------------------------------------------
# NLP
# --------------------------------------------------------------------------
class NLPError(LeadQualError):
    """Raised when text extraction cannot be completed."""


# --------------------------------------------------------------------------
# LLM provider
# --------------------------------------------------------------------------
class LLMError(LeadQualError):
    """Base class for LLM provider problems."""


class LLMConfigurationError(LLMError):
    """Raised when the provider is not configured (e.g. no API key)."""


class LLMResponseError(LLMError):
    """Raised when the provider replied but the payload failed validation."""

    def __init__(self, message: str, raw: str | None = None) -> None:
        self.raw = raw
        super().__init__(message)


class LLMCallError(LLMError):
    """Raised when the API call itself failed (network, auth, rate limit)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class PipelineError(LeadQualError):
    """Raised when the end-to-end qualification workflow cannot complete."""
