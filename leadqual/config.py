"""Application configuration.

Settings come from environment variables, optionally loaded from a ``.env``
file in the project root. Nothing secret is ever hard-coded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without the package
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


# Project root = the directory containing the `leadqual` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Default Anthropic model. Haiku is the cheapest current Claude model - a
#: full run of the ten sample leads costs well under a cent. Set
#: LEADQUAL_MODEL=claude-opus-5-5 for the most capable model instead.
DEFAULT_MODEL = "claude-haiku-5-5"

#: Default Gemini model. Flash-Lite is the cheapest current Gemini model and
#: is free of charge on the free tier.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

#: Free-tier Gemini capacity is shared and a long request can 503 several
#: times in a row before it lands, so the retry budget is deliberately high.
DEFAULT_GEMINI_RETRIES = 10

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_DB_PATH = "data/leadqual.db"

#: Keys the application knows how to read, per provider.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable tolerantly."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of application configuration."""

    db_path: Path
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    #: Attempts (including the first) for retryable Gemini failures. The free
    #: tier returns intermittent 503s, so this defaults generously.
    gemini_retries: int = DEFAULT_GEMINI_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key: str | None = field(default=None, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)
    offline: bool = False
    allow_fallback: bool = True

    @staticmethod
    def _usable(key: str | None) -> bool:
        """True when ``key`` is set and is not the .env.example placeholder."""
        if not key:
            return False
        cleaned = key.strip()
        return bool(cleaned) and "xxxx" not in cleaned.lower()

    @property
    def has_api_key(self) -> bool:
        """True when a usable Anthropic API key is configured."""
        return self._usable(self.api_key)

    @property
    def has_gemini_key(self) -> bool:
        """True when a usable Gemini API key is configured."""
        return self._usable(self.gemini_api_key)

    @property
    def active_model(self) -> str:
        """The model id the selected provider will call."""
        return self.gemini_model if self.provider == "gemini" else self.model

    @property
    def has_active_key(self) -> bool:
        """True when the *selected* provider has a usable key."""
        return self.has_gemini_key if self.provider == "gemini" else self.has_api_key

    @property
    def live_calls_enabled(self) -> bool:
        """True when a real API call should be attempted."""
        return self.has_active_key and not self.offline

    def describe(self) -> dict[str, object]:
        """A log/UI-safe view of the settings (never includes a key)."""
        return {
            "db_path": str(self.db_path),
            "provider": self.provider,
            "active_model": self.active_model,
            "anthropic_model": self.model,
            "gemini_model": self.gemini_model,
            "max_tokens": self.max_tokens,
            "anthropic_key_configured": self.has_api_key,
            "gemini_key_configured": self.has_gemini_key,
            "offline": self.offline,
            "allow_fallback": self.allow_fallback,
            "live_calls_enabled": self.live_calls_enabled,
        }


def load_settings(env_file: str | Path | None = None, **overrides: object) -> Settings:
    """Build a :class:`Settings` object from the environment.

    Args:
        env_file: Optional explicit path to a ``.env`` file. When omitted the
            project-root ``.env`` is used if present.
        **overrides: Keyword overrides applied last (used by tests and the CLI).

    Returns:
        A populated, immutable :class:`Settings`.
    """
    load_dotenv(env_file or (PROJECT_ROOT / ".env"), override=False)

    raw_db = os.getenv("LEADQUAL_DB_PATH", DEFAULT_DB_PATH)
    db_path = Path(raw_db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # Provider selection: an explicit LEADQUAL_PROVIDER always wins. With no
    # explicit choice, pick whichever provider actually has a key, so a user
    # who only has a free Gemini key does not have to configure anything else.
    provider = (os.getenv("LEADQUAL_PROVIDER") or "").strip().lower()
    if not provider:
        if Settings._usable(anthropic_key):
            provider = "anthropic"
        elif Settings._usable(gemini_key):
            provider = "gemini"
        else:
            provider = DEFAULT_PROVIDER

    settings = Settings(
        db_path=db_path,
        provider=provider,
        model=os.getenv("LEADQUAL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        gemini_model=(
            os.getenv("LEADQUAL_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
            or DEFAULT_GEMINI_MODEL
        ),
        gemini_retries=_env_int("LEADQUAL_GEMINI_RETRIES", DEFAULT_GEMINI_RETRIES),
        max_tokens=_env_int("LEADQUAL_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        api_key=anthropic_key,
        gemini_api_key=gemini_key,
        offline=_env_bool("LEADQUAL_OFFLINE", False),
        allow_fallback=_env_bool("LEADQUAL_ALLOW_FALLBACK", True),
    )

    if overrides:
        # dataclasses.replace keeps the object frozen and validated.
        from dataclasses import replace

        settings = replace(settings, **overrides)  # type: ignore[arg-type]
    return settings
