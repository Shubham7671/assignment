"""LLM provider adapters.

Two real integrations sit behind one interface:

* :class:`AnthropicProvider` — the Anthropic Messages API via the official
  ``anthropic`` SDK. This is the provider the assignment brief names, and the
  default when a key is configured.
* :class:`GeminiProvider` — the Google Gemini API via the official
  ``google-genai`` SDK, used because its Flash models have a genuinely free
  tier, so a live API call can be demonstrated without credits.

Each adapter takes the same prompts and the same JSON Schema and returns the
same :class:`ProviderResponse`, so everything above this layer — prompts,
schema validation, scoring, persistence — is provider-neutral.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..exceptions import LLMCallError, LLMConfigurationError, LLMResponseError
from .schema import QUALIFICATION_SCHEMA

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_FALLBACK = "offline-fallback"

#: Providers that can be selected with ``LEADQUAL_PROVIDER``.
SUPPORTED_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_GEMINI)

# --------------------------------------------------------------------------
# Optional SDK imports - neither is required to run the application
# --------------------------------------------------------------------------
try:
    import anthropic

    _ANTHROPIC_SDK = True
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_SDK = False

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    _GEMINI_SDK = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _GEMINI_SDK = False


@dataclass
class ProviderResponse:
    """A provider's reply, normalised across SDKs.

    Attributes:
        text: The raw response text (expected to be a JSON object).
        model: The model that actually served the request.
        response_id: Provider-side identifier, for support and audit.
        stop_reason: Why generation stopped, in the provider's vocabulary.
        input_tokens / output_tokens: Usage, when the provider reports it.
        latency_ms: Wall-clock time for the call.
    """

    text: str
    model: str
    response_id: str | None = None
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


# ==========================================================================
# Base
# ==========================================================================
class BaseProvider(ABC):
    """Interface every provider adapter implements."""

    #: Stored in ``qualification_reports.provider``.
    name: str = "unknown"

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._injected_client = client
        self._client: Any | None = client

    # -- capability -----------------------------------------------------
    @property
    @abstractmethod
    def model(self) -> str:
        """The model id this provider will call."""

    @property
    @abstractmethod
    def sdk_installed(self) -> bool:
        """Whether the provider's SDK is importable."""

    @property
    @abstractmethod
    def api_key_configured(self) -> bool:
        """Whether a usable (non-placeholder) key is configured."""

    @property
    def available(self) -> bool:
        """True when a live call can be attempted."""
        if self._injected_client is not None:
            return True
        return self.sdk_installed and self.api_key_configured and not self.settings.offline

    @property
    def status_reason(self) -> str:
        """A human-readable explanation of :attr:`available`."""
        if self._injected_client is not None:
            return "Using an injected client (test mode)."
        if not self.sdk_installed:
            return f"The SDK for '{self.name}' is not installed."
        if not self.api_key_configured:
            return (
                f"No API key found for '{self.name}' - set one in .env "
                "to enable live calls."
            )
        if self.settings.offline:
            return "LEADQUAL_OFFLINE is set, so live calls are disabled."
        return f"Ready to call the {self.name} API ({self.model})."

    def status(self) -> dict[str, Any]:
        """Describe this provider for the CLI and UI."""
        return {
            "provider": self.name,
            "sdk_installed": self.sdk_installed,
            "api_key_configured": self.api_key_configured,
            "offline_mode": self.settings.offline,
            "fallback_allowed": self.settings.allow_fallback,
            "model": self.model,
            "live_calls_enabled": self.available,
            "reason": self.status_reason,
        }

    # -- the call -------------------------------------------------------
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        """Send one request and return the normalised response.

        Raises:
            LLMConfigurationError: Credentials or model are wrong.
            LLMCallError: The call itself failed (network, rate limit, 5xx).
            LLMResponseError: The provider replied but the reply is unusable.
        """


# ==========================================================================
# Anthropic
# ==========================================================================
class AnthropicProvider(BaseProvider):
    """Anthropic Messages API via the official ``anthropic`` SDK."""

    name = PROVIDER_ANTHROPIC

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def sdk_installed(self) -> bool:
        return _ANTHROPIC_SDK

    @property
    def api_key_configured(self) -> bool:
        return self.settings.has_api_key

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not _ANTHROPIC_SDK:
            raise LLMConfigurationError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            )
        if not self.settings.has_api_key:
            raise LLMConfigurationError(
                "No ANTHROPIC_API_KEY configured. Copy .env.example to .env and add your key."
            )
        self._client = anthropic.Anthropic(api_key=self.settings.api_key, max_retries=2)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        client = self._get_client()

        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            # The system prompt is byte-identical on every call, so caching it
            # makes repeated runs cheaper.
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_prompt}],
            # Constrain the reply to the agreed contract.
            "output_config": {
                "format": {"type": "json_schema", "schema": QUALIFICATION_SCHEMA}
            },
        }

        started = time.perf_counter()
        response = self._call_api(client, request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMResponseError(
                f"The model declined this request (category: {category or 'unspecified'})."
            )
        if stop_reason == "max_tokens":
            raise LLMResponseError(
                "The response hit the max_tokens limit and is incomplete. "
                "Raise LEADQUAL_MAX_TOKENS and retry."
            )

        usage = getattr(response, "usage", None)
        parts = [
            getattr(b, "text", "") or ""
            for b in (getattr(response, "content", None) or [])
            if getattr(b, "type", None) == "text"
        ]

        return ProviderResponse(
            text="\n".join(p for p in parts if p).strip(),
            model=getattr(response, "model", self.settings.model),
            response_id=getattr(response, "id", None),
            stop_reason=stop_reason,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    def _call_api(self, client: Any, request: dict[str, Any]) -> Any:
        """Send the request, retrying once without ``output_config`` if the
        account or model does not accept structured outputs."""
        try:
            return client.messages.create(**request)
        except Exception as exc:
            if self._is_structured_output_rejection(exc) and "output_config" in request:
                retry = {k: v for k, v in request.items() if k != "output_config"}
                retry["messages"] = list(retry["messages"]) + [
                    {
                        "role": "user",
                        "content": "Reply with the JSON object only - no prose, no code fence.",
                    }
                ]
                try:
                    return client.messages.create(**retry)
                except Exception as retry_exc:
                    raise self._map_exception(retry_exc) from retry_exc
            raise self._map_exception(exc) from exc

    @staticmethod
    def _is_structured_output_rejection(exc: Exception) -> bool:
        if not _ANTHROPIC_SDK or not isinstance(exc, anthropic.BadRequestError):
            return False
        message = str(exc).lower()
        return any(
            token in message
            for token in ("output_config", "output format", "json_schema", "structured output")
        )

    @staticmethod
    def _map_exception(exc: Exception) -> Exception:
        """Translate an SDK exception, most-specific first."""
        if not _ANTHROPIC_SDK:  # pragma: no cover
            return LLMCallError(f"LLM call failed: {exc}")

        if isinstance(exc, anthropic.AuthenticationError):
            return LLMConfigurationError(
                "The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY in .env."
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return LLMConfigurationError(
                "The API key does not have permission for this model. "
                "Check the key's workspace and model access."
            )
        if isinstance(exc, anthropic.NotFoundError):
            return LLMConfigurationError(
                "Model not found. Check LEADQUAL_MODEL is a current model id."
            )
        if isinstance(exc, anthropic.RateLimitError):
            retry_after = "unknown"
            response = getattr(exc, "response", None)
            if response is not None:
                retry_after = response.headers.get("retry-after", "unknown")
            return LLMCallError(
                f"Rate limited by the Anthropic API (retry after {retry_after}s).",
                status_code=429,
            )
        if isinstance(exc, anthropic.BadRequestError):
            return LLMCallError(f"The API rejected the request: {exc}", status_code=400)
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and status >= 500:
                return LLMCallError(
                    f"Anthropic server error ({status}). Try again shortly.",
                    status_code=status,
                )
            return LLMCallError(f"Anthropic API error ({status}): {exc}", status_code=status)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMCallError(
                "Could not reach the Anthropic API. Check the network connection."
            )
        if isinstance(exc, (LLMCallError, LLMResponseError, LLMConfigurationError)):
            return exc
        return LLMCallError(f"Unexpected error calling the Anthropic API: {exc}")


# ==========================================================================
# Google Gemini
# ==========================================================================
#: Finish reasons that mean the model declined rather than answered.
_GEMINI_BLOCKED = {
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"
}


class GeminiProvider(BaseProvider):
    """Google Gemini API via the official ``google-genai`` SDK.

    Uses the stable ``client.models.generate_content`` surface rather than
    the newer Interactions API, which the installed SDK still marks
    experimental.

    Structured output is requested with ``response_json_schema``, which takes
    the same JSON Schema object sent to Anthropic. Gemini's constrained
    decoder enforces types, ``required``, ``enum``, ``additionalProperties``
    and the array bounds, but silently ignores ``minLength`` / ``maxLength``
    — which is exactly why :func:`leadqual.llm.schema.validate_payload`
    re-checks and clamps lengths in Python afterwards.
    """

    name = PROVIDER_GEMINI

    @property
    def model(self) -> str:
        return self.settings.gemini_model

    @property
    def sdk_installed(self) -> bool:
        return _GEMINI_SDK

    @property
    def api_key_configured(self) -> bool:
        return self.settings.has_gemini_key

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not _GEMINI_SDK:
            raise LLMConfigurationError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            )
        if not self.settings.has_gemini_key:
            raise LLMConfigurationError(
                "No GEMINI_API_KEY configured. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        # The free tier is capacity-shared and returns intermittent 503s
        # ("this model is currently experiencing high demand") even for a
        # trivial request. The SDK's own retry handles the backoff, so a
        # transient blip does not become a failed qualification run.
        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(
                    attempts=self.settings.gemini_retries,
                    initial_delay=1.0,
                    max_delay=30.0,
                    exp_base=1.6,
                    http_status_codes=[429, 500, 502, 503, 504],
                ),
                timeout=120_000,  # milliseconds
            ),
        )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        client = self._get_client()

        config: Any
        if _GEMINI_SDK and genai_types is not None:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.settings.max_tokens,
                response_mime_type="application/json",
                response_json_schema=QUALIFICATION_SCHEMA,
            )
        else:  # pragma: no cover - only when a test injects a client
            config = {
                "system_instruction": system_prompt,
                "max_output_tokens": self.settings.max_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": QUALIFICATION_SCHEMA,
            }

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        finish_reason = self._finish_reason(response)
        if finish_reason in _GEMINI_BLOCKED:
            raise LLMResponseError(
                f"Gemini declined this request (finish reason: {finish_reason})."
            )
        if finish_reason == "MAX_TOKENS":
            raise LLMResponseError(
                "The response hit the max_output_tokens limit and is incomplete. "
                "Raise LEADQUAL_MAX_TOKENS and retry."
            )

        text = getattr(response, "text", None) or ""
        usage = getattr(response, "usage_metadata", None)

        return ProviderResponse(
            text=text.strip(),
            model=getattr(response, "model_version", None) or self.settings.gemini_model,
            response_id=getattr(response, "response_id", None),
            stop_reason=finish_reason,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        """Pull the first candidate's finish reason out as a plain string."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        raw = getattr(candidates[0], "finish_reason", None)
        if raw is None:
            return None
        # The SDK returns an enum; .name is the stable identifier.
        return getattr(raw, "name", None) or str(raw)

    @staticmethod
    def _map_exception(exc: Exception) -> Exception:
        """Translate a google-genai exception, most-specific first."""
        if isinstance(exc, (LLMCallError, LLMResponseError, LLMConfigurationError)):
            return exc
        if not _GEMINI_SDK or genai_errors is None:  # pragma: no cover
            return LLMCallError(f"Gemini call failed: {exc}")

        if isinstance(exc, genai_errors.ClientError):
            code = getattr(exc, "code", None)
            if code in (401, 403):
                return LLMConfigurationError(
                    "Gemini rejected the credentials. Check GEMINI_API_KEY in .env "
                    f"({exc})."
                )
            if code == 404:
                return LLMConfigurationError(
                    f"Gemini model not found: {exc}. Check LEADQUAL_GEMINI_MODEL is a "
                    "current model id - see https://aistudio.google.com/docs/pricing "
                    "for the models on the free tier."
                )
            if code == 429:
                return LLMCallError(
                    "Gemini free-tier rate limit reached. Wait a minute and retry, "
                    f"or use a smaller model ({exc}).",
                    status_code=429,
                )
            return LLMCallError(f"Gemini rejected the request: {exc}", status_code=code)

        if isinstance(exc, genai_errors.ServerError):
            code = getattr(exc, "code", None)
            return LLMCallError(
                f"Gemini server error ({code}). Try again shortly.", status_code=code
            )
        if isinstance(exc, genai_errors.APIError):
            return LLMCallError(f"Gemini API error: {exc}", status_code=getattr(exc, "code", None))
        return LLMCallError(f"Unexpected error calling the Gemini API: {exc}")


# ==========================================================================
# Selection
# ==========================================================================
PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    PROVIDER_ANTHROPIC: AnthropicProvider,
    PROVIDER_GEMINI: GeminiProvider,
}


def build_provider(settings: Settings, *, client: Any | None = None) -> BaseProvider:
    """Instantiate the provider named by ``settings.provider``.

    Args:
        settings: Application settings.
        client: Optional injected SDK client (tests pass a mock here).

    Returns:
        A provider adapter.

    Raises:
        LLMConfigurationError: If the configured provider name is unknown.
    """
    provider_class = PROVIDER_CLASSES.get(settings.provider)
    if provider_class is None:
        raise LLMConfigurationError(
            f"Unknown LEADQUAL_PROVIDER {settings.provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return provider_class(settings, client=client)


def provider_availability(settings: Settings) -> list[dict[str, Any]]:
    """Status of every supported provider - used by ``leadqual status``."""
    return [
        provider_class(settings).status()
        for provider_class in PROVIDER_CLASSES.values()
    ]
