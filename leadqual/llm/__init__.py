"""LLM integration: prompt engineering, Claude API calls and validation."""

from .schema import QUALIFICATION_SCHEMA, validate_payload
from .prompts import build_system_prompt, build_user_prompt
from .providers import (
    PROVIDER_ANTHROPIC,
    PROVIDER_FALLBACK,
    PROVIDER_GEMINI,
    SUPPORTED_PROVIDERS,
    AnthropicProvider,
    GeminiProvider,
    build_provider,
    provider_availability,
)
from .client import QualificationLLM, generate_qualification

__all__ = [
    "QUALIFICATION_SCHEMA",
    "validate_payload",
    "build_system_prompt",
    "build_user_prompt",
    "QualificationLLM",
    "generate_qualification",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GEMINI",
    "PROVIDER_FALLBACK",
    "SUPPORTED_PROVIDERS",
    "AnthropicProvider",
    "GeminiProvider",
    "build_provider",
    "provider_availability",
]
