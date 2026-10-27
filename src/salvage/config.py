"""Salvage provider configuration.

Providers are OpenAI-compatible HTTP endpoints. Models are grouped by the
provider that serves them, per the architecture document:

- meta API: SAM 3.1 (eyes), Muse Wonder (foreman), Muse Spark 1.3 (reader),
  Muse Voice Transcribe (ears)
- Baseten: Muse Spark Open (second opinion, open weights)
- Fireworks / Modal: Muse Glimmer 30B (bench agent, open weights)
- Anthropic: Claude (cross-examiner, uncorrelated errors)
- Free pool: Groq, NVIDIA NIM, Gemini, OpenRouter (cheap parallel votes)

Keys live in .env only, are read through this module, and are never logged,
returned, or committed. With no keys on disk every provider imports fine and
every model caller must fall back to a mocked/cached path.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Provider(BaseModel):
    """One OpenAI-compatible model provider."""

    name: str
    """Short human name, e.g. ``meta-api``."""

    base_url: str
    """OpenAI-compatible base URL, ends in ``/v1`` style paths."""

    api_key_env: str
    """The single env var that holds this provider's key."""

    model_id: str
    """The default model id to call on this provider."""

    enabled: bool = True
    """Turn a provider off without deleting it."""

    models: dict[str, str] = Field(default_factory=dict)
    """Extra role -> model id map for providers serving several models."""

    def api_key(self) -> str | None:
        """Return the key from the environment, or None. Never log it."""
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None

    def is_available(self) -> bool:
        return self.enabled and self.api_key() is not None


PROVIDERS: list[Provider] = [
    Provider(
        name="meta-api",
        base_url="https://api.meta.ai/v1",
        api_key_env="META_API_KEY",
        model_id="muse-spark-1.3",
        models={
            "vision": "sam-3.1",
            "foreman": "muse-wonder-1.0",
            "reader": "muse-spark-1.3",
            "voice": "muse-voice-transcribe",
        },
    ),
    Provider(
        name="baseten",
        base_url="https://inference.baseten.co/v1",
        api_key_env="BASETEN_API_KEY",
        model_id="meta-models/muse-spark-open-1026",
    ),
    Provider(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        model_id="accounts/fireworks/models/muse-glimmer-30b",
    ),
    Provider(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-5",
    ),
    Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_id="llama-3.3-70b-versatile",
    ),
    Provider(
        name="nvidia-nim",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_NIM_API_KEY",
        model_id="meta/llama-3.3-70b-instruct",
    ),
    Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        model_id="gemini-2.0-flash",
    ),
    Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="meta-llama/llama-3.3-70b-instruct:free",
    ),
]


def get_provider(name: str) -> Provider:
    """Look a provider up by name; raise KeyError if unknown."""
    for provider in PROVIDERS:
        if provider.name == name:
            return provider
    raise KeyError(f"unknown provider: {name}")


def available_providers() -> list[str]:
    """Names of providers that are enabled and have a key in .env/run env."""
    return [p.name for p in PROVIDERS if p.is_available()]
