"""Application settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NovaVision"
    backend: str = "null"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    diffusion_model: str = "stabilityai/sd-turbo"
    hf_token: str | None = None
    default_style: str = "artistic"
    default_tier: str = "affect"
    width: int = 512
    height: int = 512


@lru_cache
def get_settings() -> Settings:
    return Settings()
