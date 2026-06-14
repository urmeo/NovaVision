"""Application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    backend: str = "null"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    diffusion_model: str = "stabilityai/sd-turbo"


@lru_cache
def get_settings() -> Settings:
    return Settings()
