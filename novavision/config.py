"""Application settings and pinned model revisions."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default checkpoints and the exact HF commits used for the paper; pin so a
# rerun pulls the same weights. Each pin belongs to its default model only:
# a swapped model loads unpinned (a foreign commit would not exist on it).
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
DIFFUSION_MODEL = "stabilityai/sd-turbo"
DIFFUSION_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class Settings(BaseSettings):
    # NOVA_ prefix keeps these consistent with the other NOVA_* env vars and stops
    # a generic BACKEND in a shell or base image from silently overriding the run.
    model_config = SettingsConfigDict(env_prefix="NOVA_", extra="ignore")

    backend: str = "null"
    emotion_model: str = EMOTION_MODEL
    diffusion_model: str = DIFFUSION_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()
