from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ShofoLens"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/shofolens"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Qdrant
    QDRANT_HOST: str = "4594413c-b3c6-4a73-b706-69f0a7b5c73a.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "shofo_videos"
    USE_MEMORY_QDRANT: bool = False  # Use Qdrant Cloud in production

    # ML Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Dataset
    DATASET_NAME: str = "Shofo/shofo-tiktok-general-small"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
