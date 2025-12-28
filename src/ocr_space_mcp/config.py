"""Configuration management using pydantic-settings."""

from functools import cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# API Endpoints
FREE_ENDPOINT = "https://api.ocr.space/parse/image"
DEFAULT_PRO_ENDPOINT = "https://apipro1.ocr.space/parse/image"

# File size limits (bytes)
FREE_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB
PRO_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB


class Settings(BaseSettings):
    """OCR.space MCP Server settings.

    Environment variables:
    - OCR_SPACE_API_KEY: Free tier API key (required for free tier)
    - OCR_SPACE_PRO_API_KEY: PRO tier API key (required for pro tier)
    - OCR_SPACE_PRO_ENDPOINT: Custom PRO endpoint (optional, defaults to US)
    """

    model_config = SettingsConfigDict(
        env_prefix="OCR_SPACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Free tier API key
    api_key: SecretStr | None = None

    # PRO tier
    pro_api_key: SecretStr | None = None
    pro_endpoint: str = DEFAULT_PRO_ENDPOINT

    # HTTP configuration
    timeout: float = 120.0


@cache
def get_settings() -> Settings:
    """Get settings instance.

    Uses functools.cache for efficiency - settings are immutable
    once loaded from environment variables.
    """
    return Settings()
