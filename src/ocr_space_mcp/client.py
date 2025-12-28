"""Async HTTP client for the OCR.space API."""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import (
    FREE_ENDPOINT,
    FREE_SIZE_LIMIT,
    PRO_SIZE_LIMIT,
    Settings,
    get_settings,
)
from .models import OCRResponse, Tier, get_content_type


@dataclass
class OCROptions:
    """Options for OCR API calls."""

    language: str = "eng"
    ocr_engine: int = 1
    detect_orientation: bool = False
    scale: bool = False
    is_table: bool = False
    is_overlay_required: bool = False
    is_create_searchable_pdf: bool = False
    filetype: str | None = None


class OCRSpaceClient:
    """Async client for the OCR.space API."""

    def __init__(self, settings: Settings | None = None):
        """Initialize client with settings.

        Args:
            settings: Settings instance or None to load from environment.
        """
        self._settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    def _get_api_key(self, tier: Tier) -> str:
        """Get API key for the specified tier."""
        if tier == Tier.PRO:
            if not self._settings.pro_api_key:
                raise ValueError(
                    "OCR_SPACE_PRO_API_KEY not set. "
                    "Set it in your environment or use tier='free'."
                )
            return self._settings.pro_api_key.get_secret_value()
        if not self._settings.api_key:
            raise ValueError(
                "OCR_SPACE_API_KEY not set. "
                "Get a free key at https://ocr.space/ocrapi/freekey"
            )
        return self._settings.api_key.get_secret_value()

    def _get_endpoint(self, tier: Tier) -> str:
        """Get API endpoint for the specified tier."""
        return self._settings.pro_endpoint if tier == Tier.PRO else FREE_ENDPOINT

    def get_size_limit(self, tier: Tier) -> int:
        """Get file size limit for the specified tier."""
        return PRO_SIZE_LIMIT if tier == Tier.PRO else FREE_SIZE_LIMIT

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._settings.timeout)
        return self._client

    async def ocr(
        self,
        tier: Tier,
        *,
        file_path: str | None = None,
        url: str | None = None,
        base64_image: str | None = None,
        options: OCROptions | None = None,
    ) -> OCRResponse:
        """Call the OCR.space API.

        Args:
            tier: API tier (free or pro)
            file_path: Local file path to process
            url: URL of image/PDF to process
            base64_image: Base64-encoded image data
            options: OCR options (language, engine, etc.)

        Returns:
            OCRResponse with parsed results

        Raises:
            ValueError: If no input provided or file too large
            FileNotFoundError: If file_path doesn't exist
        """
        if options is None:
            options = OCROptions()

        api_key = self._get_api_key(tier)
        endpoint = self._get_endpoint(tier)
        size_limit = self.get_size_limit(tier)

        data = self._build_request_data(api_key=api_key, options=options)

        if file_path:
            self._add_file_to_request(data, file_path, size_limit, tier)
        elif url:
            data["url"] = url
        elif base64_image:
            data["base64Image"] = base64_image
        else:
            raise ValueError("Must provide file_path, url, or base64_image")

        client = await self._ensure_client()
        response = await client.post(endpoint, data=data)
        response.raise_for_status()
        return OCRResponse.model_validate(response.json())

    def _build_request_data(self, *, api_key: str, options: OCROptions) -> dict[str, Any]:
        """Build the API request form data."""
        data: dict[str, Any] = {
            "apikey": api_key,
            "language": options.language,
            "OCREngine": str(options.ocr_engine),
            "detectOrientation": str(options.detect_orientation).lower(),
            "scale": str(options.scale).lower(),
            "isTable": str(options.is_table).lower(),
            "isOverlayRequired": str(options.is_overlay_required).lower(),
            "isCreateSearchablePdf": str(options.is_create_searchable_pdf).lower(),
        }
        if options.filetype:
            data["filetype"] = options.filetype.upper()
        return data

    def _add_file_to_request(
        self,
        data: dict[str, Any],
        file_path: str,
        size_limit: int,
        tier: Tier,
    ) -> None:
        """Add file data to request, validating size."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size > size_limit:
            limit_mb = size_limit / (1024 * 1024)
            file_mb = file_size / (1024 * 1024)
            hint = "Use tier=pro for larger files." if tier == Tier.FREE else "File too large."
            raise ValueError(
                f"File size ({file_mb:.2f} MB) exceeds {tier.value} tier "
                f"limit ({limit_mb:.0f} MB). {hint}"
            )

        content_type = get_content_type(str(path))
        with open(path, "rb") as f:
            file_data = f.read()
        b64_data = base64.b64encode(file_data).decode("utf-8")
        data["base64Image"] = f"data:{content_type};base64,{b64_data}"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OCRSpaceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def get_client(settings: Settings | None = None) -> OCRSpaceClient:
    """Get a new OCR.space client instance."""
    return OCRSpaceClient(settings)
