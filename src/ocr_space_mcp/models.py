"""Data models and constants for OCR.space MCP Server."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Tier(str, Enum):
    """API tier selection."""

    FREE = "free"  # 1MB limit, US servers, free API key
    PRO = "pro"  # 5MB limit, configurable endpoint, PRO API key


class OCREngine(int, Enum):
    """OCR processing engine."""

    ENGINE_1 = 1  # Faster, Asian languages, larger images
    ENGINE_2 = 2  # Auto-language detection, better special chars


# Supported languages with display names
LANGUAGES: dict[str, str] = {
    "ara": "Arabic",
    "bul": "Bulgarian",
    "chs": "Chinese (Simplified)",
    "cht": "Chinese (Traditional)",
    "hrv": "Croatian",
    "cze": "Czech",
    "dan": "Danish",
    "dut": "Dutch",
    "eng": "English",
    "fin": "Finnish",
    "fre": "French",
    "ger": "German",
    "gre": "Greek",
    "hun": "Hungarian",
    "kor": "Korean",
    "ita": "Italian",
    "jpn": "Japanese",
    "pol": "Polish",
    "por": "Portuguese",
    "rus": "Russian",
    "slv": "Slovenian",
    "spa": "Spanish",
    "swe": "Swedish",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "vnm": "Vietnamese",
    "auto": "Auto-detect (Engine 2 only)",
}

# Content type mappings
CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

# Supported image extensions
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


class ParsedResult(BaseModel):
    """Single page OCR result from API."""

    parsed_text: str = Field(default="", alias="ParsedText")
    file_parse_exit_code: int = Field(default=0, alias="FileParseExitCode")
    error_message: str = Field(default="", alias="ErrorMessage")
    error_details: str = Field(default="", alias="ErrorDetails")

    model_config = {"populate_by_name": True}


class OCRResponse(BaseModel):
    """Complete OCR API response."""

    exit_code: int = Field(alias="OCRExitCode")
    parsed_results: list[ParsedResult] = Field(default_factory=list, alias="ParsedResults")
    is_errored: bool = Field(default=False, alias="IsErroredOnProcessing")
    processing_time_ms: str | None = Field(default=None, alias="ProcessingTimeInMilliseconds")
    searchable_pdf_url: str | None = Field(default=None, alias="SearchablePDFURL")
    error_message: str | list[str] | None = Field(default=None, alias="ErrorMessage")
    error_details: str | None = Field(default=None, alias="ErrorDetails")

    model_config = {"populate_by_name": True}

    @property
    def error_message_str(self) -> str | None:
        """Get error message as string (API returns list or string)."""
        if self.error_message is None:
            return None
        if isinstance(self.error_message, list):
            return "; ".join(self.error_message)
        return self.error_message


def get_content_type(file_path: str) -> str:
    """Determine content type from file extension."""
    ext = Path(file_path).suffix.lower()
    return CONTENT_TYPES.get(ext, "application/octet-stream")


def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF."""
    return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str) -> bool:
    """Check if file is a supported image."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
