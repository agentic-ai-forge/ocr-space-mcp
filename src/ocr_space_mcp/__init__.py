"""OCR.space MCP Server - Text extraction from images and PDFs."""

from .client import OCRSpaceClient, get_client
from .config import Settings, get_settings
from .models import LANGUAGES, OCREngine, OCRResponse, ParsedResult, Tier
from .server import main, server

__version__ = "0.1.0"

__all__ = [
    # Server
    "main",
    "server",
    # Client
    "OCRSpaceClient",
    "get_client",
    # Config
    "Settings",
    "get_settings",
    # Models
    "Tier",
    "OCREngine",
    "OCRResponse",
    "ParsedResult",
    "LANGUAGES",
]
