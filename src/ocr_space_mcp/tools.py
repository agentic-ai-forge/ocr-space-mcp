"""MCP Tool definitions for OCR.space API."""

from mcp.types import Tool

# Common schema fragments
TIER_SCHEMA = {
    "type": "string",
    "enum": ["free", "pro"],
    "default": "free",
    "description": "API tier: 'free' (1MB limit, US servers) or 'pro' (5MB limit, EU/GDPR)",
}

LANGUAGE_SCHEMA = {
    "type": "string",
    "default": "eng",
    "description": "OCR language code (e.g., 'eng', 'ger', 'fre'). Use 'auto' with Engine 2.",
}

OCR_ENGINE_SCHEMA = {
    "type": "integer",
    "enum": [1, 2],
    "default": 1,
    "description": "Engine 1: Faster, Asian languages. Engine 2: Auto-detect, better accuracy.",
}

OUTPUT_PATH_SCHEMA = {
    "type": "string",
    "description": "Optional path to save extracted text (.txt) or full result (.json)",
}

OUTPUT_FORMAT_SCHEMA = {
    "type": "string",
    "enum": ["txt", "json"],
    "default": "txt",
    "description": "Output format: 'txt' (text only) or 'json' (full API response)",
}


def get_tools() -> list[Tool]:
    """Get all available MCP tools."""
    return [
        _ocr_file_tool(),
        _ocr_url_tool(),
        _ocr_auto_tool(),
        _split_pdf_tool(),
        _list_languages_tool(),
        _check_tier_status_tool(),
    ]


def _ocr_file_tool() -> Tool:
    """Tool definition for ocr_file."""
    return Tool(
        name="ocr_file",
        description=(
            "Extract text from a local image or PDF file using OCR.space API. "
            "Supports PNG, JPG, GIF, BMP, TIFF, PDF. "
            "Free tier: max 1 MB. PRO tier: max 5 MB (EU endpoint for GDPR). "
            "Optionally saves result to txt or json file."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the image or PDF file to OCR",
                },
                "tier": TIER_SCHEMA,
                "language": LANGUAGE_SCHEMA,
                "ocr_engine": OCR_ENGINE_SCHEMA,
                "output_path": OUTPUT_PATH_SCHEMA,
                "output_format": OUTPUT_FORMAT_SCHEMA,
                "detect_orientation": {
                    "type": "boolean",
                    "default": False,
                    "description": "Auto-rotate image based on detected text orientation",
                },
                "scale": {
                    "type": "boolean",
                    "default": False,
                    "description": "Upscale low-resolution images for better OCR",
                },
                "is_table": {
                    "type": "boolean",
                    "default": False,
                    "description": "Optimize for table-like structures",
                },
                "is_overlay_required": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include word coordinates/bounding boxes in response",
                },
                "is_create_searchable_pdf": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate a searchable PDF (URL valid for 1 hour)",
                },
            },
            "required": ["file_path"],
        },
    )


def _ocr_url_tool() -> Tool:
    """Tool definition for ocr_url."""
    return Tool(
        name="ocr_url",
        description=(
            "Extract text from an image or PDF at a URL using OCR.space API. "
            "The URL must be publicly accessible. "
            "Free tier: max 1 MB. PRO tier: max 5 MB."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Public URL of the image or PDF to OCR",
                },
                "tier": TIER_SCHEMA,
                "language": LANGUAGE_SCHEMA,
                "ocr_engine": OCR_ENGINE_SCHEMA,
                "output_path": OUTPUT_PATH_SCHEMA,
                "output_format": OUTPUT_FORMAT_SCHEMA,
                "detect_orientation": {
                    "type": "boolean",
                    "default": False,
                    "description": "Auto-rotate image",
                },
                "scale": {
                    "type": "boolean",
                    "default": False,
                    "description": "Upscale low-resolution images",
                },
                "is_table": {
                    "type": "boolean",
                    "default": False,
                    "description": "Optimize for tables",
                },
            },
            "required": ["url"],
        },
    )


def _ocr_auto_tool() -> Tool:
    """Tool definition for ocr_auto."""
    return Tool(
        name="ocr_auto",
        description=(
            "Smart OCR that automatically handles oversized files. "
            "For PDFs: splits into chunks, OCRs each, and joins results. "
            "For images: compresses to fit within size limit before OCR. "
            "Use this when you don't know if the file exceeds the size limit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the image or PDF file to OCR",
                },
                "tier": TIER_SCHEMA,
                "language": LANGUAGE_SCHEMA,
                "ocr_engine": OCR_ENGINE_SCHEMA,
                "output_path": OUTPUT_PATH_SCHEMA,
                "output_format": OUTPUT_FORMAT_SCHEMA,
            },
            "required": ["file_path"],
        },
    )


def _split_pdf_tool() -> Tool:
    """Tool definition for split_pdf."""
    return Tool(
        name="split_pdf",
        description=(
            "Split a large PDF into smaller chunks that fit within API size limits. "
            "Useful for processing PDFs that exceed the 1 MB (free) or 5 MB (pro) limit. "
            "Each chunk contains as many pages as possible while staying under the limit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to split",
                },
                "tier": {
                    "type": "string",
                    "enum": ["free", "pro"],
                    "default": "free",
                    "description": "Target tier determines size limit: free=1MB, pro=5MB",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory for output chunks (uses temp dir if not specified)",
                },
            },
            "required": ["file_path"],
        },
    )


def _list_languages_tool() -> Tool:
    """Tool definition for list_languages."""
    return Tool(
        name="list_languages",
        description="List all supported OCR languages with their codes.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )


def _check_tier_status_tool() -> Tool:
    """Tool definition for check_tier_status."""
    return Tool(
        name="check_tier_status",
        description=(
            "Check which API tiers are configured and available. "
            "Shows which environment variables are set."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )
