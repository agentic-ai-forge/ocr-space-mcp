"""OCR.space MCP Server implementation."""

import base64
import io
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from PIL import Image
from pypdf import PdfReader, PdfWriter

# API Endpoints
FREE_ENDPOINT = "https://api.ocr.space/parse/image"
DEFAULT_PRO_ENDPOINT = "https://apipro1.ocr.space/parse/image"  # US PRO default

# File size limits (bytes)
FREE_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB
PRO_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB


class Tier(str, Enum):
    """API tier selection."""

    FREE = "free"
    PRO = "pro"


class OCREngine(int, Enum):
    """OCR processing engine."""

    ENGINE_1 = 1  # Faster, Asian languages, larger images
    ENGINE_2 = 2  # Auto-language detection, better special chars


# Supported languages with display names
LANGUAGES = {
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


def get_api_key(tier: Tier) -> str:
    """Get API key for the specified tier."""
    if tier == Tier.PRO:
        key = os.environ.get("OCR_SPACE_PRO_API_KEY")
        if not key:
            raise ValueError(
                "OCR_SPACE_PRO_API_KEY not set. Set it in your environment or use tier='free'."
            )
        return key
    else:
        key = os.environ.get("OCR_SPACE_API_KEY")
        if not key:
            raise ValueError(
                "OCR_SPACE_API_KEY not set. Get a free key at https://ocr.space/ocrapi/freekey"
            )
        return key


def get_endpoint(tier: Tier) -> str:
    """Get API endpoint for the specified tier."""
    if tier == Tier.PRO:
        return os.environ.get("OCR_SPACE_PRO_ENDPOINT", DEFAULT_PRO_ENDPOINT)
    return FREE_ENDPOINT


def get_size_limit(tier: Tier) -> int:
    """Get file size limit for the specified tier."""
    return PRO_SIZE_LIMIT if tier == Tier.PRO else FREE_SIZE_LIMIT


def get_content_type(file_path: str) -> str:
    """Determine content type from file extension."""
    ext = Path(file_path).suffix.lower()
    content_types = {
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
    return content_types.get(ext, "application/octet-stream")


async def call_ocr_api(
    tier: Tier,
    *,
    file_path: str | None = None,
    url: str | None = None,
    base64_image: str | None = None,
    language: str = "eng",
    ocr_engine: int = 1,
    detect_orientation: bool = False,
    scale: bool = False,
    is_table: bool = False,
    is_overlay_required: bool = False,
    is_create_searchable_pdf: bool = False,
    is_searchable_pdf_hide_text_layer: bool = False,
    filetype: str | None = None,
) -> dict[str, Any]:
    """Call the OCR.space API."""
    api_key = get_api_key(tier)
    endpoint = get_endpoint(tier)
    size_limit = get_size_limit(tier)

    # Build form data
    data: dict[str, Any] = {
        "apikey": api_key,
        "language": language,
        "OCREngine": str(ocr_engine),
        "detectOrientation": str(detect_orientation).lower(),
        "scale": str(scale).lower(),
        "isTable": str(is_table).lower(),
        "isOverlayRequired": str(is_overlay_required).lower(),
        "isCreateSearchablePdf": str(is_create_searchable_pdf).lower(),
        "isSearchablePdfHideTextLayer": str(is_searchable_pdf_hide_text_layer).lower(),
    }

    if filetype:
        data["filetype"] = filetype.upper()

    files = None

    if file_path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size > size_limit:
            limit_mb = size_limit / (1024 * 1024)
            file_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"File size ({file_mb:.2f} MB) exceeds {tier.value} tier limit ({limit_mb:.0f} MB). "
                f"{'Use tier=pro for larger files.' if tier == Tier.FREE else 'File too large for API.'}"
            )

        # Read and encode as base64 for reliability
        content_type = get_content_type(str(path))
        with open(path, "rb") as f:
            file_data = f.read()
        b64_data = base64.b64encode(file_data).decode("utf-8")
        data["base64Image"] = f"data:{content_type};base64,{b64_data}"

    elif url:
        data["url"] = url

    elif base64_image:
        data["base64Image"] = base64_image

    else:
        raise ValueError("Must provide file_path, url, or base64_image")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, data=data, files=files)
        response.raise_for_status()
        return response.json()


def format_ocr_result(result: dict[str, Any]) -> str:
    """Format OCR result for display."""
    lines = []

    exit_code = result.get("OCRExitCode", 0)
    if exit_code == 1:
        lines.append("Status: Success")
    elif exit_code == 2:
        lines.append("Status: Partial success (some pages failed)")
    elif exit_code == 3:
        lines.append("Status: Failed")
    elif exit_code == 4:
        lines.append("Status: Error")

    if "ProcessingTimeInMilliseconds" in result:
        lines.append(f"Processing time: {result['ProcessingTimeInMilliseconds']}ms")

    if "SearchablePDFURL" in result:
        lines.append(f"Searchable PDF: {result['SearchablePDFURL']}")

    parsed_results = result.get("ParsedResults", [])
    if parsed_results:
        lines.append(f"Pages processed: {len(parsed_results)}")
        lines.append("")
        lines.append("--- Extracted Text ---")

        for i, page in enumerate(parsed_results):
            if len(parsed_results) > 1:
                lines.append(f"\n[Page {i + 1}]")
            text = page.get("ParsedText", "").strip()
            if text:
                lines.append(text)
            else:
                lines.append("(No text extracted)")

            if page.get("ErrorMessage"):
                lines.append(f"Error: {page['ErrorMessage']}")

    if result.get("ErrorMessage"):
        lines.append(f"\nAPI Error: {result['ErrorMessage']}")

    if result.get("ErrorDetails"):
        lines.append(f"Details: {result['ErrorDetails']}")

    return "\n".join(lines)


def extract_text_only(result: dict[str, Any]) -> str:
    """Extract only the parsed text from OCR result."""
    texts = []
    for page in result.get("ParsedResults", []):
        text = page.get("ParsedText", "").strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


async def save_result(
    result: dict[str, Any],
    output_path: str,
    format: str = "txt",
) -> str:
    """Save OCR result to file."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:  # txt
        text = extract_text_only(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    return str(path)


def split_pdf_by_size(
    pdf_path: str,
    max_size_bytes: int,
    output_dir: str | None = None,
) -> list[str]:
    """Split a PDF into chunks that fit within the size limit.

    Args:
        pdf_path: Path to the PDF file
        max_size_bytes: Maximum size per chunk in bytes
        output_dir: Directory for output files (uses temp dir if None)

    Returns:
        List of paths to the split PDF files
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError("PDF has no pages")

    # Use temp directory if no output dir specified
    if output_dir:
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(tempfile.mkdtemp(prefix="ocr_split_"))

    chunk_files: list[str] = []
    current_writer = PdfWriter()
    current_chunk = 1
    pages_in_chunk = 0

    for i, page in enumerate(reader.pages):
        current_writer.add_page(page)
        pages_in_chunk += 1

        # Write to buffer to check size
        buffer = io.BytesIO()
        current_writer.write(buffer)
        current_size = buffer.tell()

        # If chunk exceeds limit and has more than one page, save previous state
        if current_size > max_size_bytes and pages_in_chunk > 1:
            # Remove last page and save
            current_writer = PdfWriter()
            for j in range(i - pages_in_chunk + 1, i):
                current_writer.add_page(reader.pages[j])

            chunk_path = out_path / f"{path.stem}_chunk{current_chunk}.pdf"
            with open(chunk_path, "wb") as f:
                current_writer.write(f)
            chunk_files.append(str(chunk_path))
            current_chunk += 1

            # Start new chunk with current page
            current_writer = PdfWriter()
            current_writer.add_page(page)
            pages_in_chunk = 1

        # Handle single page exceeding limit (can't split further)
        elif current_size > max_size_bytes and pages_in_chunk == 1:
            # Save it anyway - it's a single page, can't split further
            chunk_path = out_path / f"{path.stem}_chunk{current_chunk}.pdf"
            with open(chunk_path, "wb") as f:
                current_writer.write(f)
            chunk_files.append(str(chunk_path))
            current_chunk += 1
            current_writer = PdfWriter()
            pages_in_chunk = 0

    # Save remaining pages
    if pages_in_chunk > 0:
        chunk_path = out_path / f"{path.stem}_chunk{current_chunk}.pdf"
        with open(chunk_path, "wb") as f:
            current_writer.write(f)
        chunk_files.append(str(chunk_path))

    return chunk_files


def compress_image(
    image_path: str,
    max_size_bytes: int,
    output_path: str | None = None,
    min_quality: int = 20,
) -> str:
    """Compress an image to fit within the size limit.

    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum size in bytes
        output_path: Output path (uses temp file if None)
        min_quality: Minimum JPEG quality (1-100)

    Returns:
        Path to the compressed image
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    current_size = path.stat().st_size
    if current_size <= max_size_bytes:
        return str(path)  # Already small enough

    img = Image.open(path)

    # Convert to RGB if necessary (for JPEG output)
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Determine output path
    if output_path:
        out_file = Path(output_path).expanduser().resolve()
    else:
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="ocr_compressed_")
        os.close(fd)
        out_file = Path(tmp_path)

    # Try progressively lower quality
    for quality in range(95, min_quality - 1, -5):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)

        if buffer.tell() <= max_size_bytes:
            with open(out_file, "wb") as f:
                f.write(buffer.getvalue())
            return str(out_file)

    # If still too large, also reduce dimensions
    scale = 0.9
    while scale > 0.3:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        for quality in range(85, min_quality - 1, -10):
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)

            if buffer.tell() <= max_size_bytes:
                with open(out_file, "wb") as f:
                    f.write(buffer.getvalue())
                return str(out_file)

        scale -= 0.1

    # Last resort: save at minimum quality and smallest scale
    final_size = (int(img.width * 0.3), int(img.height * 0.3))
    resized = img.resize(final_size, Image.Resampling.LANCZOS)
    resized.save(out_file, format="JPEG", quality=min_quality, optimize=True)
    return str(out_file)


def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF."""
    return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str) -> bool:
    """Check if file is an image."""
    return Path(file_path).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


# MCP Server setup
server = Server("ocr-space-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available OCR tools."""
    return [
        Tool(
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
                    "tier": {
                        "type": "string",
                        "enum": ["free", "pro"],
                        "default": "free",
                        "description": (
                            "API tier: 'free' (1MB limit, US servers) or "
                            "'pro' (5MB limit, EU servers for GDPR compliance)"
                        ),
                    },
                    "language": {
                        "type": "string",
                        "default": "eng",
                        "description": (
                            "OCR language code (e.g., 'eng', 'ger', 'fre'). "
                            "Use 'auto' for auto-detection (Engine 2 only)."
                        ),
                    },
                    "ocr_engine": {
                        "type": "integer",
                        "enum": [1, 2],
                        "default": 1,
                        "description": (
                            "Engine 1: Faster, Asian languages, larger images. "
                            "Engine 2: Auto-language detection, better special chars."
                        ),
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save extracted text (.txt) or full result (.json)",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["txt", "json"],
                        "default": "txt",
                        "description": "Output format: 'txt' (text only) or 'json' (full API response)",
                    },
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
        ),
        Tool(
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
                    "tier": {
                        "type": "string",
                        "enum": ["free", "pro"],
                        "default": "free",
                        "description": "API tier: 'free' or 'pro' (EU endpoint)",
                    },
                    "language": {
                        "type": "string",
                        "default": "eng",
                        "description": "OCR language code",
                    },
                    "ocr_engine": {
                        "type": "integer",
                        "enum": [1, 2],
                        "default": 1,
                        "description": "OCR engine (1 or 2)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save result",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["txt", "json"],
                        "default": "txt",
                        "description": "Output format",
                    },
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
        ),
        Tool(
            name="list_languages",
            description="List all supported OCR languages with their codes.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="check_tier_status",
            description=(
                "Check which API tiers are configured and available. "
                "Shows which environment variables are set."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
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
        ),
        Tool(
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
                    "tier": {
                        "type": "string",
                        "enum": ["free", "pro"],
                        "default": "free",
                        "description": "API tier: 'free' (1MB limit) or 'pro' (5MB limit)",
                    },
                    "language": {
                        "type": "string",
                        "default": "eng",
                        "description": "OCR language code",
                    },
                    "ocr_engine": {
                        "type": "integer",
                        "enum": [1, 2],
                        "default": 1,
                        "description": "OCR engine (1=faster, 2=better accuracy)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save extracted text",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["txt", "json"],
                        "default": "txt",
                        "description": "Output format",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "ocr_file":
            tier = Tier(arguments.get("tier", "free"))
            result = await call_ocr_api(
                tier,
                file_path=arguments["file_path"],
                language=arguments.get("language", "eng"),
                ocr_engine=arguments.get("ocr_engine", 1),
                detect_orientation=arguments.get("detect_orientation", False),
                scale=arguments.get("scale", False),
                is_table=arguments.get("is_table", False),
                is_overlay_required=arguments.get("is_overlay_required", False),
                is_create_searchable_pdf=arguments.get("is_create_searchable_pdf", False),
            )

            output = format_ocr_result(result)

            # Save if output path specified
            if arguments.get("output_path"):
                saved_path = await save_result(
                    result,
                    arguments["output_path"],
                    arguments.get("output_format", "txt"),
                )
                output += f"\n\nSaved to: {saved_path}"

            return [TextContent(type="text", text=output)]

        elif name == "ocr_url":
            tier = Tier(arguments.get("tier", "free"))
            result = await call_ocr_api(
                tier,
                url=arguments["url"],
                language=arguments.get("language", "eng"),
                ocr_engine=arguments.get("ocr_engine", 1),
                detect_orientation=arguments.get("detect_orientation", False),
                scale=arguments.get("scale", False),
                is_table=arguments.get("is_table", False),
            )

            output = format_ocr_result(result)

            if arguments.get("output_path"):
                saved_path = await save_result(
                    result,
                    arguments["output_path"],
                    arguments.get("output_format", "txt"),
                )
                output += f"\n\nSaved to: {saved_path}"

            return [TextContent(type="text", text=output)]

        elif name == "list_languages":
            lines = ["Supported OCR Languages:", ""]
            for code, name in sorted(LANGUAGES.items(), key=lambda x: x[1]):
                lines.append(f"  {code}: {name}")
            lines.append("")
            lines.append("Note: 'auto' requires OCR Engine 2")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "check_tier_status":
            lines = ["OCR.space API Tier Status:", ""]

            # Check free tier
            free_key = os.environ.get("OCR_SPACE_API_KEY")
            if free_key:
                masked = free_key[:4] + "..." + free_key[-4:] if len(free_key) > 8 else "***"
                lines.append(f"Free Tier: Configured (key: {masked})")
                lines.append(f"  Endpoint: {FREE_ENDPOINT}")
                lines.append("  Limit: 1 MB per file")
            else:
                lines.append("Free Tier: NOT CONFIGURED")
                lines.append("  Set OCR_SPACE_API_KEY environment variable")

            lines.append("")

            # Check PRO tier
            pro_key = os.environ.get("OCR_SPACE_PRO_API_KEY")
            if pro_key:
                masked = pro_key[:4] + "..." + pro_key[-4:] if len(pro_key) > 8 else "***"
                lines.append(f"PRO Tier: Configured (key: {masked})")
                pro_endpoint = os.environ.get("OCR_SPACE_PRO_ENDPOINT", DEFAULT_PRO_ENDPOINT)
                lines.append(f"  Endpoint: {pro_endpoint}")
                lines.append("  Limit: 5 MB per file")
                if "eu" in pro_endpoint.lower() or "europe" in pro_endpoint.lower():
                    lines.append("  GDPR: EU endpoint configured")
            else:
                lines.append("PRO Tier: NOT CONFIGURED")
                lines.append("  Set OCR_SPACE_PRO_API_KEY for PRO features")
                lines.append("  Set OCR_SPACE_PRO_ENDPOINT for EU/GDPR endpoint")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "split_pdf":
            tier = Tier(arguments.get("tier", "free"))
            size_limit = get_size_limit(tier)
            file_path = arguments["file_path"]
            output_dir = arguments.get("output_dir")

            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return [TextContent(type="text", text=f"Error: File not found: {path}")]

            if not is_pdf(str(path)):
                return [TextContent(type="text", text="Error: File is not a PDF")]

            file_size = path.stat().st_size
            if file_size <= size_limit:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"File is already within {tier.value} tier limit "
                            f"({file_size / (1024 * 1024):.2f} MB <= "
                            f"{size_limit / (1024 * 1024):.0f} MB). No splitting needed."
                        ),
                    )
                ]

            chunks = split_pdf_by_size(str(path), size_limit, output_dir)

            lines = [
                f"PDF split into {len(chunks)} chunks:",
                f"Original size: {file_size / (1024 * 1024):.2f} MB",
                f"Target limit: {size_limit / (1024 * 1024):.0f} MB ({tier.value} tier)",
                "",
                "Chunks created:",
            ]
            for i, chunk in enumerate(chunks, 1):
                chunk_size = Path(chunk).stat().st_size
                lines.append(f"  {i}. {chunk} ({chunk_size / (1024 * 1024):.2f} MB)")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "ocr_auto":
            tier = Tier(arguments.get("tier", "free"))
            size_limit = get_size_limit(tier)
            file_path = arguments["file_path"]
            language = arguments.get("language", "eng")
            ocr_engine = arguments.get("ocr_engine", 1)

            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return [TextContent(type="text", text=f"Error: File not found: {path}")]

            file_size = path.stat().st_size
            all_text: list[str] = []
            processing_notes: list[str] = []

            # Handle based on file type and size
            if is_pdf(str(path)):
                if file_size <= size_limit:
                    # PDF fits, OCR directly
                    processing_notes.append(
                        f"PDF ({file_size / (1024 * 1024):.2f} MB) fits within limit"
                    )
                    result = await call_ocr_api(
                        tier,
                        file_path=str(path),
                        language=language,
                        ocr_engine=ocr_engine,
                    )
                    all_text.append(extract_text_only(result))
                else:
                    # PDF too large, split and OCR each chunk
                    processing_notes.append(
                        f"PDF ({file_size / (1024 * 1024):.2f} MB) exceeds limit, splitting..."
                    )
                    chunks = split_pdf_by_size(str(path), size_limit)
                    processing_notes.append(f"Split into {len(chunks)} chunks")

                    for i, chunk in enumerate(chunks, 1):
                        processing_notes.append(f"Processing chunk {i}/{len(chunks)}...")
                        result = await call_ocr_api(
                            tier,
                            file_path=chunk,
                            language=language,
                            ocr_engine=ocr_engine,
                        )
                        text = extract_text_only(result)
                        if text:
                            all_text.append(f"[Chunk {i}]\n{text}")

            elif is_image(str(path)):
                if file_size <= size_limit:
                    # Image fits, OCR directly
                    processing_notes.append(
                        f"Image ({file_size / (1024 * 1024):.2f} MB) fits within limit"
                    )
                    result = await call_ocr_api(
                        tier,
                        file_path=str(path),
                        language=language,
                        ocr_engine=ocr_engine,
                    )
                    all_text.append(extract_text_only(result))
                else:
                    # Image too large, compress
                    processing_notes.append(
                        f"Image ({file_size / (1024 * 1024):.2f} MB) exceeds limit, compressing..."
                    )
                    # Use 90% of limit to be safe
                    compressed_path = compress_image(str(path), int(size_limit * 0.9))
                    new_size = Path(compressed_path).stat().st_size
                    processing_notes.append(f"Compressed to {new_size / (1024 * 1024):.2f} MB")

                    result = await call_ocr_api(
                        tier,
                        file_path=compressed_path,
                        language=language,
                        ocr_engine=ocr_engine,
                    )
                    all_text.append(extract_text_only(result))
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Error: Unsupported file type: {path.suffix}",
                    )
                ]

            # Combine results
            combined_text = "\n\n".join(all_text)
            output_lines = [
                "--- Processing Summary ---",
                *processing_notes,
                "",
                "--- Extracted Text ---",
                combined_text if combined_text else "(No text extracted)",
            ]

            output = "\n".join(output_lines)

            # Save if output path specified
            if arguments.get("output_path"):
                out_format = arguments.get("output_format", "txt")
                out_path = Path(arguments["output_path"]).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_format == "json":
                    result_data = {
                        "processing_notes": processing_notes,
                        "text": combined_text,
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False)
                else:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(combined_text)

                output += f"\n\nSaved to: {out_path}"

            return [TextContent(type="text", text=output)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


def main():
    """Run the MCP server."""
    import asyncio

    asyncio.run(_run_server())


async def _run_server():
    """Async server runner."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
