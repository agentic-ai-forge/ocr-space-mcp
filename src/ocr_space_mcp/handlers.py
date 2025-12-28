"""Tool handler implementations for OCR.space MCP Server."""

import json
from pathlib import Path
from typing import Any

from mcp.types import TextContent

from .client import OCROptions, get_client
from .config import FREE_ENDPOINT, FREE_SIZE_LIMIT, PRO_SIZE_LIMIT, get_settings
from .models import LANGUAGES, Tier, is_image, is_pdf
from .utils import (
    compress_image,
    extract_text_only,
    format_ocr_result,
    save_result,
    split_pdf_by_size,
)


def _build_options(arguments: dict[str, Any]) -> OCROptions:
    """Build OCROptions from tool arguments."""
    return OCROptions(
        language=arguments.get("language", "eng"),
        ocr_engine=arguments.get("ocr_engine", 1),
        detect_orientation=arguments.get("detect_orientation", False),
        scale=arguments.get("scale", False),
        is_table=arguments.get("is_table", False),
        is_overlay_required=arguments.get("is_overlay_required", False),
        is_create_searchable_pdf=arguments.get("is_create_searchable_pdf", False),
    )


async def handle_ocr_file(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle ocr_file tool call."""
    tier = Tier(arguments.get("tier", "free"))
    options = _build_options(arguments)

    async with get_client() as client:
        response = await client.ocr(tier, file_path=arguments["file_path"], options=options)

    output = format_ocr_result(response)
    output = _maybe_save_result(response, arguments, output)
    return [TextContent(type="text", text=output)]


async def handle_ocr_url(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle ocr_url tool call."""
    tier = Tier(arguments.get("tier", "free"))
    options = _build_options(arguments)

    async with get_client() as client:
        response = await client.ocr(tier, url=arguments["url"], options=options)

    output = format_ocr_result(response)
    output = _maybe_save_result(response, arguments, output)
    return [TextContent(type="text", text=output)]


async def handle_list_languages() -> list[TextContent]:
    """Handle list_languages tool call."""
    lines = ["Supported OCR Languages:", ""]
    for code, name in sorted(LANGUAGES.items(), key=lambda x: x[1]):
        lines.append(f"  {code}: {name}")
    lines.append("")
    lines.append("Note: 'auto' requires OCR Engine 2")
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_check_tier_status() -> list[TextContent]:
    """Handle check_tier_status tool call."""
    settings = get_settings()
    lines = ["OCR.space API Tier Status:", ""]

    # Check free tier
    if settings.api_key:
        key = settings.api_key.get_secret_value()
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        lines.append(f"Free Tier: Configured (key: {masked})")
        lines.append(f"  Endpoint: {FREE_ENDPOINT}")
        lines.append("  Limit: 1 MB per file")
    else:
        lines.append("Free Tier: NOT CONFIGURED")
        lines.append("  Set OCR_SPACE_API_KEY environment variable")

    lines.append("")

    # Check PRO tier
    if settings.pro_api_key:
        key = settings.pro_api_key.get_secret_value()
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        lines.append(f"PRO Tier: Configured (key: {masked})")
        lines.append(f"  Endpoint: {settings.pro_endpoint}")
        lines.append("  Limit: 5 MB per file")
        if "eu" in settings.pro_endpoint.lower():
            lines.append("  GDPR: EU endpoint configured")
    else:
        lines.append("PRO Tier: NOT CONFIGURED")
        lines.append("  Set OCR_SPACE_PRO_API_KEY for PRO features")
        lines.append("  Set OCR_SPACE_PRO_ENDPOINT for EU/GDPR endpoint")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_split_pdf(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle split_pdf tool call."""
    tier = Tier(arguments.get("tier", "free"))
    size_limit = PRO_SIZE_LIMIT if tier == Tier.PRO else FREE_SIZE_LIMIT
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
    lines = _format_split_result(chunks, file_size, size_limit, tier)
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_ocr_auto(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle ocr_auto tool call - smart OCR with auto file handling."""
    tier = Tier(arguments.get("tier", "free"))
    size_limit = PRO_SIZE_LIMIT if tier == Tier.PRO else FREE_SIZE_LIMIT
    file_path = arguments["file_path"]
    options = _build_options(arguments)

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {path}")]

    file_size = path.stat().st_size

    if is_pdf(str(path)):
        all_text, notes = await _ocr_pdf(path, file_size, size_limit, tier, options)
    elif is_image(str(path)):
        all_text, notes = await _ocr_image(path, file_size, size_limit, tier, options)
    else:
        return [TextContent(type="text", text=f"Error: Unsupported file type: {path.suffix}")]

    output = _format_auto_result(all_text, notes)
    output = _maybe_save_auto_result(arguments, all_text, notes, output)
    return [TextContent(type="text", text=output)]


# --- Helper functions ---


def _maybe_save_result(response, arguments: dict[str, Any], output: str) -> str:
    """Optionally save result to file and append path to output."""
    if arguments.get("output_path"):
        saved_path = save_result(
            response,
            arguments["output_path"],
            arguments.get("output_format", "txt"),
        )
        output += f"\n\nSaved to: {saved_path}"
    return output


def _format_split_result(
    chunks: list[str],
    file_size: int,
    size_limit: int,
    tier: Tier,
) -> list[str]:
    """Format PDF split result for display."""
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
    return lines


async def _ocr_pdf(
    path: Path,
    file_size: int,
    size_limit: int,
    tier: Tier,
    options: OCROptions,
) -> tuple[list[str], list[str]]:
    """Process PDF file, splitting if needed."""
    all_text: list[str] = []
    notes: list[str] = []

    if file_size <= size_limit:
        notes.append(f"PDF ({file_size / (1024 * 1024):.2f} MB) fits within limit")
        async with get_client() as client:
            response = await client.ocr(tier, file_path=str(path), options=options)
        all_text.append(extract_text_only(response))
    else:
        notes.append(f"PDF ({file_size / (1024 * 1024):.2f} MB) exceeds limit, splitting...")
        chunks = split_pdf_by_size(str(path), size_limit)
        notes.append(f"Split into {len(chunks)} chunks")

        async with get_client() as client:
            for i, chunk in enumerate(chunks, 1):
                notes.append(f"Processing chunk {i}/{len(chunks)}...")
                response = await client.ocr(tier, file_path=chunk, options=options)
                text = extract_text_only(response)
                if text:
                    all_text.append(f"[Chunk {i}]\n{text}")

    return all_text, notes


async def _ocr_image(
    path: Path,
    file_size: int,
    size_limit: int,
    tier: Tier,
    options: OCROptions,
) -> tuple[list[str], list[str]]:
    """Process image file, compressing if needed."""
    all_text: list[str] = []
    notes: list[str] = []

    if file_size <= size_limit:
        notes.append(f"Image ({file_size / (1024 * 1024):.2f} MB) fits within limit")
        file_to_ocr = str(path)
    else:
        notes.append(f"Image ({file_size / (1024 * 1024):.2f} MB) exceeds limit, compressing...")
        file_to_ocr = compress_image(str(path), int(size_limit * 0.9))
        new_size = Path(file_to_ocr).stat().st_size
        notes.append(f"Compressed to {new_size / (1024 * 1024):.2f} MB")

    async with get_client() as client:
        response = await client.ocr(tier, file_path=file_to_ocr, options=options)
    all_text.append(extract_text_only(response))

    return all_text, notes


def _format_auto_result(all_text: list[str], notes: list[str]) -> str:
    """Format auto OCR result for display."""
    combined_text = "\n\n".join(all_text)
    output_lines = [
        "--- Processing Summary ---",
        *notes,
        "",
        "--- Extracted Text ---",
        combined_text if combined_text else "(No text extracted)",
    ]
    return "\n".join(output_lines)


def _maybe_save_auto_result(
    arguments: dict[str, Any],
    all_text: list[str],
    notes: list[str],
    output: str,
) -> str:
    """Optionally save auto OCR result to file."""
    if not arguments.get("output_path"):
        return output

    out_format = arguments.get("output_format", "txt")
    out_path = Path(arguments["output_path"]).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined_text = "\n\n".join(all_text)

    if out_format == "json":
        result_data = {"processing_notes": notes, "text": combined_text}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

    return output + f"\n\nSaved to: {out_path}"
