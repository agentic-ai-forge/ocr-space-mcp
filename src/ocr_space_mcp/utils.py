"""Utility functions for OCR.space MCP Server."""

import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

from .models import OCRResponse


def format_ocr_result(response: OCRResponse) -> str:
    """Format OCR response for display."""
    lines = []

    # Status based on exit code
    status_map = {
        1: "Status: Success",
        2: "Status: Partial success (some pages failed)",
        3: "Status: Failed",
        4: "Status: Error",
    }
    lines.append(status_map.get(response.exit_code, f"Status: Unknown ({response.exit_code})"))

    if response.processing_time_ms:
        lines.append(f"Processing time: {response.processing_time_ms}ms")

    if response.searchable_pdf_url:
        lines.append(f"Searchable PDF: {response.searchable_pdf_url}")

    if response.parsed_results:
        lines.append(f"Pages processed: {len(response.parsed_results)}")
        lines.append("")
        lines.append("--- Extracted Text ---")

        for i, page in enumerate(response.parsed_results):
            if len(response.parsed_results) > 1:
                lines.append(f"\n[Page {i + 1}]")
            text = page.parsed_text.strip()
            lines.append(text if text else "(No text extracted)")
            if page.error_message:
                lines.append(f"Error: {page.error_message}")

    if response.error_message_str:
        lines.append(f"\nAPI Error: {response.error_message_str}")

    if response.error_details:
        lines.append(f"Details: {response.error_details}")

    return "\n".join(lines)


def extract_text_only(response: OCRResponse) -> str:
    """Extract only the parsed text from OCR response."""
    texts = []
    for page in response.parsed_results:
        text = page.parsed_text.strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def save_result(
    response: OCRResponse,
    output_path: str,
    output_format: str = "txt",
) -> str:
    """Save OCR result to file.

    Args:
        response: OCR response to save
        output_path: Destination file path
        output_format: 'txt' for text only, 'json' for full response

    Returns:
        Absolute path to saved file
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)
    else:
        text = extract_text_only(response)
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
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages")

    out_path = _get_output_dir(output_dir)
    return _split_pages_into_chunks(reader, path.stem, max_size_bytes, out_path)


def _get_output_dir(output_dir: str | None) -> Path:
    """Get or create output directory."""
    if output_dir:
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path
    return Path(tempfile.mkdtemp(prefix="ocr_split_"))


def _split_pages_into_chunks(
    reader: PdfReader,
    stem: str,
    max_size_bytes: int,
    out_path: Path,
) -> list[str]:
    """Split PDF pages into size-limited chunks."""
    chunk_files: list[str] = []
    current_writer = PdfWriter()
    current_chunk = 1
    pages_in_chunk = 0

    for i, page in enumerate(reader.pages):
        current_writer.add_page(page)
        pages_in_chunk += 1

        current_size = _get_writer_size(current_writer)

        if current_size > max_size_bytes:
            if pages_in_chunk > 1:
                # Chunk exceeded, save without last page
                chunk_path = _save_chunk_without_last(
                    reader, i, pages_in_chunk, stem, current_chunk, out_path
                )
                chunk_files.append(chunk_path)
                current_chunk += 1

                # Start new chunk with current page
                current_writer = PdfWriter()
                current_writer.add_page(page)
                pages_in_chunk = 1
            else:
                # Single page exceeds limit, save anyway
                chunk_path = _save_current_chunk(current_writer, stem, current_chunk, out_path)
                chunk_files.append(chunk_path)
                current_chunk += 1
                current_writer = PdfWriter()
                pages_in_chunk = 0

    # Save remaining pages
    if pages_in_chunk > 0:
        chunk_path = _save_current_chunk(current_writer, stem, current_chunk, out_path)
        chunk_files.append(chunk_path)

    return chunk_files


def _get_writer_size(writer: PdfWriter) -> int:
    """Get current size of PDF writer content."""
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.tell()


def _save_current_chunk(writer: PdfWriter, stem: str, chunk_num: int, out_path: Path) -> str:
    """Save current writer content to a chunk file."""
    chunk_path = out_path / f"{stem}_chunk{chunk_num}.pdf"
    with open(chunk_path, "wb") as f:
        writer.write(f)
    return str(chunk_path)


def _save_chunk_without_last(
    reader: PdfReader,
    current_idx: int,
    pages_in_chunk: int,
    stem: str,
    chunk_num: int,
    out_path: Path,
) -> str:
    """Save chunk excluding the last added page."""
    writer = PdfWriter()
    start_idx = current_idx - pages_in_chunk + 1
    for j in range(start_idx, current_idx):
        writer.add_page(reader.pages[j])
    return _save_current_chunk(writer, stem, chunk_num, out_path)


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

    if path.stat().st_size <= max_size_bytes:
        return str(path)  # Already small enough

    img = _prepare_image_for_jpeg(Image.open(path))
    out_file = _get_output_file(output_path)

    # Try quality reduction first
    result = _try_quality_compression(img, max_size_bytes, min_quality)
    if result:
        out_file.write_bytes(result)
        return str(out_file)

    # Try dimension reduction + quality
    result = _try_resize_compression(img, max_size_bytes, min_quality)
    if result:
        out_file.write_bytes(result)
        return str(out_file)

    # Last resort: minimum settings
    final_img = img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.Resampling.LANCZOS)
    final_img.save(out_file, format="JPEG", quality=min_quality, optimize=True)
    return str(out_file)


def _prepare_image_for_jpeg(img: Image.Image) -> Image.Image:
    """Convert image to RGB mode for JPEG output."""
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        mask = img.split()[-1] if img.mode == "RGBA" else None
        background.paste(img, mask=mask)
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _get_output_file(output_path: str | None) -> Path:
    """Get output file path."""
    if output_path:
        return Path(output_path).expanduser().resolve()
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="ocr_compressed_")
    os.close(fd)
    return Path(tmp_path)


def _try_quality_compression(
    img: Image.Image,
    max_size: int,
    min_quality: int,
) -> bytes | None:
    """Try reducing JPEG quality. Returns bytes if successful."""
    for quality in range(95, min_quality - 1, -5):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        if buffer.tell() <= max_size:
            return buffer.getvalue()
    return None


def _try_resize_compression(
    img: Image.Image,
    max_size: int,
    min_quality: int,
) -> bytes | None:
    """Try reducing dimensions + quality. Returns bytes if successful."""
    scale = 0.9
    while scale > 0.3:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        for quality in range(85, min_quality - 1, -10):
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            if buffer.tell() <= max_size:
                return buffer.getvalue()

        scale -= 0.1
    return None
