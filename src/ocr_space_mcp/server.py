"""OCR.space MCP Server - Text extraction from images and PDFs.

Exposes six tools:
- ocr_file: Extract text from local image/PDF files
- ocr_url: Extract text from image/PDF at URL
- ocr_auto: Smart OCR with automatic oversized file handling
- split_pdf: Split large PDFs into size-limited chunks
- list_languages: List supported OCR languages
- check_tier_status: Check API tier configuration

API Reference: https://ocr.space/ocrapi
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .handlers import (
    handle_check_tier_status,
    handle_list_languages,
    handle_ocr_auto,
    handle_ocr_file,
    handle_ocr_url,
    handle_split_pdf,
)
from .tools import get_tools

# MCP Server instance
server = Server("ocr-space-mcp")

# Tool handler dispatch table
HANDLERS = {
    "ocr_file": handle_ocr_file,
    "ocr_url": handle_ocr_url,
    "ocr_auto": handle_ocr_auto,
    "split_pdf": handle_split_pdf,
    "list_languages": lambda _: handle_list_languages(),
    "check_tier_status": lambda _: handle_check_tier_status(),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available OCR tools."""
    return get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool by name."""
    handler = HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        return await handler(arguments)
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


def main():
    """Run the MCP server."""
    asyncio.run(_run_server())


async def _run_server():
    """Async server runner."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
