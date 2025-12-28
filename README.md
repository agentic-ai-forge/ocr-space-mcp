# OCR.space MCP Server

[![Pipeline Status](https://gitlab.com/agentic.ai.forge/ocr-space-mcp/badges/main/pipeline.svg)](https://gitlab.com/agentic.ai.forge/ocr-space-mcp/-/pipelines)
[![Coverage](https://gitlab.com/agentic.ai.forge/ocr-space-mcp/badges/main/coverage.svg)](https://gitlab.com/agentic.ai.forge/ocr-space-mcp/-/commits/main)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python MCP server for [OCR.space](https://ocr.space/) - extract text from images and PDFs using OCR.

## Features

- **Six comprehensive tools:**
  - `ocr_file` - Extract text from local images/PDFs
  - `ocr_url` - Extract text from images/PDFs at URLs
  - `ocr_auto` - **Smart OCR** that auto-handles oversized files
  - `split_pdf` - Split large PDFs into API-compatible chunks
  - `list_languages` - Show supported OCR languages
  - `check_tier_status` - Verify API configuration

- **Automatic oversized file handling:**
  - PDFs too large? Automatically split by pages, OCR each chunk, join results
  - Images too large? Automatically compress while preserving quality

- **Dual-tier support:**
  - **Free tier**: 1 MB limit, US servers
  - **PRO tier**: 5 MB limit, EU endpoint for GDPR compliance

- **Full API parameter support:**
  - 26+ languages including auto-detection
  - Two OCR engines (optimized for different use cases)
  - Table optimization mode
  - Auto-rotation detection
  - Searchable PDF generation
  - Save results as TXT or JSON

## Installation

```bash
# With uv (recommended)
uv sync

# With pip
pip install -e .
```

## Configuration

### API Keys

Get your free API key at [ocr.space/ocrapi/freekey](https://ocr.space/ocrapi/freekey).

For PRO features (larger files, EU endpoint), sign up at [ocr.space/ocrapi](https://ocr.space/ocrapi).

### Environment Variables

Copy `.envrc.example` to `.envrc` and configure:

```bash
# Required: Free tier API key
export OCR_SPACE_API_KEY="your-free-api-key"

# Optional: PRO tier for GDPR-compliant EU processing
export OCR_SPACE_PRO_API_KEY="your-pro-api-key"
export OCR_SPACE_PRO_ENDPOINT="https://eu.api.ocr.space/parse/image"
```

Then enable with `direnv allow`.

### Claude Code / MCP Config

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "ocr-space": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/ocr-space-mcp", "ocr-space-mcp"],
      "env": {
        "OCR_SPACE_API_KEY": "${OCR_SPACE_API_KEY}",
        "OCR_SPACE_PRO_API_KEY": "${OCR_SPACE_PRO_API_KEY}",
        "OCR_SPACE_PRO_ENDPOINT": "${OCR_SPACE_PRO_ENDPOINT}"
      }
    }
  }
}
```

## Usage

### Check Configuration

```
Check my OCR.space tier status
```

### OCR a Local File

```
Extract text from /path/to/document.pdf
```

```
OCR this image using the PRO tier: /path/to/scan.png
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `file_path` | Path to image/PDF (required) | - |
| `tier` | `free` or `pro` | `free` |
| `language` | OCR language code | `eng` |
| `ocr_engine` | 1 (fast) or 2 (accurate) | 1 |
| `output_path` | Save result to file | - |
| `output_format` | `txt` or `json` | `txt` |
| `detect_orientation` | Auto-rotate image | `false` |
| `scale` | Upscale low-res images | `false` |
| `is_table` | Optimize for tables | `false` |
| `is_create_searchable_pdf` | Generate searchable PDF | `false` |

### OCR from URL

```
Extract text from https://example.com/document.png
```

### Smart OCR for Large Files (ocr_auto)

The `ocr_auto` tool automatically handles files that exceed the API size limits:

```
OCR this large document: /path/to/big-scan.pdf
```

**How it works:**
- **PDFs**: Splits into chunks by page, OCRs each chunk, joins results
- **Images**: Compresses (quality + resize) until under limit, then OCRs

**Parameters:** Same as `ocr_file`, but no size limit errors!

### Split PDF Only

If you just want to split a large PDF without OCR:

```
Split /path/to/large.pdf for free tier processing
```

Creates multiple smaller PDF files in a temp directory (or specify `output_dir`).

### List Languages

```
What OCR languages are supported?
```

**Supported languages:** Arabic, Bulgarian, Chinese (Simplified/Traditional), Croatian, Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Slovenian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese.

Use `language=auto` with Engine 2 for automatic language detection.

## Tier Comparison

| Feature | Free | PRO |
|---------|------|-----|
| File size limit | 1 MB | 5 MB |
| Server location | US only | EU available |
| Rate limit | 500/day | Unlimited |
| GDPR compliance | No | Yes (EU endpoint) |
| Price | Free | $30/month |

## OCR Engines

| Engine | Best For |
|--------|----------|
| **Engine 1** | Faster processing, Asian languages, large images |
| **Engine 2** | Auto-language detection, special characters, rotated text |

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=ocr_space_mcp --cov-report=term-missing
```

## License

MIT - see [LICENSE](LICENSE)

## Links

- [OCR.space API Documentation](https://ocr.space/ocrapi)
- [Get Free API Key](https://ocr.space/ocrapi/freekey)
- [GitLab Repository](https://gitlab.com/agentic.ai.forge/ocr-space-mcp)
- [GitHub Mirror](https://github.com/agentic-ai-forge/ocr-space-mcp)
