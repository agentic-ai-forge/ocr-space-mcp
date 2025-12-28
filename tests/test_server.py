"""Tests for OCR.space MCP server."""

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from ocr_space_mcp.server import (
    FREE_ENDPOINT,
    FREE_SIZE_LIMIT,
    LANGUAGES,
    PRO_SIZE_LIMIT,
    Tier,
    call_ocr_api,
    call_tool,
    compress_image,
    extract_text_only,
    format_ocr_result,
    get_api_key,
    get_content_type,
    get_endpoint,
    get_size_limit,
    is_image,
    is_pdf,
    list_tools,
    save_result,
    split_pdf_by_size,
)

# --- Fixtures ---


@pytest.fixture
def mock_env_free():
    """Set up free tier environment."""
    with patch.dict(os.environ, {"OCR_SPACE_API_KEY": "test-free-key"}, clear=False):
        yield


@pytest.fixture
def mock_env_pro():
    """Set up PRO tier environment."""
    with patch.dict(
        os.environ,
        {
            "OCR_SPACE_API_KEY": "test-free-key",
            "OCR_SPACE_PRO_API_KEY": "test-pro-key",
            "OCR_SPACE_PRO_ENDPOINT": "https://eu.api.ocr.space/parse/image",
        },
        clear=False,
    ):
        yield


@pytest.fixture
def sample_ocr_response():
    """Sample successful OCR API response."""
    return {
        "ParsedResults": [
            {
                "TextOverlay": None,
                "TextOrientation": "0",
                "FileParseExitCode": 1,
                "ParsedText": "Hello World\nThis is a test document.",
                "ErrorMessage": "",
                "ErrorDetails": "",
            }
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ProcessingTimeInMilliseconds": "234",
        "SearchablePDFURL": None,
    }


@pytest.fixture
def sample_multipage_response():
    """Sample multi-page OCR response."""
    return {
        "ParsedResults": [
            {"ParsedText": "Page 1 content", "FileParseExitCode": 1, "ErrorMessage": ""},
            {"ParsedText": "Page 2 content", "FileParseExitCode": 1, "ErrorMessage": ""},
            {"ParsedText": "Page 3 content", "FileParseExitCode": 1, "ErrorMessage": ""},
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ProcessingTimeInMilliseconds": "1234",
    }


@pytest.fixture
def temp_image_file():
    """Create a temporary test image file."""
    # Create a minimal valid PNG (1x1 transparent pixel)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_large_file():
    """Create a file that exceeds free tier limit."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Write 1.5 MB of data (exceeds 1 MB free limit)
        f.write(b"x" * int(1.5 * 1024 * 1024))
        yield f.name
    os.unlink(f.name)


# --- Unit Tests: Configuration ---


class TestConfiguration:
    """Test configuration and tier management."""

    def test_get_api_key_free(self, mock_env_free):
        """Test getting free tier API key."""
        key = get_api_key(Tier.FREE)
        assert key == "test-free-key"

    def test_get_api_key_pro(self, mock_env_pro):
        """Test getting PRO tier API key."""
        key = get_api_key(Tier.PRO)
        assert key == "test-pro-key"

    def test_get_api_key_free_missing(self):
        """Test error when free key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OCR_SPACE_API_KEY not set"):
                get_api_key(Tier.FREE)

    def test_get_api_key_pro_missing(self, mock_env_free):
        """Test error when PRO key is missing but requested."""
        with pytest.raises(ValueError, match="OCR_SPACE_PRO_API_KEY not set"):
            get_api_key(Tier.PRO)

    def test_get_endpoint_free(self):
        """Test free tier endpoint."""
        endpoint = get_endpoint(Tier.FREE)
        assert endpoint == FREE_ENDPOINT

    def test_get_endpoint_pro_default(self, mock_env_free):
        """Test PRO tier uses default endpoint if not configured."""
        endpoint = get_endpoint(Tier.PRO)
        assert "apipro" in endpoint

    def test_get_endpoint_pro_custom(self, mock_env_pro):
        """Test PRO tier uses custom EU endpoint."""
        endpoint = get_endpoint(Tier.PRO)
        assert endpoint == "https://eu.api.ocr.space/parse/image"

    def test_get_size_limit_free(self):
        """Test free tier size limit."""
        limit = get_size_limit(Tier.FREE)
        assert limit == FREE_SIZE_LIMIT
        assert limit == 1 * 1024 * 1024

    def test_get_size_limit_pro(self):
        """Test PRO tier size limit."""
        limit = get_size_limit(Tier.PRO)
        assert limit == PRO_SIZE_LIMIT
        assert limit == 5 * 1024 * 1024


class TestContentType:
    """Test content type detection."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.png", "image/png"),
            ("test.PNG", "image/png"),
            ("test.jpg", "image/jpeg"),
            ("test.jpeg", "image/jpeg"),
            ("test.pdf", "application/pdf"),
            ("test.gif", "image/gif"),
            ("test.tif", "image/tiff"),
            ("test.tiff", "image/tiff"),
            ("test.bmp", "image/bmp"),
            ("test.webp", "image/webp"),
            ("test.unknown", "application/octet-stream"),
        ],
    )
    def test_get_content_type(self, filename, expected):
        """Test content type detection for various file types."""
        assert get_content_type(filename) == expected


# --- Unit Tests: Result Formatting ---


class TestResultFormatting:
    """Test OCR result formatting."""

    def test_format_ocr_result_success(self, sample_ocr_response):
        """Test formatting successful OCR result."""
        output = format_ocr_result(sample_ocr_response)
        assert "Status: Success" in output
        assert "Hello World" in output
        assert "This is a test document" in output
        assert "234ms" in output

    def test_format_ocr_result_multipage(self, sample_multipage_response):
        """Test formatting multi-page result."""
        output = format_ocr_result(sample_multipage_response)
        assert "Pages processed: 3" in output
        assert "[Page 1]" in output
        assert "[Page 2]" in output
        assert "[Page 3]" in output
        assert "Page 1 content" in output
        assert "Page 3 content" in output

    def test_format_ocr_result_partial_failure(self):
        """Test formatting partial failure."""
        result = {"OCRExitCode": 2, "ParsedResults": []}
        output = format_ocr_result(result)
        assert "Partial success" in output

    def test_format_ocr_result_failure(self):
        """Test formatting complete failure."""
        result = {"OCRExitCode": 3, "ParsedResults": [], "ErrorMessage": "Invalid file"}
        output = format_ocr_result(result)
        assert "Status: Failed" in output
        assert "Invalid file" in output

    def test_extract_text_only(self, sample_ocr_response):
        """Test extracting text only."""
        text = extract_text_only(sample_ocr_response)
        assert text == "Hello World\nThis is a test document."

    def test_extract_text_only_multipage(self, sample_multipage_response):
        """Test extracting text from multiple pages."""
        text = extract_text_only(sample_multipage_response)
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert "Page 3 content" in text


# --- Unit Tests: File Saving ---


class TestFileSaving:
    """Test result file saving."""

    @pytest.mark.asyncio
    async def test_save_result_txt(self, sample_ocr_response):
        """Test saving as text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            saved = await save_result(sample_ocr_response, str(output_path), "txt")

            assert Path(saved).exists()
            content = Path(saved).read_text()
            assert "Hello World" in content
            assert "This is a test document" in content

    @pytest.mark.asyncio
    async def test_save_result_json(self, sample_ocr_response):
        """Test saving as JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            saved = await save_result(sample_ocr_response, str(output_path), "json")

            assert Path(saved).exists()
            content = json.loads(Path(saved).read_text())
            assert content["OCRExitCode"] == 1
            assert len(content["ParsedResults"]) == 1

    @pytest.mark.asyncio
    async def test_save_result_creates_directory(self, sample_ocr_response):
        """Test that saving creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.txt"
            saved = await save_result(sample_ocr_response, str(output_path), "txt")
            assert Path(saved).exists()


# --- Unit Tests: API Calls ---


class TestAPICall:
    """Test API call functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_file_success(self, mock_env_free, temp_image_file, sample_ocr_response):
        """Test successful file OCR."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        result = await call_ocr_api(Tier.FREE, file_path=temp_image_file)

        assert result["OCRExitCode"] == 1
        assert len(result["ParsedResults"]) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_url_success(self, mock_env_free, sample_ocr_response):
        """Test successful URL OCR."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        result = await call_ocr_api(Tier.FREE, url="https://example.com/image.png")

        assert result["OCRExitCode"] == 1

    @pytest.mark.asyncio
    async def test_ocr_file_not_found(self, mock_env_free):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            await call_ocr_api(Tier.FREE, file_path="/nonexistent/file.png")

    @pytest.mark.asyncio
    async def test_ocr_file_too_large_free(self, mock_env_free, temp_large_file):
        """Test error when file exceeds free tier limit."""
        with pytest.raises(ValueError, match="exceeds free tier limit"):
            await call_ocr_api(Tier.FREE, file_path=temp_large_file)

    @pytest.mark.asyncio
    async def test_ocr_no_input(self, mock_env_free):
        """Test error when no input provided."""
        with pytest.raises(ValueError, match="Must provide"):
            await call_ocr_api(Tier.FREE)

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_with_options(self, mock_env_free, temp_image_file, sample_ocr_response):
        """Test OCR with various options."""
        route = respx.post(FREE_ENDPOINT).mock(
            return_value=httpx.Response(200, json=sample_ocr_response)
        )

        await call_ocr_api(
            Tier.FREE,
            file_path=temp_image_file,
            language="ger",
            ocr_engine=2,
            detect_orientation=True,
            scale=True,
            is_table=True,
        )

        # Verify request was made with correct parameters
        assert route.called
        request = route.calls[0].request
        # The request body contains form data
        body = request.content.decode()
        assert "ger" in body
        assert "OCREngine" in body


# --- Unit Tests: MCP Tools ---


class TestMCPTools:
    """Test MCP tool definitions and handlers."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test tool listing."""
        tools = await list_tools()
        assert len(tools) == 6

        tool_names = [t.name for t in tools]
        assert "ocr_file" in tool_names
        assert "ocr_url" in tool_names
        assert "list_languages" in tool_names
        assert "check_tier_status" in tool_names
        assert "split_pdf" in tool_names
        assert "ocr_auto" in tool_names

    @pytest.mark.asyncio
    async def test_list_languages_tool(self):
        """Test list_languages tool."""
        result = await call_tool("list_languages", {})
        assert len(result) == 1
        text = result[0].text
        assert "English" in text
        assert "German" in text
        assert "eng:" in text
        assert "ger:" in text

    @pytest.mark.asyncio
    async def test_check_tier_status_free_only(self, mock_env_free):
        """Test tier status with only free tier configured."""
        result = await call_tool("check_tier_status", {})
        text = result[0].text
        assert "Free Tier: Configured" in text
        assert "PRO Tier: NOT CONFIGURED" in text

    @pytest.mark.asyncio
    async def test_check_tier_status_both_tiers(self, mock_env_pro):
        """Test tier status with both tiers configured."""
        result = await call_tool("check_tier_status", {})
        text = result[0].text
        assert "Free Tier: Configured" in text
        assert "PRO Tier: Configured" in text
        assert "eu.api.ocr.space" in text

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_file_tool(self, mock_env_free, temp_image_file, sample_ocr_response):
        """Test ocr_file tool."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        result = await call_tool("ocr_file", {"file_path": temp_image_file})

        text = result[0].text
        assert "Status: Success" in text
        assert "Hello World" in text

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_file_tool_with_save(
        self, mock_env_free, temp_image_file, sample_ocr_response
    ):
        """Test ocr_file tool with file saving."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.txt"
            result = await call_tool(
                "ocr_file",
                {
                    "file_path": temp_image_file,
                    "output_path": str(output_path),
                    "output_format": "txt",
                },
            )

            text = result[0].text
            assert "Saved to:" in text
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test handling of unknown tool."""
        result = await call_tool("nonexistent_tool", {})
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test error handling in tool calls."""
        result = await call_tool("ocr_file", {"file_path": "/nonexistent/file.png"})
        assert "Error:" in result[0].text


# --- Unit Tests: Languages ---


class TestLanguages:
    """Test language support."""

    def test_all_languages_defined(self):
        """Test all expected languages are defined."""
        expected = ["eng", "ger", "fre", "spa", "ita", "jpn", "chs", "cht", "kor", "ara", "rus"]
        for lang in expected:
            assert lang in LANGUAGES

    def test_auto_language(self):
        """Test auto language detection option."""
        assert "auto" in LANGUAGES
        assert "Engine 2" in LANGUAGES["auto"]


# --- Unit Tests: File Type Detection ---


class TestFileTypeDetection:
    """Test file type detection helpers."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("document.pdf", True),
            ("test.png", False),
            ("test.jpg", False),
            ("test.txt", False),
        ],
    )
    def test_is_pdf(self, filename, expected):
        """Test PDF detection."""
        assert is_pdf(filename) == expected

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.png", True),
            ("test.PNG", True),
            ("test.jpg", True),
            ("test.jpeg", True),
            ("test.gif", True),
            ("test.bmp", True),
            ("test.tif", True),
            ("test.tiff", True),
            ("test.webp", True),
            ("test.pdf", False),
            ("test.txt", False),
        ],
    )
    def test_is_image(self, filename, expected):
        """Test image detection."""
        assert is_image(filename) == expected


# --- Unit Tests: PDF Splitting ---


class TestPDFSplitting:
    """Test PDF splitting functionality."""

    @pytest.fixture
    def temp_pdf_file(self):
        """Create a temporary minimal PDF file."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            # Create a minimal PDF with 3 pages
            for _ in range(3):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            yield f.name
        os.unlink(f.name)

    def test_split_pdf_file_not_found(self):
        """Test error when PDF not found."""
        with pytest.raises(FileNotFoundError):
            split_pdf_by_size("/nonexistent/file.pdf", 1024 * 1024)

    def test_split_pdf_creates_chunks(self, temp_pdf_file):
        """Test that splitting creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small limit to force splitting
            chunks = split_pdf_by_size(temp_pdf_file, 100, tmpdir)

            assert len(chunks) >= 1
            for chunk in chunks:
                assert Path(chunk).exists()
                assert Path(chunk).suffix == ".pdf"

    def test_split_pdf_no_split_needed(self, temp_pdf_file):
        """Test that small PDFs don't need splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use large limit - no splitting needed
            chunks = split_pdf_by_size(temp_pdf_file, 10 * 1024 * 1024, tmpdir)

            assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_split_pdf_tool_file_not_found(self):
        """Test split_pdf tool with missing file."""
        result = await call_tool("split_pdf", {"file_path": "/nonexistent/file.pdf"})
        assert "Error: File not found" in result[0].text

    @pytest.mark.asyncio
    async def test_split_pdf_tool_not_pdf(self, temp_image_file):
        """Test split_pdf tool with non-PDF file."""
        result = await call_tool("split_pdf", {"file_path": temp_image_file})
        assert "Error: File is not a PDF" in result[0].text

    @pytest.mark.asyncio
    async def test_split_pdf_tool_already_small(self, temp_pdf_file):
        """Test split_pdf tool with file already under limit."""
        result = await call_tool("split_pdf", {"file_path": temp_pdf_file, "tier": "pro"})
        # 5MB limit should be enough for a small test PDF
        assert "No splitting needed" in result[0].text


# --- Unit Tests: Image Compression ---


class TestImageCompression:
    """Test image compression functionality."""

    @pytest.fixture
    def temp_large_image(self):
        """Create a temporary large image file."""
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a large image (1000x1000 with random-ish data)
            img = PILImage.new("RGB", (1000, 1000), color=(255, 128, 64))
            img.save(f.name, format="PNG")
            yield f.name
        os.unlink(f.name)

    def test_compress_image_file_not_found(self):
        """Test error when image not found."""
        with pytest.raises(FileNotFoundError):
            compress_image("/nonexistent/file.png", 1024 * 1024)

    def test_compress_image_already_small(self, temp_image_file):
        """Test that small images are returned as-is."""
        result = compress_image(temp_image_file, 10 * 1024 * 1024)
        # Should return the original file since it's under the limit
        assert result == temp_image_file

    def test_compress_image_reduces_size(self, temp_large_image):
        """Test that compression reduces file size."""
        original_size = Path(temp_large_image).stat().st_size
        target_size = original_size // 2  # Target half the size

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compressed.jpg"
            result = compress_image(temp_large_image, target_size, str(output_path))

            assert Path(result).exists()
            new_size = Path(result).stat().st_size
            assert new_size <= target_size


# --- Unit Tests: OCR Auto Tool ---


class TestOCRAutoTool:
    """Test ocr_auto tool."""

    @pytest.mark.asyncio
    async def test_ocr_auto_file_not_found(self):
        """Test ocr_auto tool with missing file."""
        result = await call_tool("ocr_auto", {"file_path": "/nonexistent/file.png"})
        assert "Error: File not found" in result[0].text

    @pytest.mark.asyncio
    async def test_ocr_auto_unsupported_file(self):
        """Test ocr_auto tool with unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name

        try:
            result = await call_tool("ocr_auto", {"file_path": temp_file})
            assert "Unsupported file type" in result[0].text
        finally:
            os.unlink(temp_file)

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_auto_small_image(self, mock_env_free, temp_image_file, sample_ocr_response):
        """Test ocr_auto with small image (no compression needed)."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        result = await call_tool("ocr_auto", {"file_path": temp_image_file})

        text = result[0].text
        assert "Processing Summary" in text
        assert "fits within limit" in text
        assert "Hello World" in text

    @pytest.mark.asyncio
    @respx.mock
    async def test_ocr_auto_with_output(self, mock_env_free, temp_image_file, sample_ocr_response):
        """Test ocr_auto with output file saving."""
        respx.post(FREE_ENDPOINT).mock(return_value=httpx.Response(200, json=sample_ocr_response))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.txt"
            result = await call_tool(
                "ocr_auto",
                {
                    "file_path": temp_image_file,
                    "output_path": str(output_path),
                },
            )

            text = result[0].text
            assert "Saved to:" in text
            assert output_path.exists()
            content = output_path.read_text()
            assert "Hello World" in content
