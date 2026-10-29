"""tests/test_main.py
Unit tests for the Conversational Sketch Artist FastAPI application.

These tests verify the core functionality of the API endpoints and the
`generate_image` utility function without invoking heavy model loading.
The diffusers pipeline is monkey‑patched with a lightweight mock that returns
a dummy Pillow image, ensuring the tests run quickly in CI environments.

The test suite covers:
* Successful image generation via the `/generate` endpoint.
* Validation errors for missing or empty prompts.
* Direct invocation of the `generate_image` function.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import MagicMock, patch

# Import the FastAPI app and the public generate_image function.
# The project structure is expected to expose these symbols from `app.main`.
from app.main import app, generate_image  # type: ignore

# Create a TestClient instance for HTTP request testing.
client = TestClient(app)


def _create_dummy_image() -> Image.Image:
    """Return a simple 1x1 white PNG image for testing purposes."""
    img = Image.new("RGB", (1, 1), color="white")
    return img


def _mock_pipeline(prompt: str, **kwargs: Any) -> Image.Image:
    """
    Mock replacement for the diffusers pipeline.

    Args:
        prompt: The text prompt passed to the pipeline (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A dummy Pillow image.
    """
    return _create_dummy_image()


@pytest.fixture(autouse=True)
def mock_diffusers_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically replace the heavy diffusers pipeline with a lightweight mock
    for the duration of each test.

    The actual implementation in `app.main` is expected to import
    `StableDiffusionPipeline` (or similar) from `diffusers`. We replace the
    `from_pretrained` class method to return an object whose `__call__`
    returns a dummy image.
    """
    # Create a mock pipeline object whose __call__ returns a dummy image.
    mock_pipe = MagicMock()
    mock_pipe.__call__.side_effect = _mock_pipeline

    # Patch the class method that loads the model.
    with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
        yield


def test_generate_image_function_returns_image() -> None:
    """
    Verify that the public `generate_image` function returns a Pillow Image
    instance and that the image content matches the dummy image produced by
    the mocked pipeline.
    """
    prompt = "A cat wearing a hat"
    result = generate_image(prompt)

    assert isinstance(result, Image.Image), "generate_image should return a Pillow Image"
    # The dummy image is 1x1 white; check size and pixel value.
    assert result.size == (1, 1), "Dummy image should be 1x1 pixels"
    assert result.getpixel((0, 0)) == (255, 255, 255), "Pixel should be white (255,255,255)"


def test_generate_endpoint_success() -> None:
    """
    Test the `/generate` POST endpoint with a valid prompt.
    The response should be JSON containing a base64‑encoded image string.
    """
    response = client.post("/generate", json={"prompt": "A futuristic city skyline at sunset"})
    assert response.status_code == 200, f"Unexpected status: {response.text}"
    data = response.json()
    assert "image_base64" in data, "Response JSON must contain 'image_base64' key"

    # Decode the base64 string and verify it yields a valid image.
    image_bytes = io.BytesIO()
    image_bytes.write(io.BytesIO(bytes.fromhex(data["image_base64"])).getvalue())
    image_bytes.seek(0)
    img = Image.open(image_bytes)
    assert isinstance(img, Image.Image), "Decoded data should be a Pillow Image"
    assert img.size == (1, 1), "Decoded image should match dummy size"


def test_generate_endpoint_missing_prompt() -> None:
    """
    Ensure the API returns a 422 validation error when the prompt field is omitted.
    """
    response = client.post("/generate", json={})
    assert response.status_code == 422
    error_detail = response.json().get("detail", [])
    assert any(err["loc"] == ["body", "prompt"] for err in error_detail), "Missing prompt should be reported"


def test_generate_endpoint_empty_prompt() -> None:
    """
    Verify that an empty string prompt is treated as invalid and returns a 422.
    """
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422
    error_detail = response.json().get("detail", [])
    # The validation should flag the prompt as too short or empty.
    assert any(err["loc"] == ["body", "prompt"] for err in error_detail), "Empty prompt should be reported"


def test_generate_endpoint_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulate an unexpected exception inside `generate_image` and confirm the
    API returns a 500 error with an appropriate message.
    """
    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr("app.main.generate_image", explode)

    response = client.post("/generate", json={"prompt": "Anything"})
    assert response.status_code == 500
    json_body = response.json()
    assert json_body.get("detail") == "Internal Server Error", "Unexpected error message"