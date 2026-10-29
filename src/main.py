import argparse
import base64
import io
import logging
import os
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from diffusers import StableDiffusionPipeline
import uvicorn

# Streamlit imports are optional and only used when running the UI.
# Import lazily to avoid unnecessary dependencies when only the API is needed.
try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

# ------------------------------------------------------------------------------
# Configuration & Logging
# ------------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Schema for a generation request."""
    prompt: str = Field(..., description="Text prompt describing the desired illustration.")
    negative_prompt: str | None = Field(
        None,
        description="Optional negative prompt to steer the model away from unwanted concepts.",
    )
    num_inference_steps: int = Field(
        50,
        ge=1,
        le=150,
        description="Number of denoising steps. Higher values improve quality at the cost of speed.",
    )
    guidance_scale: float = Field(
        7.5,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale. Higher values follow the prompt more closely.",
    )
    seed: int | None = Field(
        None,
        description="Random seed for reproducibility. If omitted, a random seed is used.",
    )


class GenerateResponse(BaseModel):
    """Schema for a generation response."""
    image_base64: str = Field(..., description="Base64‑encoded PNG image.")
    seed: int = Field(..., description="Seed used for generation.")
    model: str = Field(..., description="Model identifier used for generation.")


# ------------------------------------------------------------------------------
# Model handling
# ------------------------------------------------------------------------------

class ModelManager:
    """
    Singleton manager for the Stable Diffusion pipeline.
    Lazily loads the model on first use and caches it for subsequent calls.
    """

    _pipeline: StableDiffusionPipeline | None = None

    @classmethod
    def get_pipeline(cls) -> StableDiffusionPipeline:
        """Return a ready‑to‑use StableDiffusionPipeline instance."""
        if cls._pipeline is None:
            logger.info("Loading Stable Diffusion model %s on %s", DEFAULT_MODEL, DEVICE)
            try:
                cls._pipeline = StableDiffusionPipeline.from_pretrained(
                    DEFAULT_MODEL,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                )
                cls._pipeline = cls._pipeline.to(DEVICE)
                # Enable memory efficient attention if available
                if hasattr(cls._pipeline, "enable_xformers_memory_efficient_attention"):
                    cls._pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Model loaded successfully.")
            except Exception as exc:
                logger.exception("Failed to load the model.")
                raise RuntimeError("Could not load the Stable Diffusion model.") from exc
        return cls._pipeline


def generate_image(request: GenerateRequest) -> GenerateResponse:
    """
    Generate an illustration from a text prompt using Stable Diffusion.

    Args:
        request: Validated generation request.

    Returns:
        GenerateResponse containing a base64 PNG image and metadata.
    """
    pipeline = ModelManager.get_pipeline()

    generator = torch.Generator(device=DEVICE)
    seed = request.seed if request.seed is not None else torch.seed()
    generator.manual_seed(seed)

    logger.debug(
        "Generating image with prompt='%s', negative_prompt='%s', steps=%d, guidance=%.2f, seed=%d",
        request.prompt,
        request.negative_prompt,
        request.num_inference_steps,
        request.guidance_scale,
        seed,
    )

    try:
        output = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        image = output.images[0]
    except Exception as exc:
        logger.exception("Image generation failed.")
        raise RuntimeError("Failed to generate image.") from exc

    # Encode image to PNG and then to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return GenerateResponse(
        image_base64=img_b64,
        seed=seed,
        model=DEFAULT_MODEL,
    )


# ------------------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Conversational Sketch Artist API",
    description="Generate illustrations from textual prompts using Stable Diffusion.",
    version="1.0.0",
)

# Allow all origins for simplicity; tighten in production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
    """
    FastAPI endpoint that receives a prompt and returns a generated illustration.
    """
    try:
        return generate_image(request)
    except ValidationError as ve:
        logger.warning("Validation error: %s", ve)
        raise HTTPException(status_code=422, detail=ve.errors())
    except RuntimeError as re:
        logger.error("Generation error: %s", re)
        raise HTTPException(status_code=500, detail=str(re))


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": ModelManager._pipeline is not None}


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------

def _run_streamlit_ui() -> None:
    """
    Launch a minimal Streamlit interface that talks to the local generation function.
    This UI is only available when the `streamlit` package is installed.
    """
    if st is None:  # pragma: no cover
        raise ImportError("Streamlit is not installed. Install it to use the UI.")

    st.set_page_config(page_title="Conversational Sketch Artist", layout="centered")
    st.title("🖌️ Conversational Sketch Artist")
    st.markdown(
        "Enter a textual description and let the AI create a sketch for you. "
        "You can tweak generation parameters for more control."
    )

    with st.form(key="generation_form"):
        prompt = st.text_area("Prompt", height=100, help="Describe the illustration you want.")
        negative_prompt = st.text_input(
            "Negative Prompt (optional)",
            help="Concepts you want the model to avoid."
        )
        steps = st.slider("Inference Steps", min_value=1, max_value=150, value=50)
        guidance = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
        seed = st.number_input("Seed (optional)", min_value=0, step=1, value=0)
        submit = st.form_submit_button("Generate")

    if submit:
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            request = GenerateRequest(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed if seed != 0 else None,
            )
            with st.spinner("Generating illustration..."):
                try:
                    response = generate_image(request)
                    img_bytes = base64.b64decode(response.image_base64)
                    st.image(img_bytes, caption=f"Seed: {response.seed}", use_column_width=True)
                except Exception as exc:  # pragma: no cover
                    logger.exception("Failed to generate image via UI.")
                    st.error(f"Generation failed: {exc}")


# ------------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point. Supports two sub‑commands:

    * `api` – starts the FastAPI server with Uvicorn.
    * `ui`  – launches the Streamlit interface.
    """
    parser = argparse.ArgumentParser(
        description="Conversational Sketch Artist – Text‑to‑Illustration service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api", help="Run the FastAPI server.")
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind the API server.",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the API server.",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto‑reload (useful during development).",
    )

    ui_parser = subparsers.add_parser("ui", help="Run the Streamlit UI.")
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port on which Streamlit will serve the UI.",
    )
    ui_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode (no browser auto‑open).",
    )

    args = parser.parse_args()

    if args.command == "api":
        uvicorn.run(
            "src.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    elif args.command == "ui":
        if st is None:  # pragma: no cover
            raise ImportError("Streamlit is required to run the UI. Install with `pip install streamlit`.")
        # Streamlit expects the script path; we invoke it via the command line.
        # Build the command arguments programmatically.
        import subprocess
        cmd = [
            "streamlit",
            "run",
            __file__,
            "--server.port",
            str(args.port),
        ]
        if args.headless:
            cmd.append("--server.headless")
        # Use subprocess to hand off execution to Streamlit.
        subprocess.run(cmd, check=True)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
