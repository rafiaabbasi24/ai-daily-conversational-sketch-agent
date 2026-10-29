# Conversational Sketch Artist: Text-to-Illustration Agent

## Project Overview
**Conversational Sketch Artist** is an end‑to‑end generative AI system that turns natural‑language prompts into high‑quality illustrations. Users interact via a chat‑style UI (Streamlit) while a FastAPI backend orchestrates prompt handling, image generation, and optional refinement. The core model leverages diffusion pipelines from 🤗 `diffusers` and language understanding from `transformers`.

Key capabilities:
- Real‑time text‑to‑image generation with configurable diffusion models.
- Conversational context retention (optional chat history).
- Fast, asynchronous API serving for scalable deployments.
- Simple, extensible architecture for swapping models or adding post‑processing steps.

## Architecture Diagram
```
+-------------------+        +-------------------+        +-------------------+
|   Streamlit UI    | <----> |   FastAPI Server  | <----> |  Diffusion Model  |
| (chat & canvas)   |        | (prompt routing)  |        | (diffusers)       |
+-------------------+        +-------------------+        +-------------------+
```

## Tech Stack
- **Python 3.10+**
- **FastAPI** – high‑performance async web framework
- **Streamlit** – interactive UI for chat and image display
- **PyTorch** – tensor computations & model execution
- **diffusers** – diffusion pipelines (Stable Diffusion, etc.)
- **transformers** – text encoders & optional CLIP guidance
- **uvicorn** – ASGI server for FastAPI
- **python‑dotenv** – environment variable management

## Quick Start

### Prerequisites
- Python 3.10 or newer
- Git
- An NVIDIA GPU with CUDA 11.8+ (recommended for reasonable generation times)
- `conda` or `virtualenv` (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/conversational-sketch-artist.git
cd conversational-sketch-artist

# Create a virtual environment (using venv)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root with the following keys (adjust values as needed):

```
# Model identifiers (can be any diffusers checkpoint)
DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
TEXT_ENCODER_MODEL=openai/clip-vit-large-patch14

# Generation parameters
GENERATION_STEPS=50
GUIDANCE_SCALE=7.5
SEED=42   # optional, for reproducibility

# Server settings
HOST=0.0.0.0
PORT=8000
```

### Running the Backend (FastAPI)

```bash
# Start the API server
uvicorn app.main:app --host $HOST --port $PORT --reload
```

The API will be reachable at `http://localhost:8000`. Swagger UI is available at `http://localhost:8000/docs`.

### Running the Frontend (Streamlit)

```bash
# In a new terminal (same venv)
streamlit run ui/app.py
```

Open the provided URL (typically `http://localhost:8501`) to start chatting with the Sketch Artist.

## API Reference

### POST `/generate`
Generate an illustration from a textual prompt.

**Request JSON**
```json
{
  "prompt": "A whimsical steampunk city at sunset",
  "negative_prompt": "low quality, blurry",
  "seed": 12345,
  "steps": 60,
  "guidance_scale": 8.0
}
```

**Response JSON**
```json
{
  "image_base64": "<base64‑encoded PNG>",
  "metadata": {
    "prompt": "...",
    "seed": 12345,
    "steps": 60,
    "guidance_scale": 8.0,
    "model": "runwayml/stable-diffusion-v1-5"
  }
}
```

### GET `/health`
Simple health‑check endpoint returning `{"status":"ok"}`.

## Usage Example (Python client)

```python
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://localhost:8000/generate"
payload = {
    "prompt": "A cyberpunk fox wearing sunglasses",
    "steps": 45,
    "guidance_scale": 7.0,
    "seed": 2024
}

response = requests.post(API_URL, json=payload)
data = response.json()
img_bytes = base64.b64decode(data["image_base64"])
image = Image.open(BytesIO(img_bytes))
image.show()
```

## Testing

The repository includes a pytest suite covering core generation logic and API endpoints.

```bash
# Install test dependencies
pip install pytest httpx

# Run the tests
pytest -vv
```

All tests should pass on a machine with the required GPU resources. CI pipelines are configured to run tests in a CPU‑only environment using a lightweight diffusion checkpoint.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Install development dependencies: `pip install -r requirements-dev.txt`.
4. Ensure code style passes `ruff` and type checks with `mypy`.
5. Write tests for new functionality.
6. Submit a pull request with a clear description and reference to any related issues.

## License

This project is licensed under the **Apache License 2.0**. See `LICENSE` for the full text.

## Acknowledgements

- The 🤗 `diffusers` and `transformers` libraries for providing open‑source model implementations.
- The FastAPI and Streamlit communities for excellent documentation and examples.
- All contributors who helped shape the Conversational Sketch Artist.