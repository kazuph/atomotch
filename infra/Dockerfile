FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY miotts_server/ miotts_server/
COPY presets/ presets/
COPY scripts/ scripts/
COPY run_server.py run_gradio.py stt_server.py LICENSE README.md ./

# Install Python 3.12 and all dependencies
# Then add stt_server extras: google-genai + faster-whisper
RUN uv python install 3.12 && uv sync --python 3.12 && uv pip install google-genai faster-whisper

EXPOSE 8001

CMD ["uv", "run", "python", "run_server.py", "--host", "0.0.0.0"]
