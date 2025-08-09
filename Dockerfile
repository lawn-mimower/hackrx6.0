# ---------- Builder ----------
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force pre-bake model into /opt/models
RUN mkdir -p /opt/models/sentence-transformers/all-MiniLM-L6-v2
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
import os
model_name = "sentence-transformers/all-MiniLM-L6-v2"
out_dir = f"/opt/models/{model_name}"
SentenceTransformer(model_name).save(out_dir)
print("Pre-baked model saved to", out_dir)
PY

# ---------- Final ----------
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Copy pre-baked model into /app/models
COPY --from=builder /opt/models /app/models

# Create cache directory
RUN mkdir -p /app/cache/hf

# Set environment variables
ENV HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    EMBEDDING_MODEL=./models/sentence-transformers/all-MiniLM-L6-v2 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
