FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies needed for LightGBM and C++ builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, models, and configs
COPY config.yaml .
COPY src/ src/
COPY serving/ serving/
COPY models/ models/

EXPOSE 8000

# Health check to ensure API is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI serving server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
