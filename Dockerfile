# DisasterResponseEnv — Dockerfile (root level as required by OpenEnv spec)
# Build:  docker build -t disaster-response .
# Run:    docker run -p 8000:8000 disaster-response

FROM python:3.11-slim

# Required by OpenEnv spec — enables Gradio web interface
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY server/ ./server/
COPY openenv.yaml .
COPY README.md .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
