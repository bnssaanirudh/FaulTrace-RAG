FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy packages
COPY packages/ ./packages/
COPY apps/api/ ./apps/api/

# Install packages in editable mode
RUN pip install --no-cache-dir \
    -e packages/core \
    -e packages/data \
    -e packages/gold \
    -e packages/pipelines \
    -e packages/reporting \
    -e apps/api

# Create data and artifact directories
RUN mkdir -p data/generated artifacts/runs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Seed demo data on startup, then launch API
CMD ["sh", "-c", "python -m faulttrace_data.cli generate --scales 10,50,200,1000 --seed 42 --output-dir data/generated/worlds 2>/dev/null; python -m uvicorn faulttrace_api.main:app --host 0.0.0.0 --port 8000 --app-dir apps/api"]
