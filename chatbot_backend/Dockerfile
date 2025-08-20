# syntax=docker/dockerfile:1

# Use a recent Python slim image
FROM python:3.11-slim

# Set environment to avoid pyc and ensure output is unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required by pandas, openpyxl, pdfminer.six, and general build tools
# We keep build tools in a single layer and remove apt lists to keep image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2 \
    libxslt1.1 \
    libxslt1-dev \
    libjpeg62-turbo \
    zlib1g \
    zlib1g-dev \
    libopenblas0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip tooling and install python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
# Avoid copying local DB or virtualenv; we'll use a .dockerignore, but add guards here too.
COPY src /app/src
COPY interfaces /app/interfaces

# Create a non-root user for runtime
RUN useradd -m appuser
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Default command: start uvicorn server
# Note: Bind to 0.0.0.0 for container networking
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
