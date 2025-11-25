# Production Dockerfile - Architecture 2
# Multi-stage build for optimized image size

# ==============================================
# Stage 1: Base image with CUDA support
# ==============================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# ==============================================
# Stage 2: Dependencies
# ==============================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements
COPY requirements_v2.txt .

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other requirements
RUN pip3 install -r requirements_v2.txt --no-cache-dir

# ==============================================
# Stage 3: Application
# ==============================================
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY blood_cell_system.py .
COPY training_pipeline.py .
COPY inference_pipeline_v2.py .
COPY services/ services/
COPY config/ config/

# Create necessary directories
RUN mkdir -p checkpoints data results logs uploads

# Set permissions
RUN chmod +x services/api/main.py

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["python3", "-m", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ==============================================
# Stage 4: Worker (for Celery)
# ==============================================
FROM dependencies AS worker

WORKDIR /app

COPY blood_cell_system.py .
COPY inference_pipeline_v2.py .
COPY services/worker/ services/worker/
COPY config/ config/

RUN mkdir -p checkpoints data results logs

CMD ["celery", "-A", "services.worker.tasks", "worker", "--loglevel=info", "--concurrency=4"]
