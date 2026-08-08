FROM python:3.10-slim

WORKDIR /app

# Set environment variables to optimize RAM usage
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1

# Install system dependencies (build-essential for psycopg2)
RUN apk add --no-cache gcc musl-dev postgresql-dev libffi-dev || \
    apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies without cache to save disk space
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build time so it doesn't download on every startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('keepitreal/vietnamese-sbert')"

COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI with uvicorn (only 1 worker to save RAM)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
