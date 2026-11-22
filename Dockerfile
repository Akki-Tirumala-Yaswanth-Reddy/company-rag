FROM python:3.12-slim

# Logs go straight to stdout (Cloud Logging picks them up); no .pyc clutter
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. App code and data
COPY app/ app/
COPY scripts/ scripts/
COPY data/docs/ data/docs/
COPY chroma_db/ chroma_db/
RUN python scripts/seed_db.py

# 3. Don't run as root inside the container
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud Run tells the app which port to use via $PORT (8080 by default)
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]