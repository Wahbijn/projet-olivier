# Image de l'API (à construire depuis la racine du projet : voir docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch CPU (image bien plus légère que la version CUDA)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY configs/ configs/
COPY src/ src/
COPY api/ api/

# Le modèle n'est pas copié dans l'image : il est monté dans /app/models (volume),
# ce qui permet de changer de version sans reconstruire l'image.
RUN mkdir -p models && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
