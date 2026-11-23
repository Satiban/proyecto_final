FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 1. Dependencias fijadas primero (aprovecha la caché de Docker)
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r ./api/requirements.txt

# 2. Pipeline compartido (importado, no copiado) y código del servicio
COPY src/__init__.py ./src/__init__.py
COPY src/pipeline.py ./src/pipeline.py
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]