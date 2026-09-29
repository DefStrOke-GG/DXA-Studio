FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-geometry.txt /tmp/requirements-geometry.txt
COPY service/requirements.txt /tmp/requirements-service.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
       torch==2.12.1+cu130 torchvision==0.27.1+cu130 \
    && python -m pip install -r /tmp/requirements-geometry.txt -r /tmp/requirements-service.txt

EXPOSE 8765

CMD ["python", "-X", "utf8", "-m", "dxa_project.service", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]
