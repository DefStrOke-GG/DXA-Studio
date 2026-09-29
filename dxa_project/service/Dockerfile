FROM python:3.13-slim

ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1

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
    && case "$TORCH_VARIANT" in cpu|cu130) ;; *) echo "Unsupported TORCH_VARIANT=$TORCH_VARIANT (expected cpu or cu130)" >&2; exit 2 ;; esac \
    && python -m pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
       "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip install -r /tmp/requirements-geometry.txt -r /tmp/requirements-service.txt

EXPOSE 8765

CMD ["python", "-X", "utf8", "-m", "dxa_project.service", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]
