# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# Stage 1: Build wheel using uv
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv build --wheel

# Stage 2: Minimal runtime image
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app
COPY --from=builder /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl

RUN mkdir -p /data/output /home/lmds \
    && groupadd lmds \
    && useradd -g lmds -u 1000 -m -d /home/lmds lmds \
    && chown -R 1000:1000 /app /data/output /home/lmds

USER 1000:1000

ENTRYPOINT ["model-run"]
