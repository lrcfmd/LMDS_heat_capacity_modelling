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
RUN groupadd -r lmds && useradd -r -g lmds -u 1000 -m -d /home/lmds lmds
WORKDIR /app
COPY --from=builder /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl && rm -f /app/*.whl

# The platform runs the container as uid 1000. The runner creates /data/input
# (file inputs) and /data/output (results), so /app and all of /data (and
# /home/lmds) must be writable by that user.
RUN mkdir -p /data/input /data/output && chown -R 1000:1000 /app /data /home/lmds
USER 1000:1000
ENTRYPOINT ["model-run"]
