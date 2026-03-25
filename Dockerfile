FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY mcp_registry.json ./mcp_registry.json

EXPOSE 9000

ENV ORCHESTRATOR_HOST=0.0.0.0 \
    ORCHESTRATOR_PORT=9000 \
    ORCHESTRATOR_ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,https://localhost:8080,https://127.0.0.1:8080,https://localhost:5173,https://127.0.0.1:5173

CMD ["sh", "-c", "uv run uvicorn src.server.server:app --host 0.0.0.0 --port ${PORT:-${ORCHESTRATOR_PORT:-9000}}"]
