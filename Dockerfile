FROM python:3.12.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["python", "-m", "src.server.server"]