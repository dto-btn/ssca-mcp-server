FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY src/ ./src/

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["python", "-m", "src.server.server"]