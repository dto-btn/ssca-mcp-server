FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml .
COPY src/ ./src/

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["python", "-m", "src.server.server"]