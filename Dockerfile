# Use a slim Python 3.12 image as the base
FROM python:3.12-slim-bookworm

# Install 'uv' for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml .
# Copy lockfile as well
COPY uv.lock* .

# Install dependencies using uv
# --no-install-project ensures we only install dependencies first
RUN uv sync --frozen --no-install-project --no-dev

# Copy the source code and other necessary files
COPY src/ src/
COPY mcp_registry.json .

# Install the project itself
RUN uv sync --frozen --no-dev

# Set environment variables for the orchestrator
ENV PATH="/app/.venv/bin:$PATH"

# Expose the orchestrator port
EXPOSE 8000

# Command to run the orchestrator using uvicorn
CMD ["uv", "run", "uvicorn", "src.server.server:app", "--host", "0.0.0.0", "--port", "8000"]
