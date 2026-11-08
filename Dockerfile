FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml .
RUN uv pip install --system --no-cache "fastapi[standard]" httpx pydantic numpy pytest pytest-asyncio uvicorn
COPY src/ src/
COPY tests/ tests/
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
