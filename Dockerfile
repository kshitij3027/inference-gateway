FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS test
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-httpx pyyaml
COPY pyproject.toml .
COPY gateway/ gateway/
COPY config/ config/
COPY tests/ tests/
CMD ["python", "-m", "pytest", "tests/", "-v"]

FROM python:3.12-slim AS runtime
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY gateway/ gateway/
COPY config/ config/
EXPOSE 8080
USER appuser
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
