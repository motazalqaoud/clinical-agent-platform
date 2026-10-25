# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CLINAGENT_CORPUS=/app/corpus/guidelines \
    CLINAGENT_FHIR=/app/data/fhir/synthetic-bundle.json
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels \
    && useradd --uid 10001 --no-create-home appuser
COPY corpus ./corpus
COPY data/fhir ./data/fhir
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "clinagent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
