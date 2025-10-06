
FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt ./requirements.txt


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY src ./src
COPY dvc.yaml ./dvc.yaml


RUN mkdir -p artifacts/registry/staging/current \
    artifacts/reports \
    artifacts/forecasts \
    data/processed

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


EXPOSE 8000

ENV MODE=api

ENTRYPOINT ["/entrypoint.sh"]