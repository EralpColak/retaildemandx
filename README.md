# RetailDemandX — End-to-End Perakende Talep Tahmini (MLOps)

Saatlik mağaza×SKU satışları için uçtan uca bir tahmin pipeline’ı:
**veri → validasyon → özellik mühendisliği → çok-ufuklu eğitim → değerlendirme → model kayıt → servis & batch**  
MLOps: **DVC** (pipeline & veri/model izleme), **MLflow** (deney takip), **GitHub Actions** (CI), **Docker** (taşınabilir runtime).

![pipeline](artifacts/reports/metrics_per_h.png)

## Hızlı Başlangıç
```bash
# 1) Ortam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Pipeline (DVC)
dvc repro     # ingest → validate → features → train → evaluate → register

# 3) MLflow UI
mlflow ui --backend-store-uri "file:artifacts/mlflow"  # http://127.0.0.1:5000

# 4) API (lokal)
python -m uvicorn src.serving.app:app --reload --port 8000
# Test:
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/predict?store_id=S1&sku=SKU1&horizons=24"

# 5) Docker
docker build -t retaildemandx:latest .
docker run --rm -it -p 8000:8000 \
  -e MODE=api \
  -v "$(pwd)/artifacts/registry/staging/current:/app/artifacts/registry/staging/current:ro" \
  -v "$(pwd)/data/processed:/app/data/processed:ro" \
  retaildemandx:latest