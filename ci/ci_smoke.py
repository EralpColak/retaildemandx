import pathlib
import json
import os

os.environ.setdefault("PYTHONPATH", ".") 

from src.data import ingest_synthetic
from src.data import validate as validate_mod
from src.features import build_features
from src.models import train, evaluate, register


def _call_best(mod, *args, **kwargs):
    for name in ("main", "run"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"{mod.__name__} içinde main/run bulunamadı")

def run():
    pathlib.Path("artifacts").mkdir(exist_ok=True, parents=True)

    # 1) Ingest (küçük veri)
    ingest_synthetic.main(
        out_dir="data/raw",
        start="2024-01-01",
        end="2024-01-07",
        n_stores=2,
        n_skus=2,
        
    )

    # 2) Validate
    _call_best(validate_mod)

    # 3) Features
    build_features.main()

    # 4) Train (küçük horizon)
    train.main(features_path="data/processed/features.parquet", horizons=range(1, 5))

    # 5) Evaluate
    evaluate.main()

    # 6) Register
    register.main()

    # Smoke kontrolü
    summary = json.load(open("artifacts/reports/summary.json"))
    assert pathlib.Path("artifacts/registry/staging/current/manifest.json").exists()
    assert pathlib.Path("artifacts/reports/metrics_per_h.png").exists()
    assert pathlib.Path("artifacts/reports/report.html").exists()
    print("✅ Smoke OK. Summary:", summary)


if __name__ == "__main__":
    run()