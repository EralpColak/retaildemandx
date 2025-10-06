
import pathlib
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Dict, Optional

REG_CURRENT = pathlib.Path("artifacts/registry/staging/current")
FEATURES_PATH = pathlib.Path("data/processed/features.parquet")

app = FastAPI(title="RetailDemandX API", version="0.1")

# ----- model yükleyici ----- #
def load_models(reg_dir: pathlib.Path) -> Dict[int, object]:
    if not reg_dir.exists():
        raise FileNotFoundError(f"Registry not found: {reg_dir}")
    models: Dict[int, object] = {}
    for hdir in sorted(reg_dir.glob("h*/")):
        try:
            h = int(hdir.name.replace("h", ""))
        except Exception:
            continue
        pkl = hdir / "model.pkl"
        if pkl.exists():
            models[h] = joblib.load(pkl)
    if not models:
        raise RuntimeError("No models found in registry.")
    return models


MODELS: Dict[int, object] = {}
HORIZONS = []

@app.on_event("startup")
def _startup_load():
    global MODELS, HORIZONS
    MODELS = load_models(REG_CURRENT)
    HORIZONS = sorted(MODELS.keys())

# ----- Yardımcılar ----- #
def _select_feature_row(store_id: str, sku: str, ts_str: Optional[str]):
    """Features dosyasından ilgili son satırı seç ve X döndür."""
    if not FEATURES_PATH.exists():
        raise HTTPException(500, f"Features file not found: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH).sort_values(["store_id", "sku", "ts"])
    df = df[(df["store_id"] == store_id) & (df["sku"] == sku)]
    if df.empty:
        raise HTTPException(404, f"No rows for store_id={store_id}, sku={sku}")
    if ts_str:
        ts = pd.Timestamp(ts_str)
        df = df[df["ts"] <= ts]
        if df.empty:
            raise HTTPException(404, f"No past rows before ts={ts_str} for {store_id}/{sku}")
    row = df.iloc[-1]
    X = df.drop(columns=[c for c in ["ts", "date", "sales", "doy"] if c in df.columns]).iloc[[-1]]
    return row, X

# ----- Endpoint'ler ----- #
@app.get("/health")
def health():
    return {"status": "ok", "models": len(MODELS), "horizons": HORIZONS}

@app.post("/reload")
def reload_models():
    _startup_load()
    return {"ok": True, "models": len(MODELS), "horizons": HORIZONS}

@app.get("/predict")
def predict(
    store_id: str = Query(...),
    sku: str = Query(...),
    ts: Optional[str] = Query(None, description="ISO datetime string, örn: 2024-03-30T12:00:00"),
    horizons: int = Query(24, ge=1, le=24),
):
    if not MODELS:
        _startup_load()
    row, X = _select_feature_row(store_id, sku, ts)
    preds = {}
    for h in range(1, horizons + 1):
        model = MODELS.get(h)
        if model is None:
            continue
        preds[f"h{h}"] = float(model.predict(X)[0])
    return {
        "predictions": preds,
        "meta": {
            "store_id": store_id,
            "sku": sku,
            "base_ts": str(row["ts"]),
            "used_row_index": int(X.index[0]),
            "horizons_served": HORIZONS,
        },
    }

