# src/pipelines/batch_forecast.py
import argparse, json, pathlib
import joblib
import pandas as pd
from typing import Dict, Optional

REG_CURRENT = pathlib.Path("artifacts/registry/staging/current")
FEATURES_PATH = pathlib.Path("data/processed/features.parquet")
OUT_DIR = pathlib.Path("artifacts/forecasts")

def load_models(reg_dir: pathlib.Path) -> Dict[int, object]:
    if not reg_dir.exists():
        raise FileNotFoundError(f"Registry not found: {reg_dir}")
    models: Dict[int, object] = {}
    for hdir in sorted(reg_dir.glob("h*/")):
        name = hdir.name  # e.g., h01
        try:
            h = int(name.replace("h", ""))
        except Exception:
            continue
        pkl = hdir / "model.pkl"
        if pkl.exists():
            models[h] = joblib.load(pkl)
    if not models:
        raise RuntimeError("No models found under registry.")
    return models

def select_last_rows(features_path: pathlib.Path, asof: Optional[str]) -> pd.DataFrame:
    df = pd.read_parquet(features_path).sort_values(["store_id", "sku", "ts"])
    if asof:
        ts = pd.Timestamp(asof)
        df = df[df["ts"] <= ts]
    # her store×sku için son kayıt
    idx = df.groupby(["store_id", "sku"])["ts"].idxmax()
    last_df = df.loc[idx].copy().sort_values(["store_id", "sku"]).reset_index(drop=True)
    return last_df

def make_X(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["ts", "date", "sales", "doy"] if c in df.columns]
    return df.drop(columns=drop_cols)

def run(asof: Optional[str], horizons: int, out_dir: pathlib.Path):
    models = load_models(REG_CURRENT)
    last_df = select_last_rows(FEATURES_PATH, asof)
    X = make_X(last_df)

    rows = []
    for i, row in last_df.iterrows():
        x_i = X.iloc[[i]]
        for h in range(1, horizons + 1):
            mdl = models.get(h)
            if mdl is None:
                continue
            yhat = float(mdl.predict(x_i)[0])
            rows.append({
                "store_id": row["store_id"],
                "sku": row["sku"],
                "base_ts": row["ts"],
                "h": h,
                "y_pred": yhat,
            })
    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["store_id", "sku", "h"]).reset_index(drop=True)

    # geniş form
    wide_df = long_df.pivot_table(index=["store_id", "sku", "base_ts"],
                                  columns="h", values="y_pred").reset_index()
    wide_df.columns = [f"h{c}" if isinstance(c, int) else c for c in wide_df.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "batch_forecast_long.csv", index=False)
    wide_df.to_csv(out_dir / "batch_forecast_wide.csv", index=False)

    manifest = {
        "asof": asof or "latest",
        "horizons": list(range(1, horizons + 1)),
        "n_series": int(wide_df.shape[0]),
        "registry": str(REG_CURRENT),
        "features": str(FEATURES_PATH),
        "outputs": ["batch_forecast_long.csv", "batch_forecast_wide.csv"],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Batch forecast done. Rows: {len(long_df)} | Series: {manifest['n_series']}")
    print(f"Saved to: {out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asof", type=str, default=None,
                   help="Bu zamana kadar olan son kayıtlar kullanılır (ISO datetime). Boş bırakılırsa en son.")
    p.add_argument("--horizons", type=int, default=24)
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(asof=args.asof, horizons=args.horizons, out_dir=pathlib.Path(args.out_dir))