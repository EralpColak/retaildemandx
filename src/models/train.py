# src/models/train.py
import os
import json
import pathlib
from math import sqrt
from typing import Dict, Iterable

import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
import joblib


# ---------------- MLflow & dizinler ---------------- #
MLFLOW_URI = "file:artifacts/mlflow"
EXPERIMENT_NAME = "retaildemandx"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

ART_DIR = pathlib.Path("artifacts")
MODELS_DIR = ART_DIR / "models"
ART_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- metrikler ----------------- #
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    diff = 2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-8)
    return float(np.mean(diff) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_pred - y_true)) /
                 np.maximum(np.sum(np.abs(y_true)), 1e-8) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(sqrt(mean_squared_error(y_true, y_pred)))


# ----------------- veri hazırlık ----------------- #
def load_features(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(["store_id", "sku", "ts"]).reset_index(drop=True)


def build_targets(df: pd.DataFrame, horizons: Iterable[int] = range(1, 25)) -> Dict[int, pd.Series]:
    """Her horizon için hedefi ileri kaydır (shift(-h)) -> y(t+h)."""
    targets: Dict[int, pd.Series] = {}
    grp = df.groupby(["store_id", "sku"])["sales"]
    for h in horizons:
        targets[h] = grp.shift(-h)
    return targets


def train_val_split_by_time(df: pd.DataFrame, val_days: int = 7):
    max_ts = df["ts"].max()
    val_start = max_ts - pd.Timedelta(days=val_days)
    train_idx = df["ts"] < val_start
    val_idx = df["ts"] >= val_start
    # numpy maskelere çevir
    return train_idx.values, val_idx.values, val_start


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Girdi kolonlarını seç: hedef ve meta sütunları dışarıda bırak."""
    drop_cols = {"ts", "date", "sales", "doy"}  # varsa
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# ----------------- eğitim ----------------- #
def fit_one_horizon(
    h: int,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df_original: pd.DataFrame,
) -> dict:
    """Tek bir ufuk (h) için modeli eğit, metrikleri hesapla, artefaktları yaz."""
    # hedefi olmayan satırları at
    valid_rows = ~y.isna().values
    X_h = X.loc[valid_rows]
    y_h = y.loc[valid_rows].values
    tr = train_idx[valid_rows]
    va = val_idx[valid_rows]

    n_tr = int(np.sum(tr))
    n_va = int(np.sum(va))

    # train tamamen boşsa bu horizon'u atla (smoke için güvenli)
    if n_tr == 0:
        print(f"[WARN] h={h}: train split is empty after masks; skipping.")
        return {"rmse": float("nan"), "smape": float("nan"), "wape": float("nan")}

    # kolon seçimi
    cat_cols = [c for c in ["store_id", "sku"] if c in X_h.columns]
    num_cols = [c for c in X_h.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # az örnekte fallback, normalde LGBM
    if n_tr < 5:
        print(f"[WARN] h={h}: train samples={n_tr} (<5). Using DummyRegressor fallback.")
        estimator = DummyRegressor(strategy="mean")
    else:
        estimator = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    pipe = Pipeline(steps=[("pre", pre), ("est", estimator)])

    # ---- fit ---- #
    pipe.fit(X_h.iloc[tr], y_h[tr])

    # ---- metrikler ---- #
    if n_va > 0:
        preds = pipe.predict(X_h.iloc[va])
        m = {
            "rmse": rmse(y_h[va], preds),
            "smape": smape(y_h[va], preds),
            "wape": wape(y_h[va], preds),
        }
    else:
        preds = np.array([])
        m = {"rmse": float("nan"), "smape": float("nan"), "wape": float("nan")}

    # ---- artefaktlar ---- #
    out_dir = MODELS_DIR / f"h{h:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.pkl")

    # validasyon tahminleri (varsa) yaz
    if n_va > 0:
        idx_va = X_h.iloc[va].index
        meta = df_original.loc[idx_va, ["ts", "store_id", "sku"]]
        val_df = (
            pd.DataFrame({"h": h, "y_true": y_h[va], "y_pred": preds}, index=idx_va)
            .join(meta)
            [["h", "ts", "store_id", "sku", "y_true", "y_pred"]]
        )
        val_df.to_csv(out_dir / "val_preds.csv", index=False)

    # ---- MLflow log ---- #
    with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
        mlflow.log_metric("rmse", m["rmse"])
        mlflow.log_metric("smape", m["smape"])
        mlflow.log_metric("wape", m["wape"])
        mlflow.log_artifact(str(out_dir / "model.pkl"))
        if n_va > 0:
            mlflow.log_artifact(str(out_dir / "val_preds.csv"))
        # Sklearn model log (LightGBM pipeline'ı) — başarısız olursa sessiz geç
        try:
            import mlflow.sklearn as mlfs  # lazy import
            mlfs.log_model(pipe, artifact_path="model")
        except Exception:
            pass

    return m


def main(
    features_path: str = "data/processed/features.parquet",
    horizons: Iterable[int] = range(1, 25),
    val_days: int = 7,
):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features(features_path)
    X = select_features(df)
    targets = build_targets(df, horizons=horizons)
    train_idx, val_idx, val_start = train_val_split_by_time(df, val_days=val_days)

    agg_metrics: Dict[int, dict] = {}

    with mlflow.start_run(run_name=f"lgbm_direct_h{min(horizons)}-{max(horizons)}"):
        mlflow.log_param("horizons", list(horizons))
        mlflow.log_param("val_days", val_days)
        mlflow.log_param("model", "LGBMRegressor|DummyRegressor(fallback)")
        mlflow.log_param("preprocess", "OneHot + passthrough")

        for h in horizons:
            m = fit_one_horizon(h, X, targets[h], train_idx, val_idx, df_original=df)
            agg_metrics[h] = m

        # özet metrikler (NaN'leri es geçerek ortalama)
        def _nanmean(xs):
            arr = np.array(xs, dtype=float)
            return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

        rmse_mean = _nanmean([v["rmse"] for v in agg_metrics.values()])
        smape_mean = _nanmean([v["smape"] for v in agg_metrics.values()])
        wape_mean = _nanmean([v["wape"] for v in agg_metrics.values()])

        mlflow.log_metric("rmse_mean_h1_24", rmse_mean)
        mlflow.log_metric("smape_mean_h1_24", smape_mean)
        mlflow.log_metric("wape_mean_h1_24", wape_mean)

        metrics_path = ART_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "rmse_mean_h1_24": rmse_mean,
                    "smape_mean_h1_24": smape_mean,
                    "wape_mean_h1_24": wape_mean,
                },
                f,
                indent=2,
            )

        print(
            f"Train done. Val start: {val_start}. "
            f"mean RMSE={rmse_mean:.3f}, sMAPE={smape_mean:.2f}%, WAPE={wape_mean:.2f}%"
        )


if __name__ == "__main__":
    main()