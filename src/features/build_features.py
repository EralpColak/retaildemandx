# src/features/build_features.py
import pandas as pd
import numpy as np
import pathlib
from typing import Sequence


def add_lags(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str = "sales",
    lags: Sequence[int] = (1, 2, 3, 6, 12, 24),
) -> pd.DataFrame:
    """Belirtilen gecikmeleri (lag) ekler."""
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def add_rolling(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str = "sales",
    windows: Sequence[int] = (3, 6, 12, 24),
) -> pd.DataFrame:
    """Kaydırmalı ortalama/std (window-based) özellikleri ekler (1 adım gecikmeyle kaçırtarak)."""
    grp = df.groupby(group_cols)[target_col]
    for w in windows:
        # bir adım gecikmeli rolling, target sızıntısını engeller
        roll_src = grp.shift(1).rolling(w)
        df[f"{target_col}_rollmean{w}"] = roll_src.mean().reset_index(level=0, drop=True)
        df[f"{target_col}_rollstd{w}"] = roll_src.std().reset_index(level=0, drop=True)
    return df


def build_features(
    in_path: str = "data/raw/sales.parquet",
    out_path: str = "data/processed/features.parquet",
) -> pd.DataFrame:
    """Ham satış verisinden modellemeye hazır öznitelikleri üretir ve kaydeder."""
    df = pd.read_parquet(in_path)

    # deterministik sıralama
    df = df.sort_values(["store_id", "sku", "ts"]).reset_index(drop=True)

    # geçmiş satış temelli öznitelikler
    df = add_lags(df, ["store_id", "sku"])
    df = add_rolling(df, ["store_id", "sku"])

    # takvim sinüs/kosinüs (haftanın günü)
    if "dow" in df.columns:
        df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # etkileşim terimleri
    if "promo" in df.columns and "is_holiday" in df.columns:
        df["promo_x_holiday"] = df["promo"] * df["is_holiday"].astype(int)
    if "promo" in df.columns and "is_weekend" in df.columns:
        df["promo_x_weekend"] = df["promo"] * df["is_weekend"].astype(int)

    # kayıp değerleri temizle (lag/rolling sonrası ön uç için güvenli)
    df = df.dropna().reset_index(drop=True)

    # yaz ve döndür
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ Features saved to {out_path} — shape: {df.shape}")
    return df


def main(
    in_path: str = "data/raw/sales.parquet",
    out_path: str = "data/processed/features.parquet",
) -> None:
    """CI smoke veya komut satırı için giriş noktası."""
    build_features(in_path=in_path, out_path=out_path)


if __name__ == "__main__":
    main()