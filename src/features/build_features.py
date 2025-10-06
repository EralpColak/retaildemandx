import pandas as pd
import numpy as np
import pathlib

def add_lags(df, group_cols, target_col="sales", lags=[1, 2, 3, 6, 12, 24]):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = (
            df.groupby(group_cols)[target_col].shift(lag)
        )
    return df

def add_rolling(df, group_cols, target_col="sales", windows=[3, 6, 12, 24]):
    for w in windows:
        df[f"{target_col}_rollmean{w}"] = (
            df.groupby(group_cols)[target_col].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        )
        df[f"{target_col}_rollstd{w}"] = (
            df.groupby(group_cols)[target_col].shift(1).rolling(w).std().reset_index(level=0, drop=True)
        )
    return df

def build_features(in_path="data/raw/sales.parquet", out_path="data/processed/features.parquet"):
    df = pd.read_parquet(in_path)

    df = df.sort_values(["store_id", "sku", "ts"])

    df = add_lags(df, ["store_id", "sku"])
    df = add_rolling(df, ["store_id", "sku"])

    
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

   
    df["promo_x_holiday"] = df["promo"] * df["is_holiday"].astype(int)
    df["promo_x_weekend"] = df["promo"] * df["is_weekend"].astype(int)

    df = df.dropna()

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ Features saved to {out_path} — shape: {df.shape}")
    return df

if __name__ == "__main__":
    build_features()