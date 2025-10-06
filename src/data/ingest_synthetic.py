import numpy as np, pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import argparse, pathlib
import holidays

def make_calendar(start="2024-01-01", end="2024-12-31", tz=None):
    idx = pd.date_range(start, end, freq="h", tz=tz)  # 'h'
    cal = pd.DataFrame({"ts": idx})
    cal["date"] = cal["ts"].dt.date
    cal["dow"]  = cal["ts"].dt.dayofweek
    cal["hour"] = cal["ts"].dt.hour
    cal["month"]= cal["ts"].dt.month
    cal["is_weekend"] = cal["dow"] >= 5
    cal["sin_hour"] = np.sin(2*np.pi*cal["hour"]/24)
    cal["cos_hour"] = np.cos(2*np.pi*cal["hour"]/24)
    cal["doy"] = cal["ts"].dt.day_of_year
    cal["sin_doy"] = np.sin(2*np.pi*cal["doy"]/365)
    cal["cos_doy"] = np.cos(2*np.pi*cal["doy"]/365)
    it_holidays = holidays.IT(years=[2024])
    cal["is_holiday"] = cal["ts"].dt.date.map(lambda d: d in it_holidays)
    return cal

def make_weather(cal_df, seed=42):
    rng = np.random.default_rng(seed)
    w = cal_df[["ts"]].copy()
    base_temp = 12 + 10*cal_df["sin_doy"] + rng.normal(0, 2, size=len(cal_df))
    rain_prob = 0.15 + 0.1*(1 - cal_df["sin_doy"])
    rain = rng.binomial(1, np.clip(rain_prob, 0.05, 0.5))
    w["temp_c"] = base_temp - 3*rain
    w["rain"] = rain
    return w

def demand_process(cal_df, weather_df, n_stores=5, n_skus=3, seed=7):
    rng = np.random.default_rng(seed)
    stores = [f"S{i+1}" for i in range(n_stores)]
    skus   = [f"SKU{j+1}" for j in range(n_skus)]
    base_store = rng.uniform(0.8, 1.2, size=n_stores)
    base_sku   = rng.uniform(0.6, 1.4, size=n_skus)

    df = cal_df.join(weather_df.drop(columns="ts"))
    out_rows = []
    for si, s in enumerate(stores):
        for kj, k in enumerate(skus):
            season_day = 1.0 + 0.5*df["sin_hour"] + 0.3*(df["hour"].between(18,22))
            weekend_boost = 1.0 + 0.25*df["is_weekend"]
            holiday_effect = 1.0 + 0.35*df["is_holiday"]
            temp_effect = 1.0 + 0.02*(df["temp_c"] - 15)
            rain_effect = 1.0 - 0.10*df["rain"]

            lam = (2.0 * base_store[si] * base_sku[kj] *
                   season_day * weekend_boost * holiday_effect *
                   temp_effect * rain_effect)
            y = rng.poisson(np.clip(lam, 0.05, None))
            tmp = df[["ts"]].copy()
            tmp["store_id"]= s
            tmp["sku"]     = k
            tmp["sales"]   = y
            out_rows.append(tmp)

    sales = pd.concat(out_rows, ignore_index=True)
    return sales

def add_promotions(sales_df, seed=13):
    rng = np.random.default_rng(seed)
    sales_df = sales_df.copy()
    sales_df["promo"] = 0
    for (s, k), grp in sales_df.groupby(["store_id","sku"]):
        idx = grp.sample(frac=0.05, random_state=rng.integers(0, 1e9)).index
        sales_df.loc[idx, "promo"] = 1
    return sales_df

def write_parquet(df, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def main(out_dir="data/raw", start="2024-01-01", end="2024-03-31",
         n_stores=5, n_skus=3):
    cal = make_calendar(start, end)
    weather = make_weather(cal)
    sales = demand_process(cal, weather, n_stores=n_stores, n_skus=n_skus)
    sales = sales.merge(weather, on="ts")
    sales = sales.merge(cal.drop(columns=["doy","sin_doy","cos_doy"]), on="ts")
    sales = add_promotions(sales)
    write_parquet(sales, f"{out_dir}/sales.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-03-31")
    parser.add_argument("--n_stores", type=int, default=5)
    parser.add_argument("--n_skus", type=int, default=3)
    args = parser.parse_args()
    main(**vars(args))