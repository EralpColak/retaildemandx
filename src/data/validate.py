import pandera as pa
import pandas as pd

# Kolon şeması: tip + kurallar
schema = pa.DataFrameSchema(
    {
        "ts": pa.Column(pa.DateTime),                   # timestamp
        "store_id": pa.Column(str),
        "sku": pa.Column(str),
        "sales": pa.Column(pa.Int, checks=pa.Check.ge(0)),
        "temp_c": pa.Column(pa.Float),
        "rain": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
        "hour": pa.Column(pa.Int, checks=[pa.Check.ge(0), pa.Check.le(23)]),
        "month": pa.Column(pa.Int, checks=[pa.Check.ge(1), pa.Check.le(12)]),
        "is_weekend": pa.Column(bool),
        "sin_hour": pa.Column(pa.Float),
        "cos_hour": pa.Column(pa.Float),
        "is_holiday": pa.Column(bool),
        "date": pa.Column(object),  # pandas date (Python date objesi)
        "dow": pa.Column(pa.Int),   # day of week 0..6
        "promo": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
    },
    coerce=True,   # tipleri otomatik uyumla
)

def validate(path="data/raw/sales.parquet"):
    df = pd.read_parquet(path)
    schema.validate(df, lazy=True)  # tüm hataları topla
    print(f"OK: {path} passed schema checks with {len(df)} rows.")

if __name__ == "__main__":
    validate()