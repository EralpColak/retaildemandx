import pandera as pa
import pandas as pd


schema = pa.DataFrameSchema(
    {
        "ts": pa.Column(pa.DateTime),                  
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
        "date": pa.Column(object),  
        "dow": pa.Column(pa.Int),   
        "promo": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
    },
    coerce=True,   
)

def validate(path="data/raw/sales.parquet"):
    df = pd.read_parquet(path)
    schema.validate(df, lazy=True)  
    print(f"OK: {path} passed schema checks with {len(df)} rows.")

if __name__ == "__main__":
    validate()
def main(input_path: str = "data/raw/sales.parquet"):
    """
    CLI ve ci_smoke için basit giriş noktası.
    Gerekirse mevcut 'validate(...)' ya da benzer fonksiyonunu çağır.
    """
  
    try:
        validate(input_path)        
    except NameError:
        
        import pandas as pd
        from pandera import pandas as pa

        df = pd.read_parquet(input_path)

        SalesSchema.validate(df)   

    print("✅ validation ok")


if __name__ == "__main__":
    
    main()