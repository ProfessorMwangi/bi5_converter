import polars as pl
df = pl.read_parquet("../data/training/EURUSD_3s.parquet")
print(df)
