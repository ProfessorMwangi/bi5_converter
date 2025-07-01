import polars as pl
df = pl.read_parquet(
    "../data/output/EURUSD/EURUSD_20250623_23h_ticks.parquet")
print(df)
