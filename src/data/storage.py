import os, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from typing import Optional
from ..core.utils import ensure_dir

def parquet_path(base_dir: str, symbol: str, interval: str) -> str:
    d = ensure_dir(os.path.join(base_dir, "parquet"))
    fname = f"{symbol}_{interval}.parquet"
    return os.path.join(d, fname)

def save_bars(df: pd.DataFrame, path: str):
    # Expect columns: ['ts','open','high','low','close','volume']
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)

def load_bars(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    table = pq.read_table(path)
    return table.to_pandas()
