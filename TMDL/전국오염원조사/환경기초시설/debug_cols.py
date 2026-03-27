import pandas as pd
from pathlib import Path
import glob

data_dir = Path("C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터")
xls_files = glob.glob(str(data_dir / "*.xls*"))

with open("debug_columns_after_dropna.txt", "w", encoding="utf-8") as f:
    for file in xls_files:
        f.write(f"File: {Path(file).name}\n")
        temp_df = pd.read_excel(file, sheet_name="방류량", skiprows=3, header=None)
        f.write(f"  Cols before dropna: {temp_df.shape[1]}\n")
        temp_df = temp_df.dropna(axis=1, how='all')
        f.write(f"  Cols after dropna: {temp_df.shape[1]}\n")
        f.write(f"  First 5 values of index 0: {temp_df.iloc[:5, 0].tolist()}\n")
        f.write("-" * 50 + "\n")
