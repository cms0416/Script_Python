import pandas as pd
from pathlib import Path

file_path = Path("C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터/250721_환경기초시설_입력검증_강원특별자치도_철원군.xlsx")

with open("debug_out.txt", "w", encoding="utf-8") as f:
    try:
        df = pd.read_excel(file_path, sheet_name="방류량", skiprows=3, header=None)
        f.write(f"Shape: {df.shape}\n")
        if df.shape[1] > 13:
            bod_col = df[9]
            f.write(f"BOD col index 9: na_count={bod_col.isna().sum()}\n")
            bod_num = pd.to_numeric(bod_col, errors='coerce')
            f.write(f"BOD numeric > 0: {(bod_num > 0).sum()}\n")
            
            f.write("First 10 BOD values:\n")
            for val in bod_col.head(10):
                f.write(f"{val}\n")
        else:
            f.write("Not enough cols\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
