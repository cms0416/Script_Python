import pandas as pd
from pathlib import Path

file_path = Path("C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터/2024년기준_전국오염원_조사자료_환경기초시설_가확정_250721.xlsx")

with open("debug_out_header.txt", "w", encoding="utf-8") as f:
    f.write("Script started. Checking if file exists: " + str(file_path.exists()) + "\n")
    f.flush()
    try:
        f.write("Calling read_excel...\n")
        f.flush()
        df = pd.read_excel(file_path, sheet_name="방류량", nrows=5)
        f.write("read_excel finished.\n")
        f.flush()
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Row 1: {df.iloc[0].tolist()}\n")
        f.write(f"Row 2: {df.iloc[1].tolist()}\n")
        f.write(f"Row 3: {df.iloc[2].tolist()}\n")
        f.write(f"Row 4: {df.iloc[3].tolist()}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
