import pandas as pd

file_path = "C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터/2024년기준_전국오염원_조사자료_환경기초시설_가확정_250721.xlsx"

with open("debug_73mb_header.txt", "w", encoding="utf-8") as f:
    f.write(f"Reading file: {file_path}\n")
    df = pd.read_excel(file_path, sheet_name="방류량", nrows=5)
    f.write(f"Columns: {list(df.columns)}\n")
    for i in range(5):
        f.write(f"Row {i}: {df.iloc[i].tolist()}\n")
