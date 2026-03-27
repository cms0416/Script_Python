import pandas as pd
import sys

file1 = "C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터/2024년기준_전국오염원_조사자료_환경기초시설_가확정_250721.xlsx"

with open("debug_names_out.txt", "w", encoding="utf-8") as f:
    f.write("Reading 73MB file...\n")
    df = pd.read_excel(file1, sheet_name="방류량", skiprows=3, header=None)
    f.write(f"Columns: {len(df.columns)}\n")
    f.write(f"Target index 0 unique values:\n{list(df[0].dropna().unique()[:20])}\n")
    f.write(f"Target index 1 unique values:\n{list(df[1].dropna().unique()[:20])}\n")
