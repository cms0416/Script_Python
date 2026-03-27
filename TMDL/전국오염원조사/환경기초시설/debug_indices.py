import pandas as pd

file1 = "C:/Coding/TMDL/전국오염원조사/기준배출수질/환경기초시설 데이터/2024년기준_전국오염원_조사자료_환경기초시설_가확정_250721.xlsx"

with open("debug_exact_indices.txt", "w", encoding="utf-8") as f:
    f.write("Reading 73MB file with skiprows=0 to get real headers...\n")
    df_head = pd.read_excel(file1, sheet_name="방류량", nrows=2)
    
    # We want to see what is at column offsets 6, 7, 9, 10, 12, 13, 14, 15
    indices = [0, 6, 7, 9, 10, 12, 13, 14, 15]
    
    for idx in indices:
        if idx < df_head.shape[1]:
            val0 = df_head.iloc[0, idx]
            val1 = df_head.iloc[1, idx]
            f.write(f"Index {idx}: Row0='{val0}', Row1='{val1}'\n")
