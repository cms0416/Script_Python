import pandas as pd

r_file = "C:/Coding/TMDL/전국오염원조사/기준배출수질/Output/기준배출수질_2024년기준.xlsx"
py_file = "C:/Coding/TMDL/전국오염원조사/기준배출수질/Output/기준배출수질_2024년기준_py.xlsx"

df_r = pd.read_excel(r_file, sheet_name="기준배출수질_최종")
df_py = pd.read_excel(py_file, sheet_name="기준배출수질_최종")

df_merged = pd.merge(
    df_r[['시설명', 'TP_기준']], 
    df_py[['시설명', 'TP_기준']], 
    on='시설명', suffixes=('_R', '_Py')
)

df_diff = df_merged[df_merged['TP_기준_R'] != df_merged['TP_기준_Py']].head(10)
print("TP_기준 Differences (first 10):")
for _, row in df_diff.iterrows():
    print(f"{row['시설명']}: R={row['TP_기준_R']}, Py={row['TP_기준_Py']}")
