import pandas as pd
from pathlib import Path

BASE_DIR = Path("C:/Coding/TMDL/전국오염원조사/기준배출수질/Output")
r_path = BASE_DIR / "기준배출수질_2024년기준.xlsx"
py_path = BASE_DIR / "기준배출수질_2024년기준_py.xlsx"

df_r = pd.read_excel(r_path, sheet_name="BOD_정규성검증")
df_py = pd.read_excel(py_path, sheet_name="BOD_정규성검증")

r_facilities = set(df_r['시설명'].unique())
py_facilities = set(df_py['시설명'].unique())

diff_r = r_facilities - py_facilities
diff_py = py_facilities - r_facilities

print(f"BOD_정규성검증 - R_only ({len(diff_r)}): {diff_r}")
print(f"BOD_정규성검증 - Py_only ({len(diff_py)}): {diff_py}")

df_r_final = pd.read_excel(r_path, sheet_name="기준배출수질_최종")
df_py_final = pd.read_excel(py_path, sheet_name="기준배출수질_최종")

r_fin = set(df_r_final['시설명'].unique())
py_fin = set(df_py_final['시설명'].unique())

print(f"기준배출수질_최종 - R_only ({len(r_fin - py_fin)}): {r_fin - py_fin}")
print(f"기준배출수질_최종 - Py_only ({len(py_fin - r_fin)}): {py_fin - r_fin}")
