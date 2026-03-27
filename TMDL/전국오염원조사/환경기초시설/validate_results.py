import pandas as pd
import numpy as np
from pathlib import Path

# 경로 설정
BASE_DIR = Path("C:/Coding/TMDL/전국오염원조사/기준배출수질/Output")
r_output_path = BASE_DIR / "기준배출수질_2024년기준.xlsx"
py_output_path = BASE_DIR / "기준배출수질_2024년기준_py.xlsx"

print(f"R Result File: {r_output_path.exists()}")
print(f"Python Result File: {py_output_path.exists()}\n")

if not r_output_path.exists() or not py_output_path.exists():
    print("One or both files are missing. Validation aborted.")
    exit(1)

# R 결과물과 Python 결과물 시트 목록 확인
xls_r = pd.ExcelFile(r_output_path)
xls_py = pd.ExcelFile(py_output_path)

print(f"R Sheets: {xls_r.sheet_names}")
print(f"Py Sheets: {xls_py.sheet_names}\n")

# 검증할 주요 시트 목록
sheets_to_check = [
    "기준배출수질_최종",
    "BOD_정규성검증", "BOD_기준배출수질_가", "BOD_기준배출수질_나",
    "TP_정규성검증", "TP_기준배출수질_가", "TP_기준배출수질_나"
]

for sheet in sheets_to_check:
    print(f"--- Checking Sheet: {sheet} ---")
    if sheet not in xls_r.sheet_names or sheet not in xls_py.sheet_names:
        print(f"  -> Sheet missing in one of the files")
        continue
        
    df_r = pd.read_excel(r_output_path, sheet_name=sheet)
    df_py = pd.read_excel(py_output_path, sheet_name=sheet)
    
    print(f"  R Shape: {df_r.shape}")
    print(f"  Py Shape: {df_py.shape}")
    
    if df_r.shape != df_py.shape:
        print("  -> SHAPE MISMATCH")
        continue
    
    # 텍스트 컬럼 등 불일치 여부 확인을 위해 정렬 후 비교
    df_r_sorted = df_r.sort_values(by=df_r.columns[0]).reset_index(drop=True)
    df_py_sorted = df_py.sort_values(by=df_py.columns[0]).reset_index(drop=True)
    
    # 수치형 컬럼 비교 (오차 허용 범위)
    numeric_cols = df_r_sorted.select_dtypes(include=[np.number]).columns
    
    diff_count_total = 0
    for col in numeric_cols:
        if col in df_py_sorted.columns:
            # NA 동일시 취급
            r_vals = df_r_sorted[col].fillna(0)
            py_vals = df_py_sorted[col].fillna(0)
            
            # 절대 오차가 0.05 이상인 경우 찾기 (반올림 차이 감안)
            diff = np.abs(r_vals - py_vals) > 0.05
            if diff.any():
                diff_count = diff.sum()
                diff_count_total += diff_count
                print(f"  -> Column [{col}] has {diff_count} diffs")
    
    if diff_count_total == 0:
        print("  -> NUMERIC VALUES MATCH (Tolerance: 0.05)")
    else:
        print(f"  -> TOTAL {diff_count_total} NUMERIC DIFFS FOUND")
        
    print()

print("Validation Script Finished.")
