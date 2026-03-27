import os
import sys

# Windows 환경에서 한글 경로 문제를 우회하기 위한 래퍼(Wrapper) 스크립트
script_path = r"c:\Coding\Script_Python\TMDL\전국오염원조사\환경기초시설\환경기초시설_기준배출수질_정리.py"
print(f"Executing: {script_path}")

try:
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 실행 시 발생하는 상세한 예외를 확인하기 위해 환경 세팅
    import traceback
    try:
        exec(code, {"__file__": script_path, "__name__": "__main__"})
        print("Execution completed successfully.")
    except Exception as e:
        print(f"Error during execution:")
        traceback.print_exc()

except Exception as e:
    print(f"Error reading wrapper wrapper script: {e}")
