import runpy

print("Executing 환경기초시설_기준배출수질_정리.py as __main__...")
try:
    runpy.run_path('환경기초시설_기준배출수질_정리.py', run_name='__main__')
    print("Execution complete.")
except Exception as e:
    print(f"Error during execution: {e}")
