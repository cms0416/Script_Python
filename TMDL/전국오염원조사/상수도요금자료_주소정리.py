#%%  관련 모듈 및 패키지 import
import pandas as pd
import glob
import numpy as np


#%%  모든 누적관리대장 파일에서 "협의"가 포함안된 시트 자료 합치기
ws_address = pd.DataFrame()

for f in glob.glob("E:\R\총량\Data\전국오염원조사\생활계 물사용량 주소 검증\*.xlsx"):
    df = pd.read_excel(f)
    ws_address = ws_address.append(df, ignore_index=True)

print(ws_address)


#%%
ws_address2 = pd.DataFrame()

ws_address2 = ws_address.assign(시도="강원도", 테스트="테스트")

print(ws_address2)


# %%
