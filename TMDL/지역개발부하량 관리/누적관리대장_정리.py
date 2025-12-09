#%%  관련 모듈 및 패키지 import
import pandas as pd
import glob
import numpy as np


#%%  누적관리대장 파일 경로 리스트 작성
list_filepath = glob.glob("E:\Coding\TMDL\지역개발부하량 관리\2022\*.xls")


#%%  모든 누적관리대장 파일에서 "협의"가 포함안된 시트 자료 합치기
data = []
for filepath in list_filepath:
    # 각 파일의 시트명 리스트 작성
    sheets = pd.ExcelFile(filepath).sheet_names
    # "(협의)" 포함 시트명 찾기
    remove = [i for i in sheets if "(협의)" in i]
    # "(협의)" 포함 시트명 리스트에서 삭제
    sheets = [i for i in sheets if i not in remove]

    # 각 파일별, 시트별 순차적으로 합치기
    for sheet in sheets:
        data.append(pd.read_excel(filepath, sheet, skiprows=3, header=None))
    data_merged = pd.concat(data)

print(data_merged)


#%%  필요 없는 열 제거 및 열 이름 설정
data_fix = data_merged.iloc[:, np.r_[0:5, 8:12, 53, 54, 59, 60, 77, 119, 120, 125, 126, 146, 147]]
data_fix.columns = [
    "관리자번호",
    "시군",
    "단위유역",
    "할당일자",
    "사업명",
    "착공연도",
    "완공연도",
    "준공여부",
    "삭감량_BOD",
    "BOD지역개발_점",
    "BOD지역개발_비점",
    "BOD소진_점",
    "BOD소진_비점",
    "삭감량_TP",
    "TP지역개발_점",
    "TP지역개발_비점",
    "TP소진_점",
    "TP소진_비점",
    "협의일자",
    "협의상태",
]

# %%  날짜 서식 변경
data_fix["할당일자"] = data_fix["할당일자"].str.replace(".", "-")
data_fix["협의일자"] = data_fix["협의일자"].str.replace(".", "-")

# %%
data_fix.dtypes

# %%  날짜 데이터 형식 변경
data_fix["할당일자"] = pd.to_datetime(data_fix["할당일자"], format="%Y-%m-%d")
data_fix["협의일자"] = pd.to_datetime(data_fix["협의일자"], format="%Y-%m-%d")

# %%  할당연도, 협의연도 추가
data_fix["할당연도"] = data_fix["할당일자"].dt.year
data_fix["협의연도"] = data_fix["협의일자"].dt.year

# %%  계산연도 추가
data_fix["계산연도"] = np.where(
    data_fix["관리자번호"] == "기본계획_최초개발",
    2019,
    np.where(
        data_fix["관리자번호"] == "기본계획_기승인",
        2020,
        np.where(pd.isna(data_fix["협의일자"]), data_fix["할당연도"], data_fix["협의연도"]),
    ),
)

# %%
