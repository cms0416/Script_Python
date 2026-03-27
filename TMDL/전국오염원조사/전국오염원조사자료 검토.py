# %% [markdown]
# # 전국오염원조사 자료 정리

# %% [markdown]
# ******************************************************************************
# ### 환경설정 및 공통함수 정의

##### 라이브러리 로드 ########################################################
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings

# 경고 메시지 무시(선택사항)
warnings.filterwarnings('ignore')

##### 경로 및 연도 설정 ########################################################
# 작업 경로 지정(환경에 맞게 수정)
BASE_DIR = Path(r"C:/Coding/TMDL/전국오염원조사")
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

final_year = 2024
period = 5
years = list(range(final_year - period + 1, final_year + 1))

##### 공통파일(단위유역별 점유율) 불러오기 ###################################
share_path = BASE_DIR / "단위유역별_점유율_조정.xlsx"

share = pd.read_excel(share_path)

##### 함수 정의 ##############################################################

# 0. 반올림 사용자 정의 함수 (func_round2.R 대체)
def round2(x, n=0):
    """
    R의 기본 짝수 반올림(Banker's rounding) 대신 
    사사오입(Arithmetic rounding)을 수행하는 벡터화 함수
    """
    if pd.isna(x).all() if isinstance(x, pd.Series) else pd.isna(x):
        return x
    posneg = np.sign(x)
    z = np.abs(x) * (10 ** n)
    z = np.trunc(z + 0.5)
    z = z / (10 ** n)
    return z * posneg

# 1. 유역 및 시군별 소계 계산 함수
def subtotal(df, class_col=None):
    """
    R의 janitor::adorn_totals() 로직을 구현한 다중 그룹 소계 산출 함수
    """
    df_res = df.copy()
    
    if class_col is None:
        # 단위유역별 '강원도' (시군 합계)
        sub1 = df_res.groupby(['연도', '단위유역']).sum(numeric_only=True).reset_index()
        sub1['시군'] = '강원도'
        df_res = pd.concat([df_res, sub1], ignore_index=True)
        
        # 시군별 '소계' (단위유역 합계) 
        # (앞서 추가된 '강원도'의 단위유역 '소계'도 동시 계산됨)
        sub2 = df_res.groupby(['연도', '시군']).sum(numeric_only=True).reset_index()
        sub2['단위유역'] = '소계'
        df_res = pd.concat([df_res, sub2], ignore_index=True)
    else:
        # 단위유역 & 분류별 '강원도' (시군 합계)
        sub1 = df_res.groupby(['연도', '단위유역', class_col]).sum(numeric_only=True).reset_index()
        sub1['시군'] = '강원도'
        df_res = pd.concat([df_res, sub1], ignore_index=True)
        
        # 단위유역 & 시군별 '소계' (분류 합계)
        sub2 = df_res.groupby(['연도', '단위유역', '시군']).sum(numeric_only=True).reset_index()
        sub2[class_col] = '소계'
        df_res = pd.concat([df_res, sub2], ignore_index=True)
        
        # 시군 & 분류별 '소계' (단위유역 합계)
        sub3 = df_res.groupby(['연도', '시군', class_col]).sum(numeric_only=True).reset_index()
        sub3['단위유역'] = '소계'
        df_res = pd.concat([df_res, sub3], ignore_index=True)
        
    return df_res

# 2. 동리기준 자료 소계 계산 함수_읍면동리 포함
def subtotal_dongri(df, class_col=None):
    # 읍면동 NA 제거 공통 처리
    df_res = df.dropna(subset=['읍면동']).copy()
    
    if class_col is None:
        # 분류 없는 경우
        sub = df_res.groupby(['연도', '단위유역', '시군'], dropna=False).sum(numeric_only=True).reset_index()
        sub['읍면동'] = '소계'
        sub['리'] = '소계'
        df_res = pd.concat([df_res, sub], ignore_index=True)
    else:
        # 분류 있는 경우 (리 열에 NaN이 있어도 그룹 유지)
        sub = df_res.groupby(['연도', '단위유역', '시군', '읍면동', '리'], dropna=False).sum(numeric_only=True).reset_index()
        sub[class_col] = '소계'
        df_res = pd.concat([df_res, sub], ignore_index=True)
        
    return df_res

# 3. 단위유역/시군 순서 지정 함수
ORDER_CITY = [
    "춘천시", "원주시", "강릉시", "태백시", "삼척시", "홍천군",
    "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군",
    "양구군", "인제군", "고성군", "동해시", "속초시", "양양군"
]

ORDER_BASIN = [
    "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A",
    "섬강A", "섬강B", "북한A", "북한B", "소양A", "인북A", "소양B", "북한C",
    "홍천A", "한탄A", "제천A", "한강B", "한강D", "북한D", "임진A", "한탄B",
    "낙본A", "기타"
]

def order_func(df, value_col, class_col=None):
    df = df.copy()
    
    # 권역 추가
    conditions = [
        df['단위유역'].isin(["골지A", "오대A", "주천A", "평창A", "옥동A", "한강A"]),
        df['단위유역'].isin(["섬강A", "섬강B"]),
        df['단위유역'].isin(["북한A", "북한B", "소양A", "소양B", "인북A", "북한C"]),
        df['단위유역'] == "홍천A",
        df['단위유역'] == "한탄A",
        df['단위유역'].isin(["한강B", "제천A", "한강D"]),
        df['단위유역'].isin(["북한D", "한탄B", "임진A"]),
        df['단위유역'] == "낙본A"
    ]
    choices = ["남한강", "섬강", "북한강", "홍천강", "한탄강", "충청북도", "경기도", "낙동강"]
    df['권역'] = np.select(conditions, choices, default="기타")
    
    # 데이터 구조에 따른 열 순서 분기
    if '읍면동' in df.columns:
        base_order = ['권역', '단위유역', '시군', '읍면동', '리']
    else:
        base_order = ['권역', '시군', '단위유역']
        
    if class_col and class_col not in base_order:
        base_order.append(class_col)
        
    index_cols = [c for c in base_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in index_cols + ['연도', value_col]]
    index_cols.extend(other_cols)
    
    # ★ 추가 안전장치: pivot_table 실행 전 인덱스 열에 NaN이 혹시라도 있으면 빈칸 처리
    for c in index_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("")
            
    pivot_df = df.pivot_table(
        index=index_cols, 
        columns='연도', 
        values=value_col, 
        aggfunc='sum'
    ).reset_index()
    
    # 팩터(Categorical) 레벨 설정
    pivot_df['단위유역'] = pd.Categorical(pivot_df['단위유역'], categories=["소계"] + ORDER_BASIN, ordered=True)
    pivot_df['시군'] = pd.Categorical(pivot_df['시군'], categories=["강원도"] + ORDER_CITY, ordered=True)
    
    # 행 정렬 순서 결정
    sort_cols = ['시군', '단위유역']
    if class_col and class_col in pivot_df.columns:
        sort_cols.append(class_col)
        
    # R과 동일하게 명시적으로 지정한 열에 대해서만 정렬을 수행하고, 지정되지 않은 
    # 나머지 열은 원본 데이터프레임의 순서를 그대로 유지하는 kind='stable' 옵션 적용
    return pivot_df.sort_values(by=sort_cols, kind='stable').reset_index(drop=True)

# 4. 지명 변경된 경우 변경 지명 적용 및 동리코드 추가
def dongri_code(df):
    df = df.copy()
    
    # 지명 변경 조건 설정
    mask1 = (df['시군'] == '양구군') & (df['읍면동'] == '남면')
    mask2 = (df['시군'] == '홍천군') & (df['읍면동'] == '동면')
    mask3 = (df['시군'] == '영월군') & (df['읍면동'] == '중동면')
    mask4 = (df['시군'] == '영월군') & (df['읍면동'] == '수주면')
    
    df.loc[mask1, '읍면동'] = "국토정중앙면"
    df.loc[mask2, '읍면동'] = "영귀미면"
    df.loc[mask3, '읍면동'] = "산솔면"
    df.loc[mask4, '읍면동'] = "무릉도원면"
    
    # 동리코드 추가 ("리"가 없는 경우 빈칸 처리 후 공백 정리)
    df['리'] = df['리'].fillna("")
    df['동리코드'] = (df['시군'] + " " + df['읍면동'] + " " + df['리']).str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df

# 5. 동리별 단위유역 점유율 계산 함수
def share_cal(source_col, class_list=None):
    # share 변수는 전역 변수를 참조
    res = share.groupby(['동리코드', '단위유역', '시군'])[source_col].sum(numeric_only=True).reset_index()
    
    # 분류목록이 있을 경우 Cross Join 수행 (R의 unnest 리스트 확장 대체)
    if class_list:
        class_df = pd.DataFrame({'분류': class_list})
        res = res.merge(class_df, how='cross')
        
    return res

# 6. 시군 기준 자료와 동리 기준 자료 합치기 함수
def bind_dongri(sigun_df, dongri_df):
    sigun_df = sigun_df.copy()
    dongri_df = dongri_df.copy()
    
    # 읍면동, 리에 소계 입력
    sigun_df['읍면동'] = '소계'
    sigun_df['리'] = '소계'
    
    # 병합되는 최종 결과물의 열 순서를 '단위유역' -> '시군'으로 명시적 재배치
    base_cols = ['권역', '단위유역', '시군', '읍면동', '리']
    sigun_other = [c for c in sigun_df.columns if c not in base_cols]
    dongri_other = [c for c in dongri_df.columns if c not in base_cols]
    
    sigun_df = sigun_df[base_cols + sigun_other]
    dongri_df = dongri_df[base_cols + dongri_other]
    
    return pd.concat([sigun_df, dongri_df], ignore_index=True)

# %% [markdown]
# ******************************************************************************
# ###  생활계 - 인구


# 파일 불러오기
# 데이터 경로지정 및 데이터 목록
file_dir = BASE_DIR / "생활계"

# 지정한 연도만 선택 (파일명 앞 4자리 연도 추출)
files = [
    f for f in file_dir.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

# 경로지정된 생활계 파일 합치기
df_list = []
for f in files:
    temp = pd.read_excel(f, skiprows=4, header=None)
    temp = temp.iloc[:, [0, 1, 2, 3, 4, 5, 18]]
    df_list.append(temp)

생활계_인구_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
# 변수명 지정 및 데이터 정리
생활계_인구_2정리 = 생활계_인구_1원본.copy()
생활계_인구_2정리.columns = [
    "연도", "행정구역코드", "시도", "시군", "읍면동", "리", "가정인구합계"
]

# 수치데이터 및 연도 숫자로 지정
생활계_인구_2정리['연도'] = pd.to_numeric(생활계_인구_2정리['연도'], errors='coerce')
생활계_인구_2정리['가정인구합계'] = pd.to_numeric(생활계_인구_2정리['가정인구합계'], errors='coerce')

# 지명 변경된 경우 변경 지명 적용 및 동리코드 추가 (Step 1 함수 활용)
생활계_인구_2정리 = dongri_code(생활계_인구_2정리)

# 연도 및 동리별 가정인구합계 정리
생활계_인구_3동리 = 생활계_인구_2정리.groupby(
    ['연도', '동리코드', '시군', '읍면동', '리'], dropna=False
)['가정인구합계'].sum().reset_index()

# ------------------------------------------------------------------------------
# 각 동리별 단위유역 점유율 계산(소유역 점유율 합계)
share_생활계 = share_cal(source_col="생활계")

# 유역/시군 기준 인구 합계 연도별 정리
years_df = pd.DataFrame({'연도': years})
base_df_생활계 = years_df.merge(share_생활계, how='cross')

생활계_인구_4합계 = base_df_생활계.merge(
    생활계_인구_3동리, 
    on=["동리코드", "연도", "시군"], 
    how='left'
)

# 동리별 가정인구합계와 유역 점유율 계산 (round2 적용)
생활계_인구_4합계['총인구'] = round2(생활계_인구_4합계['생활계'] * 생활계_인구_4합계['가정인구합계'], 0)

# 유역, 시군별 총인구 합계
생활계_인구_4합계 = 생활계_인구_4합계.groupby(
    ['연도', '단위유역', '시군']
)['총인구'].sum().reset_index()

# 소계 계산, 단위유역/시군 순서 지정, 연도 기준 wide 포맷 변환
생활계_인구_4합계 = subtotal(생활계_인구_4합계)
생활계_인구_4합계 = order_func(생활계_인구_4합계, value_col="총인구")

# 오염원 및 분류 열 추가
idx_basin = 생활계_인구_4합계.columns.get_loc('단위유역')
생활계_인구_4합계.insert(idx_basin + 1, '오염원', '생활계')
생활계_인구_4합계.insert(idx_basin + 2, '분류', '인구')

# 데이터 정수형 변환(결측치는 0으로 처리)
data_cols = [c for c in 생활계_인구_4합계.columns if isinstance(c, int) or str(c).isdigit()]
생활계_인구_4합계[data_cols] = 생활계_인구_4합계[data_cols].fillna(0).astype(int)

# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

# 유역/동리 기준 인구 합계 연도별 정리
생활계_인구_4합계_동리 = base_df_생활계.merge(
    생활계_인구_3동리, 
    on=["동리코드", "연도", "시군"], 
    how='left'
)

# 동리별 가정인구합계와 유역 점유율 계산 (round2 적용)
생활계_인구_4합계_동리['총인구'] = round2(생활계_인구_4합계_동리['생활계'] * 생활계_인구_4합계_동리['가정인구합계'], 0)

# 유역, 시군, 동리별 총인구 합계
생활계_인구_4합계_동리 = 생활계_인구_4합계_동리.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리'], dropna=False
)['총인구'].sum().reset_index()

# 소계 계산, 단위유역/시군 순서 지정, 연도 기준 wide 포맷 변환
생활계_인구_4합계_동리 = subtotal_dongri(생활계_인구_4합계_동리)
생활계_인구_4합계_동리 = order_func(생활계_인구_4합계_동리, value_col="총인구")

# 오염원 및 분류 열 추가
idx_ri = 생활계_인구_4합계_동리.columns.get_loc('리')
생활계_인구_4합계_동리.insert(idx_ri + 1, '오염원', '생활계')
생활계_인구_4합계_동리.insert(idx_ri + 2, '분류', '인구')

# 데이터 정수형 변환(결측치는 0으로 처리)
data_cols_dongri = [c for c in 생활계_인구_4합계_동리.columns if isinstance(c, int) or str(c).isdigit()]
생활계_인구_4합계_동리[data_cols_dongri] = 생활계_인구_4합계_동리[data_cols_dongri].fillna(0).astype(int)

# 시군 기준 자료와 동리 기준 자료 합치기
생활계_인구_4합계_동리 = bind_dongri(생활계_인구_4합계, 생활계_인구_4합계_동리)


# %% [markdown]
# ******************************************************************************
# ###  생활계 - 물사용량

# 파일 불러오기  
file_dir = BASE_DIR / "생활계"

files = [
    f for f in file_dir.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

# 경로지정된 생활계 파일 합치기 (Sheet 2)
df_list = []
for f in files:
    # R: sheet = 2 -> Python: sheet_name=1 (0부터 시작)
    temp = pd.read_excel(f, sheet_name=1, skiprows=4, header=None)
    # R: select(1:6, 19, 36) -> Python: 0:5, 18, 35
    temp = temp.iloc[:, [0, 1, 2, 3, 4, 5, 18, 35]]
    df_list.append(temp)

생활계_물사용량_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
## 변수명 지정 및 데이터 정리
생활계_물사용량_2정리 = 생활계_물사용량_1원본.copy()
생활계_물사용량_2정리.columns = [
    "연도", "행정구역코드", "시도", "시군", "읍면동", "리", 
    "가정용물사용합계", "영업용물사용합계"
]

# 수치데이터 및 연도 숫자로 지정
num_cols = ["연도", "가정용물사용합계", "영업용물사용합계"]
생활계_물사용량_2정리[num_cols] = 생활계_물사용량_2정리[num_cols].apply(pd.to_numeric, errors='coerce')

# 지명 변경된 경우 변경 지명 적용 및 동리코드 추가
생활계_물사용량_2정리 = dongri_code(생활계_물사용량_2정리)

# 결측치 0 처리 후 물사용량 합계 계산
생활계_물사용량_2정리[['가정용물사용합계', '영업용물사용합계']] = 생활계_물사용량_2정리[['가정용물사용합계', '영업용물사용합계']].fillna(0)
생활계_물사용량_2정리['물사용량'] = 생활계_물사용량_2정리['가정용물사용합계'] + 생활계_물사용량_2정리['영업용물사용합계']

## 연도 및 동리별 물사용량합계 정리 (dropna=False 유지)
생활계_물사용량_3동리 = 생활계_물사용량_2정리.groupby(
    ['연도', '동리코드', '시군', '읍면동', '리'], dropna=False
)['물사용량'].sum().reset_index()

# ------------------------------------------------------------------------------
## 각 동리별 단위유역 점유율 계산(소유역 점유율 합계)
share_생활계 = share_cal(source_col="생활계")

생활계_물사용량_4합계 = base_df_생활계.merge(
    생활계_물사용량_3동리, 
    on=["동리코드", "연도", "시군"], 
    how='left'
)

# 동리별 물사용량합계와 유역 점유율 계산 (소수점 첫째자리 반올림)
생활계_물사용량_4합계['물사용량'] = round2(생활계_물사용량_4합계['생활계'] * 생활계_물사용량_4합계['물사용량'], 1)

생활계_물사용량_4합계 = 생활계_물사용량_4합계.groupby(
    ['연도', '단위유역', '시군']
)['물사용량'].sum().reset_index()

# 소계 계산 및 wide 포맷 변환
생활계_물사용량_4합계 = subtotal(생활계_물사용량_4합계)
생활계_물사용량_4합계 = order_func(생활계_물사용량_4합계, value_col="물사용량")

idx_basin = 생활계_물사용량_4합계.columns.get_loc('단위유역')
생활계_물사용량_4합계.insert(idx_basin + 1, '오염원', '생활계')
생활계_물사용량_4합계.insert(idx_basin + 2, '분류', '물사용량')

# 물사용량 소수점 첫째자리까지 표현(결측치는 0으로 처리)
data_cols_water = [c for c in 생활계_물사용량_4합계.columns if isinstance(c, int) or str(c).isdigit()]
생활계_물사용량_4합계[data_cols_water] = 생활계_물사용량_4합계[data_cols_water].fillna(0).round(1)


# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

생활계_물사용량_4합계_동리 = base_df_생활계.merge(
    생활계_물사용량_3동리, 
    on=["동리코드", "연도", "시군"], 
    how='left'
)

생활계_물사용량_4합계_동리['물사용량'] = round2(생활계_물사용량_4합계_동리['생활계'] * 생활계_물사용량_4합계_동리['물사용량'], 1)

생활계_물사용량_4합계_동리 = 생활계_물사용량_4합계_동리.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리'], dropna=False
)['물사용량'].sum().reset_index()

생활계_물사용량_4합계_동리 = subtotal_dongri(생활계_물사용량_4합계_동리)
생활계_물사용량_4합계_동리 = order_func(생활계_물사용량_4합계_동리, value_col="물사용량")

idx_ri = 생활계_물사용량_4합계_동리.columns.get_loc('리')
생활계_물사용량_4합계_동리.insert(idx_ri + 1, '오염원', '생활계')
생활계_물사용량_4합계_동리.insert(idx_ri + 2, '분류', '물사용량')

# 시군 기준 자료와 동리 기준 자료 합치기
생활계_물사용량_4합계_동리 = bind_dongri(생활계_물사용량_4합계, 생활계_물사용량_4합계_동리)

# 물사용량 소수점 첫째자리까지 표현(결측치는 0으로 처리)
data_cols_water_dongri = [c for c in 생활계_물사용량_4합계_동리.columns if isinstance(c, int) or str(c).isdigit()]
생활계_물사용량_4합계_동리[data_cols_water_dongri] = 생활계_물사용량_4합계_동리[data_cols_water_dongri].fillna(0).round(1)


# %% [markdown]
# ******************************************************************************
# ###  축산계
    
# 파일 불러오기  
file_dir_livestock = BASE_DIR / "축산계"

files_livestock = [
    f for f in file_dir_livestock.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

df_list = []
for f in files_livestock:
    temp = pd.read_excel(f, skiprows=6, header=None)
    year = int(f.name[:4])
    temp.insert(0, '연도', year)
    
    # Python index: 0, 11, 12, 13, 16, 17
    temp = temp.iloc[:, [0, 11, 12, 13, 16, 17]]
    df_list.append(temp)

축산계_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
## 변수명 지정 및 데이터 정리
축산계_2정리 = 축산계_1원본.copy()
축산계_2정리.columns = ["연도", "시군", "읍면동", "리", "분류", "사육두수"]

# 수치데이터 지정 및 사육두수 결측치 0 처리
축산계_2정리['연도'] = pd.to_numeric(축산계_2정리['연도'], errors='coerce')
축산계_2정리['사육두수'] = pd.to_numeric(축산계_2정리['사육두수'], errors='coerce').fillna(0)

# 지명 변경된 경우 적용
축산계_2정리 = dongri_code(축산계_2정리)

# 축산계 축종 변환
conditions = [
    축산계_2정리['분류'] == "한우(소)",
    축산계_2정리['분류'] == "유우(젖소)",
    축산계_2정리['분류'] == "돼지",
    축산계_2정리['분류'] == "마필(말)",
    축산계_2정리['분류'].isin(["산양(염소포함)", "면양(육양포함)", "사슴"]),
    축산계_2정리['분류'] == "개",
    축산계_2정리['분류'].isin(["닭", "오리", "타조", "가금기타"])
]
choices = ["한우", "젖소", "돼지", "말", "양, 사슴", "개", "가금"]
축산계_2정리['분류'] = np.select(conditions, choices, default="-")

## 연도 및 동리별 사육두수합계 정리
축산계_3동리 = 축산계_2정리.groupby(
    ['연도', '동리코드', '분류', '시군', '읍면동', '리'], dropna=False
)['사육두수'].sum().reset_index()

# ------------------------------------------------------------------------------
## 각 동리별 단위유역 점유율 계산 및 축종추가
share_축산계 = share_cal(
    source_col="축산계", 
    class_list=["젖소", "한우", "말", "돼지", "양, 사슴", "개", "가금"]
)

# ★수정: 축산계 전용 base_df 재정의★
years_df = pd.DataFrame({'연도': years})
base_df_축산계 = years_df.merge(share_축산계, how='cross')

## 유역/시군 기준 합계 연도별 정리
축산계_4합계 = base_df_축산계.merge(
    축산계_3동리, 
    on=["동리코드", "연도", "시군", "분류"], 
    how='left'
)

# 사육두수 계산 (정수형 반올림)
축산계_4합계['총사육두수'] = round2(축산계_4합계['축산계'] * 축산계_4합계['사육두수'], 0)

축산계_4합계 = 축산계_4합계.groupby(
    ['연도', '단위유역', '시군', '분류']
)['총사육두수'].sum().reset_index()

# 소계 계산 및 wide 포맷 변환
축산계_4합계 = subtotal(축산계_4합계, class_col="분류")

cat_order_livestock = ["젖소", "한우", "말", "돼지", "양, 사슴", "개", "가금", "소계"]
축산계_4합계['분류'] = pd.Categorical(축산계_4합계['분류'], categories=cat_order_livestock, ordered=True)

축산계_4합계 = order_func(축산계_4합계, value_col="총사육두수", class_col="분류")
idx_class = 축산계_4합계.columns.get_loc('분류')
축산계_4합계.insert(idx_class, '오염원', '축산계')

# 사육두수 정수형 변환
data_cols_livestock = [c for c in 축산계_4합계.columns if isinstance(c, int) or str(c).isdigit()]
축산계_4합계[data_cols_livestock] = 축산계_4합계[data_cols_livestock].fillna(0).astype(int)

# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

축산계_4합계_동리 = base_df_축산계.merge(
    축산계_3동리, 
    on=["동리코드", "연도", "시군", "분류"], 
    how='left'
)

축산계_4합계_동리['총사육두수'] = round2(축산계_4합계_동리['축산계'] * 축산계_4합계_동리['사육두수'], 0)

축산계_4합계_동리 = 축산계_4합계_동리.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리', '분류'], dropna=False
)['총사육두수'].sum().reset_index()

축산계_4합계_동리 = subtotal_dongri(축산계_4합계_동리, class_col="분류")
축산계_4합계_동리['분류'] = pd.Categorical(축산계_4합계_동리['분류'], categories=cat_order_livestock, ordered=True)

축산계_4합계_동리 = order_func(축산계_4합계_동리, value_col="총사육두수", class_col="분류")
idx_class_dongri = 축산계_4합계_동리.columns.get_loc('분류')
축산계_4합계_동리.insert(idx_class_dongri, '오염원', '축산계')

# 사육두수 정수형 변환
data_cols_livestock_dongri = [c for c in 축산계_4합계_동리.columns if isinstance(c, int) or str(c).isdigit()]
축산계_4합계_동리[data_cols_livestock_dongri] = 축산계_4합계_동리[data_cols_livestock_dongri].fillna(0).astype(int)

## 시군 및 동리 기준 자료 병합
축산계_4합계_동리 = bind_dongri(축산계_4합계, 축산계_4합계_동리)


# %% [markdown]
# ******************************************************************************
# ###  산업계
    
# 파일 불러오기
file_dir_industry = BASE_DIR / "산업계"

files_industry = [
    f for f in file_dir_industry.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

df_list = []
for f in files_industry:
    # R: skip = 5
    temp = pd.read_excel(f, skiprows=5, header=None)
    year = int(f.name[:4])
    
    # 연도 추가 (1열)
    temp.insert(0, '연도', year)
    
    # 2022년 이전 자료의 경우 부분위탁량(76열) 뒤에 폐기물처리 항목 추가
    # Python에서 77열(0-based index 77) 위치에 빈 열 삽입
    if year < 2022:
        temp.insert(77, '폐기물처리', "")
    
    # Python Index (0-based): 0, 3, 5, 7, 8, 9, 10, 11, 12, 23, 72, 81
    temp = temp.iloc[:, [0, 3, 5, 7, 8, 9, 10, 11, 12, 23, 72, 81]].copy()
    
    # ★수정: pd.concat 시 열 이름 불일치로 인한 추가 열 생성 에러를 방지하기 위해 추출 즉시 이름 지정
    temp.columns = [
        "연도", "휴업", "업소명", "시도", "시군", "읍면동", "리",
        "본번", "부번", "분류", "폐수발생량", "폐수방류량"
    ]
    df_list.append(temp)

산업계_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
## 변수명 지정 및 데이터 정리
산업계_2정리 = 산업계_1원본.copy()

# 휴업인 경우 삭제 (휴업 열이 결측치인 행만 보존)
산업계_2정리 = 산업계_2정리[산업계_2정리['휴업'].isna()].copy()

# 수치데이터 지정 및 결측치 0 처리
num_cols_ind = ['폐수발생량', '폐수방류량', '연도']
산업계_2정리[num_cols_ind] = 산업계_2정리[num_cols_ind].apply(pd.to_numeric, errors='coerce').fillna(0)

# 지명 변경 적용 및 동리코드 추가
산업계_2정리 = dongri_code(산업계_2정리)

# 폐수방류량이 음수인 경우 0으로 수정
산업계_2정리.loc[산업계_2정리['폐수방류량'] < 0, '폐수방류량'] = 0

## 연도 및 동리별 합계 정리
산업계_2정리['업소수'] = 1
산업계_3동리 = 산업계_2정리.groupby(
    ['연도', '동리코드', '시군', '읍면동', '리', '분류'], dropna=False
)[['업소수', '폐수발생량', '폐수방류량']].sum().reset_index()

# ------------------------------------------------------------------------------
## 각 동리별 단위유역 점유율 계산 (소유역 점유율 합계)
share_산업계 = share_cal(
    source_col="산업계",
    class_list=["1종", "2종", "3종", "4종", "5종"]
)

## 유역/시군 기준 합계 연도별 정리
years_df = pd.DataFrame({'연도': years})
base_df_산업계 = years_df.merge(share_산업계, how='cross')

산업계_4합계_0 = base_df_산업계.merge(
    산업계_3동리,
    on=["동리코드", "연도", "시군", "분류"],
    how='left'
)

# 점유율 계산 (업소수는 정수 반올림, 폐수는 소수점 첫째자리)
산업계_4합계_0['업소수'] = round2(산업계_4합계_0['산업계'] * 산업계_4합계_0['업소수'], 0)
산업계_4합계_0['폐수발생량'] = round2(산업계_4합계_0['산업계'] * 산업계_4합계_0['폐수발생량'], 1)
산업계_4합계_0['폐수방류량'] = round2(산업계_4합계_0['산업계'] * 산업계_4합계_0['폐수방류량'], 1)

산업계_4합계_0 = 산업계_4합계_0.groupby(
    ['연도', '단위유역', '시군', '분류']
)[['업소수', '폐수발생량', '폐수방류량']].sum().reset_index()

# 소계 계산
산업계_4합계_0 = subtotal(산업계_4합계_0, class_col="분류")

cat_order_industry = ["1종", "2종", "3종", "4종", "5종", "소계"]
산업계_4합계_0['분류'] = pd.Categorical(산업계_4합계_0['분류'], categories=cat_order_industry, ordered=True)

## 각 연도를 열로 변경 (wide 포맷) - 항목별 분리 후 병합
# 1. 업소수
산업계_4합계_업소수 = 산업계_4합계_0.drop(columns=['폐수발생량', '폐수방류량'])
산업계_4합계_업소수 = order_func(산업계_4합계_업소수, value_col="업소수", class_col="분류")
산업계_4합계_업소수.insert(산업계_4합계_업소수.columns.get_loc('분류'), '오염원', '산업계_업소수')

# 2. 폐수발생량
산업계_4합계_폐수발생량 = 산업계_4합계_0.drop(columns=['업소수', '폐수방류량'])
산업계_4합계_폐수발생량 = order_func(산업계_4합계_폐수발생량, value_col="폐수발생량", class_col="분류")
산업계_4합계_폐수발생량.insert(산업계_4합계_폐수발생량.columns.get_loc('분류'), '오염원', '산업계_폐수발생량')

# 3. 폐수방류량
산업계_4합계_폐수방류량 = 산업계_4합계_0.drop(columns=['업소수', '폐수발생량'])
산업계_4합계_폐수방류량 = order_func(산업계_4합계_폐수방류량, value_col="폐수방류량", class_col="분류")
산업계_4합계_폐수방류량.insert(산업계_4합계_폐수방류량.columns.get_loc('분류'), '오염원', '산업계_폐수방류량')

## 데이터 정수형 및 소수점 변환 처리
# 1. 업소수는 정수로 표현
year_cols_upso = [c for c in 산업계_4합계_업소수.columns if isinstance(c, int) or str(c).isdigit()]
산업계_4합계_업소수[year_cols_upso] = 산업계_4합계_업소수[year_cols_upso].fillna(0).astype(int)     

# 2. 폐수발생량과 폐수방류량은 소수점 첫째자리까지 표현(결측치는 0으로 처리)
year_cols_waste = [c for c in 산업계_4합계_폐수발생량.columns if isinstance(c, int) or str(c).isdigit()]
산업계_4합계_폐수발생량[year_cols_waste] = 산업계_4합계_폐수발생량[year_cols_waste].fillna(0).round(1)
산업계_4합계_폐수방류량[year_cols_waste] = 산업계_4합계_폐수방류량[year_cols_waste].fillna(0).round(1)


## 항목별 병합
산업계_4합계 = pd.concat([산업계_4합계_업소수, 산업계_4합계_폐수발생량, 산업계_4합계_폐수방류량], ignore_index=True)


# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

산업계_4합계_동리_0 = base_df_산업계.merge(
    산업계_3동리,
    on=["동리코드", "연도", "시군", "분류"],
    how='left'
)

산업계_4합계_동리_0['업소수'] = round2(산업계_4합계_동리_0['산업계'] * 산업계_4합계_동리_0['업소수'], 0)
산업계_4합계_동리_0['폐수발생량'] = round2(산업계_4합계_동리_0['산업계'] * 산업계_4합계_동리_0['폐수발생량'], 1)
산업계_4합계_동리_0['폐수방류량'] = round2(산업계_4합계_동리_0['산업계'] * 산업계_4합계_동리_0['폐수방류량'], 1)

산업계_4합계_동리_0 = 산업계_4합계_동리_0.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리', '분류'], dropna=False
)[['업소수', '폐수발생량', '폐수방류량']].sum().reset_index()

산업계_4합계_동리_0 = subtotal_dongri(산업계_4합계_동리_0, class_col="분류")
산업계_4합계_동리_0['분류'] = pd.Categorical(산업계_4합계_동리_0['분류'], categories=cat_order_industry, ordered=True)

## 각 연도를 열로 변경 (wide 포맷)
# 1. 업소수
산업계_4합계_동리_업소수 = 산업계_4합계_동리_0.drop(columns=['폐수발생량', '폐수방류량'])
산업계_4합계_동리_업소수 = order_func(산업계_4합계_동리_업소수, value_col="업소수", class_col="분류")
산업계_4합계_동리_업소수.insert(산업계_4합계_동리_업소수.columns.get_loc('분류'), '오염원', '산업계_업소수')

# 2. 폐수발생량
산업계_4합계_동리_폐수발생량 = 산업계_4합계_동리_0.drop(columns=['업소수', '폐수방류량'])
산업계_4합계_동리_폐수발생량 = order_func(산업계_4합계_동리_폐수발생량, value_col="폐수발생량", class_col="분류")
산업계_4합계_동리_폐수발생량.insert(산업계_4합계_동리_폐수발생량.columns.get_loc('분류'), '오염원', '산업계_폐수발생량')

# 3. 폐수방류량
산업계_4합계_동리_폐수방류량 = 산업계_4합계_동리_0.drop(columns=['업소수', '폐수발생량'])
산업계_4합계_동리_폐수방류량 = order_func(산업계_4합계_동리_폐수방류량, value_col="폐수방류량", class_col="분류")
산업계_4합계_동리_폐수방류량.insert(산업계_4합계_동리_폐수방류량.columns.get_loc('분류'), '오염원', '산업계_폐수방류량')

## 데이터 정수형 및 소수점 변환 처리
# 1. 업소수는 정수로 표현
year_cols_upso = [c for c in 산업계_4합계_동리_업소수.columns if isinstance(c, int) or str(c).isdigit()]
산업계_4합계_동리_업소수[year_cols_upso] = 산업계_4합계_동리_업소수[year_cols_upso].fillna(0).astype(int)     

# 2. 폐수발생량과 폐수방류량은 소수점 첫째자리까지 표현(결측치는 0으로 처리)
year_cols_waste = [c for c in 산업계_4합계_동리_폐수발생량.columns if isinstance(c, int) or str(c).isdigit()]
산업계_4합계_동리_폐수발생량[year_cols_waste] = 산업계_4합계_동리_폐수발생량[year_cols_waste].fillna(0).round(1)
산업계_4합계_동리_폐수방류량[year_cols_waste] = 산업계_4합계_동리_폐수방류량[year_cols_waste].fillna(0).round(1)

## 항목별 병합
산업계_4합계_동리 = pd.concat([산업계_4합계_동리_업소수, 산업계_4합계_동리_폐수발생량, 산업계_4합계_동리_폐수방류량], ignore_index=True)

## 시군 기준 자료와 동리 기준 자료 합치기
산업계_4합계_동리 = bind_dongri(산업계_4합계, 산업계_4합계_동리)


# %% [markdown]
# ******************************************************************************
# ###  토지계
    
# 파일 불러오기
file_dir_land = BASE_DIR / "토지계"

files_land = [
    f for f in file_dir_land.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

df_list = []
for f in files_land:
    # R: skip=2, col_names=F. dtype=str을 사용하여 데이터 병합 시 자료형 충돌 방지
    temp = pd.read_excel(f, skiprows=2, header=None, dtype=str)
    year = int(f.name[:4])
    temp.insert(0, '연도', year)
    
    # 두 번째 열(index 1) 삭제 (R: select(-2))
    temp = temp.drop(columns=[1]).copy()
    
    # 변수명 지정
    temp.columns = [
        "연도", "시도", "시군", "읍면동", "리", "총면적", "전", "답", "과수원",
        "목장용지", "임야", "광천지", "염전", "대지", "공장용지", "학교용지",
        "주차장", "주유소용지", "창고용지", "도로", "철도용지", "제방", "하천",
        "구거", "유지", "양어장", "수도용지", "공원", "체육용지", "유원지",
        "종교용지", "사적지", "묘지", "잡종지"
    ]
    df_list.append(temp)

토지계_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
## 데이터 정리
토지계_2정리 = 토지계_1원본.copy()

# 연도 및 지목별 면적 숫자로 변환
토지계_2정리['연도'] = pd.to_numeric(토지계_2정리['연도'], errors='coerce')
num_cols_land = 토지계_2정리.columns[5:] # 총면적 ~ 잡종지
for col in num_cols_land:
    토지계_2정리[col] = pd.to_numeric(토지계_2정리[col], errors='coerce').fillna(0)

# 지명 변경 적용 및 동리코드 추가
토지계_2정리 = dongri_code(토지계_2정리)

# 토지계 지목 변환 (총량 기술지침 발생원단위 지목에 맞춰 조정)
토지계_2정리['기타초지'] = 토지계_2정리['목장용지'] + 토지계_2정리['공원'] + 토지계_2정리['묘지'] + 토지계_2정리['사적지']
토지계_2정리['기타'] = 토지계_2정리['광천지'] + 토지계_2정리['염전'] + 토지계_2정리['제방'] + 토지계_2정리['하천'] + 토지계_2정리['구거'] + 토지계_2정리['유지'] + 토지계_2정리['양어장'] + 토지계_2정리['잡종지']
토지계_2정리['공공시설지역'] = 토지계_2정리['학교용지'] + 토지계_2정리['창고용지'] + 토지계_2정리['종교용지']
토지계_2정리['교통지역'] = 토지계_2정리['주차장'] + 토지계_2정리['도로'] + 토지계_2정리['철도용지'] + 토지계_2정리['수도용지']

# 기존 세부 지목 열 삭제
drop_cols = [
    "목장용지", "공원", "묘지", "사적지", "광천지", "염전", "제방", "하천", "구거", 
    "유지", "양어장", "잡종지", "학교용지", "창고용지", "종교용지", "주차장", "도로", 
    "철도용지", "수도용지", "총면적"
]
토지계_2정리 = 토지계_2정리.drop(columns=drop_cols)

# pivot_longer 역할: 지목들을 '분류' 열로 통합 (Wide to Long)
id_vars = ["연도", "동리코드", "시도", "시군", "읍면동", "리"]
value_vars = [c for c in 토지계_2정리.columns if c not in id_vars]
토지계_2정리 = 토지계_2정리.melt(id_vars=id_vars, value_vars=value_vars, var_name="분류", value_name="면적")

## 연도 및 동리 별 지목 면적 정리
토지계_3동리 = 토지계_2정리.groupby(
    ['연도', '동리코드', '시군', '읍면동', '리', '분류'], dropna=False
)['면적'].sum().reset_index()

# ------------------------------------------------------------------------------
## 각 동리별 단위유역 점유율 계산 (소유역 점유율 합계)
land_categories = [
    "전", "답", "임야", "대지", "과수원", "기타초지", "기타", "공장용지", 
    "공공시설지역", "교통지역", "주유소용지", "체육용지", "유원지"
]
# 전역 변수인 share에서 토지계 관련 분류만 추출 후 Long 포맷으로 변환
share_토지계 = share[['동리코드', '단위유역', '시군'] + land_categories].copy()
share_토지계 = share_토지계.melt(
    id_vars=['동리코드', '단위유역', '시군'],
    value_vars=land_categories,
    var_name='분류',
    value_name='토지계'
)
share_토지계 = share_토지계.groupby(
    ['동리코드', '단위유역', '시군', '분류'], dropna=False
)['토지계'].sum().reset_index()

## 유역/시군 기준 지목별 면적 합계 연도별 정리
years_df = pd.DataFrame({'연도': years})
base_df_토지계 = years_df.merge(share_토지계, how='cross')

토지계_4합계 = base_df_토지계.merge(
    토지계_3동리, 
    on=['동리코드', '연도', '시군', '분류'], 
    how='left'
)
토지계_4합계['총면적'] = 토지계_4합계['토지계'] * 토지계_4합계['면적']

# 면적 합산 및 단위 변환 (㎡ → ㎢), round2(3자리) 적용
토지계_4합계 = 토지계_4합계.groupby(
    ['연도', '단위유역', '시군', '분류']
)['총면적'].sum().reset_index()
토지계_4합계['총면적'] = round2(토지계_4합계['총면적'] / (10**6), 3)

# 소계 계산 후 전망자료 반영을 위해 단위유역 '소계' 임시 제외
토지계_4합계 = subtotal(토지계_4합계, class_col="분류")
토지계_4합계 = 토지계_4합계[토지계_4합계['단위유역'] != '소계'].copy()

# ------------------------------------------------------------------------------
## ***** 토지계 소계 전망값과 동일하게 계산 (보정) *****
# 오염원전망 파일 경로 지정 (상위 폴더의 오염원증감현황 폴더 내)
PROSPECT_PATH = BASE_DIR.parent / "오염원증감현황" / "오염원 전망(시군별 정리).xlsx"

# 파일 존재 여부 엄격히 확인 (없으면 에러 발생시키고 스크립트 강제 중단)
if not PROSPECT_PATH.exists():
    raise FileNotFoundError(f"[오류] 필수 파일 누락: '{PROSPECT_PATH}' 파일이 존재하지 않습니다. 파일을 해당 경로에 위치시킨 후 다시 실행하십시오.")

오염원전망 = pd.read_excel(PROSPECT_PATH)

# 전망자료 내 토지계 "소계" 만 분리
오염원전망_토지계 = 오염원전망[(오염원전망['오염원'] == '토지계') & (오염원전망['분류'] == '소계')].copy()
오염원전망_토지계 = 오염원전망_토지계.rename(columns={'2030년': '전망'})[['시군', '단위유역', '분류', '전망']]

# 전망자료와 당해년도(final_year) 토지계 자료 "소계" 차이값 계산
토지계_final = 토지계_4합계[(토지계_4합계['연도'] == final_year) & (토지계_4합계['분류'] == '소계')].copy()
오염원전망_토지계_a = 토지계_final.merge(오염원전망_토지계, on=['시군', '단위유역', '분류'], how='inner')
오염원전망_토지계_a['차이'] = 오염원전망_토지계_a['전망'] - 오염원전망_토지계_a['총면적']
오염원전망_토지계_a = 오염원전망_토지계_a[['연도', '시군', '단위유역', '분류', '차이']]

# "소계"의 차이값을 지목 중 "임야"에 반영할 수 있도록 추가
오염원전망_토지계_임야 = 오염원전망_토지계_a.copy()
오염원전망_토지계_임야['분류'] = '임야'
오염원전망_토지계_b = pd.concat([오염원전망_토지계_a, 오염원전망_토지계_임야], ignore_index=True)

# "소계"와 "임야"에 차이값 반영
토지계_4합계 = 토지계_4합계.merge(오염원전망_토지계_b, on=['연도', '시군', '단위유역', '분류'], how='left')
토지계_4합계['차이'] = 토지계_4합계['차이'].fillna(0)
토지계_4합계['총면적'] = 토지계_4합계['총면적'] + 토지계_4합계['차이']
토지계_4합계 = 토지계_4합계.drop(columns=['차이'])

# 다시 시군별 합계(소계) 계산 (전망값 적용 후)
sub_sigun = 토지계_4합계.groupby(['연도', '시군', '분류'], dropna=False)['총면적'].sum(numeric_only=True).reset_index()
sub_sigun['단위유역'] = '소계'
토지계_4합계 = pd.concat([토지계_4합계, sub_sigun], ignore_index=True)

# 팩터 레벨 설정 및 Wide 포맷 변환
cat_order_land = [
    "전", "답", "과수원", "기타초지", "임야", "기타", "대지", "공장용지",
    "공공시설지역", "교통지역", "주유소용지", "체육용지", "유원지", "소계"
]
토지계_4합계['분류'] = pd.Categorical(토지계_4합계['분류'], categories=cat_order_land, ordered=True)
토지계_4합계 = order_func(토지계_4합계, value_col="총면적", class_col="분류")
토지계_4합계.insert(토지계_4합계.columns.get_loc('분류'), '오염원', '토지계')

# 토지계 면적은 소수점 3자리까지 표현(㎢ 단위)
year_cols_land = [c for c in 토지계_4합계.columns if isinstance(c, int) or str(c).isdigit()]
토지계_4합계[year_cols_land] = 토지계_4합계[year_cols_land].fillna(0).round(3)


# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

토지계_4합계_동리 = base_df_토지계.merge(
    토지계_3동리, 
    on=['동리코드', '연도', '시군', '분류'], 
    how='left'
)
토지계_4합계_동리['총면적'] = 토지계_4합계_동리['토지계'] * 토지계_4합계_동리['면적']

토지계_4합계_동리 = 토지계_4합계_동리.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리', '분류'], dropna=False
)['총면적'].sum().reset_index()

# 단위변환(㎡ → ㎢) 및 반올림
토지계_4합계_동리['총면적'] = round2(토지계_4합계_동리['총면적'] / (10**6), 3)

토지계_4합계_동리 = subtotal_dongri(토지계_4합계_동리, class_col="분류")
토지계_4합계_동리['분류'] = pd.Categorical(토지계_4합계_동리['분류'], categories=cat_order_land, ordered=True)

토지계_4합계_동리 = order_func(토지계_4합계_동리, value_col="총면적", class_col="분류")
토지계_4합계_동리.insert(토지계_4합계_동리.columns.get_loc('분류'), '오염원', '토지계')

# 토지계 면적은 소수점 3자리까지 표현(㎢ 단위)
year_cols_land_dongri = [c for c in 토지계_4합계_동리.columns if isinstance(c, int) or str(c).isdigit()]
토지계_4합계_동리[year_cols_land_dongri] = 토지계_4합계_동리[year_cols_land_dongri].fillna(0).round(3)

## 시군 기준 자료와 동리 기준 자료 합치기
토지계_4합계_동리 = bind_dongri(토지계_4합계, 토지계_4합계_동리)


# %% [markdown]
# ******************************************************************************
# ###  양식계
    
# 파일 불러오기
file_dir_aqua = BASE_DIR / "양식계"

files_aqua = [
    f for f in file_dir_aqua.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

df_list = []
for f in files_aqua:
    # R: skip = 3
    temp = pd.read_excel(f, skiprows=3, header=None)
    year = int(f.name[:4])
    temp.insert(0, '연도', year)
    
    # 2021년 이전 자료와 이후 자료의 열 위치(인덱스) 변화 대응
    if year < 2021:
        # 2021년 이전에 빈 열 3개가 없으므로 당겨진 인덱스 추출
        temp = temp[['연도', 2, 5, 6, 7, 13, 17, 23, 25]].copy()
    else:
        # 2021년 이후 기준 인덱스 추출
        temp = temp[['연도', 5, 8, 9, 10, 16, 20, 26, 28]].copy()
        
    # 추출 즉시 변수명 통일 (concat 에러 방지)
    temp.columns = [
        "연도", "업소명", "시군", "읍면동", "리", "분류", 
        "시설면적", "방류하천", "휴업"
    ]
    df_list.append(temp)

양식계_1원본 = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
## 변수명 지정 및 데이터 정리
양식계_2정리 = 양식계_1원본.copy()

# 시설면적, 연도 숫자로 지정 및 결측치 0 처리
양식계_2정리['연도'] = pd.to_numeric(양식계_2정리['연도'], errors='coerce')
양식계_2정리['시설면적'] = pd.to_numeric(양식계_2정리['시설면적'], errors='coerce').fillna(0)

# 지명 변경된 경우 변경 지명 적용 및 동리코드 추가
양식계_2정리 = dongri_code(양식계_2정리)

## 연도 및 동리별 시설면적 합계 정리
양식계_3동리 = 양식계_2정리.groupby(
    ['연도', '동리코드', '시군', '읍면동', '리', '분류'], dropna=False
)['시설면적'].sum().reset_index()

# ------------------------------------------------------------------------------
## 각 동리별 단위유역 점유율 계산 및 축종추가
share_양식계 = share_cal(
    source_col="양식계",
    class_list=["가두리", "유수식", "도전양식", "지수식"]
)

## 유역/시군 기준 합계 연도별 정리
years_df = pd.DataFrame({'연도': years})
base_df_양식계 = years_df.merge(share_양식계, how='cross')

양식계_4합계 = base_df_양식계.merge(
    양식계_3동리,
    on=["동리코드", "연도", "시군", "분류"],
    how='left'
)

# 동리별 시설면적 합계와 유역 점유율 계산 (소수점 둘째자리 반올림)
양식계_4합계['시설면적'] = round2(양식계_4합계['양식계'] * 양식계_4합계['시설면적'], 2)

양식계_4합계 = 양식계_4합계.groupby(
    ['연도', '단위유역', '시군', '분류']
)['시설면적'].sum().reset_index()

# 소계 계산
양식계_4합계 = subtotal(양식계_4합계, class_col="분류")

# 팩터(Categorical) 레벨 설정 및 Wide 포맷 변환
cat_order_aqua = ["가두리", "유수식", "도전양식", "지수식", "소계"]
양식계_4합계['분류'] = pd.Categorical(양식계_4합계['분류'], categories=cat_order_aqua, ordered=True)

양식계_4합계 = order_func(양식계_4합계, value_col="시설면적", class_col="분류")
양식계_4합계.insert(양식계_4합계.columns.get_loc('분류'), '오염원', '양식계_시설면적')

# 데이터 정수형 및 소수점 변환 처리(시설면적은 소수점 둘째자리까지 표현)
year_cols_aqua = [c for c in 양식계_4합계.columns if isinstance(c, int) or str(c).isdigit()]
양식계_4합계[year_cols_aqua] = 양식계_4합계[year_cols_aqua].fillna(0).round(2)

# ------------------------------------------------------------------------------
############################  동리 기준 자료 정리  #############################

양식계_4합계_동리 = base_df_양식계.merge(
    양식계_3동리,
    on=["동리코드", "연도", "시군", "분류"],
    how='left'
)

# 동리별 시설면적 합계와 유역 점유율 계산
양식계_4합계_동리['시설면적'] = round2(양식계_4합계_동리['양식계'] * 양식계_4합계_동리['시설면적'], 2)

양식계_4합계_동리 = 양식계_4합계_동리.groupby(
    ['연도', '단위유역', '시군', '읍면동', '리', '분류'], dropna=False
)['시설면적'].sum().reset_index()

# 소계 계산
양식계_4합계_동리 = subtotal_dongri(양식계_4합계_동리, class_col="분류")
양식계_4합계_동리['분류'] = pd.Categorical(양식계_4합계_동리['분류'], categories=cat_order_aqua, ordered=True)

# Wide 포맷 변환 및 오염원 열 추가
양식계_4합계_동리 = order_func(양식계_4합계_동리, value_col="시설면적", class_col="분류")
양식계_4합계_동리.insert(양식계_4합계_동리.columns.get_loc('분류'), '오염원', '양식계_시설면적')

# 데이터 정수형 및 소수점 변환 처리(시설면적은 소수점 둘째자리까지 표현)
year_cols_aqua_dongri = [c for c in 양식계_4합계_동리.columns if isinstance(c, int) or str(c).isdigit()]
양식계_4합계_동리[year_cols_aqua_dongri] = 양식계_4합계_동리[year_cols_aqua_dongri].fillna(0).round(2)

## 시군 기준 자료와 동리 기준 자료 합치기
양식계_4합계_동리 = bind_dongri(양식계_4합계, 양식계_4합계_동리)


# %% [markdown]
# ******************************************************************************
# ###  매립계
    
## 매립장 현황 데이터 불러오기
LANDFILL_STATUS_PATH = BASE_DIR / "매립계" / "매립장 현황" / "매립장 현황.xlsx"
if not LANDFILL_STATUS_PATH.exists():
    raise FileNotFoundError(f"[오류] 매립장 현황 파일 누락: '{LANDFILL_STATUS_PATH}' 파일이 존재하지 않습니다.")

매립장현황 = pd.read_excel(LANDFILL_STATUS_PATH)[['매립시설명', '시군', '단위유역']]

# 매립계 집계를 위한 기본 뼈대(Base DataFrame) 생성 (단위유역, 시군, 연도 교차조합)
# R의 반복문(for i in years)을 대체하여 연산 속도 최적화
share_base_매립 = share[['단위유역', '시군']].drop_duplicates()
years_df = pd.DataFrame({'연도': years})
base_df_매립 = years_df.merge(share_base_매립, how='cross')


##### ===== 매립계 - 시설수 ====================================================

## ***** 파일 불러오기  *******************************************************
file_dir_landfill = BASE_DIR / "매립계"

files_landfill = [
    f for f in file_dir_landfill.glob("*.xls*") 
    if re.match(r"^\d{4}", f.name) and int(f.name[:4]) in years
]

df_list = []
for f in files_landfill:
    # R: skip=3, col_names=F
    temp = pd.read_excel(f, skiprows=3, header=None)
    year = int(f.name[:4])
    temp.insert(0, '연도', year)
    
    # R 열 선택: select(연도, 2, 6:8, 18) -> Python Index: 0, 1, 5, 6, 7, 17
    temp = temp.iloc[:, [0, 1, 5, 6, 7, 17]].copy()
    temp.columns = ["연도", "매립시설명", "시군", "읍면동", "리", "가동유무"]
    df_list.append(temp)

매립계_시설수_1원본 = pd.concat(df_list, ignore_index=True)

## 변수명 지정 및 데이터 정리
매립계_시설수_2정리 = 매립계_시설수_1원본.copy()

# 지명 변경 적용 및 동리코드 추가
매립계_시설수_2정리 = dongri_code(매립계_시설수_2정리)

# 단위유역 추가 및 연도 숫자로 변환
매립계_시설수_2정리 = 매립계_시설수_2정리.merge(매립장현황, on=["매립시설명", "시군"], how='left')
매립계_시설수_2정리['연도'] = pd.to_numeric(매립계_시설수_2정리['연도'], errors='coerce')

## 유역/시군 기준 매립장 시설수 연도별 정리
# 베이스 데이터프레임과 병합하여 시설수 카운트
매립계_시설수_3합계 = base_df_매립.merge(
    매립계_시설수_2정리,
    on=['단위유역', '시군', '연도'],
    how='left'
)

# 시설수 산정: 매립시설명이 결측치가 아닌(존재하는) 데이터의 개수를 셈
매립계_시설수_3합계['시설수'] = 매립계_시설수_3합계['매립시설명'].notna().astype(int)

매립계_시설수_3합계 = 매립계_시설수_3합계.groupby(
    ['연도', '단위유역', '시군']
)['시설수'].sum().reset_index()

## 소계 계산, 단위유역/시군 순서 지정, 연도 기준 wide 포맷 변환
매립계_시설수_3합계 = subtotal(매립계_시설수_3합계)
매립계_시설수_3합계 = order_func(매립계_시설수_3합계, value_col="시설수")

# 오염원 및 분류 지정
idx_basin_landfill = 매립계_시설수_3합계.columns.get_loc('단위유역')
매립계_시설수_3합계.insert(idx_basin_landfill + 1, '오염원', '매립계_시설수')
매립계_시설수_3합계.insert(idx_basin_landfill + 2, '분류', '소계')

# 정수형 변환
year_cols_landfill = [c for c in 매립계_시설수_3합계.columns if isinstance(c, int) or str(c).isdigit()]
매립계_시설수_3합계[year_cols_landfill] = 매립계_시설수_3합계[year_cols_landfill].fillna(0).astype(int)


##### ===== 매립계 - 침출수 발생유량 ===========================================

## ***** 파일 불러오기  *******************************************************
df_list = []
for f in files_landfill:
    # R: sheet=2, skip=2, col_names=F
    # Python: sheet_name=1 (0-based)
    temp = pd.read_excel(f, sheet_name=1, skiprows=2, header=None)
    year = int(f.name[:4])
    temp.insert(0, '연도', year)
    
    # R 열 선택: select(연도, 2, 5) -> Python Index: 0, 1, 4
    temp = temp.iloc[:, [0, 1, 4]].copy()
    temp.columns = ["연도", "매립시설명", "발생유량"]
    df_list.append(temp)

매립계_침출수_1원본 = pd.concat(df_list, ignore_index=True)

## 변수명 지정
매립계_침출수_2정리 = 매립계_침출수_1원본.copy()

## 침출수 발생유량 연평균 계산
매립계_침출수_2정리['연도'] = pd.to_numeric(매립계_침출수_2정리['연도'], errors='coerce')
매립계_침출수_2정리['발생유량'] = pd.to_numeric(매립계_침출수_2정리['발생유량'], errors='coerce')

# 매립시설명 및 연도별 평균 계산 후 반올림 (소수점 첫째자리)
매립계_침출수_3연평균 = 매립계_침출수_2정리.groupby(
    ['매립시설명', '연도']
)['발생유량'].mean().reset_index()
매립계_침출수_3연평균['발생유량'] = round2(매립계_침출수_3연평균['발생유량'], 1).fillna(0)

## 침출수 발생유량 자료에 시군, 단위유역 추가
매립계_침출수_3연평균 = 매립계_침출수_3연평균.merge(매립장현황, on='매립시설명', how='left')
매립계_침출수_3연평균 = 매립계_침출수_3연평균[매립계_침출수_3연평균['시군'].notna()].copy()

## 유역/시군 기준 침출수 발생유량 연도별 정리
매립계_침출수_4합계 = base_df_매립.merge(
    매립계_침출수_3연평균,
    on=['단위유역', '시군', '연도'],
    how='left'
)

# 결측치 0 변환 후 그룹바이 합계
매립계_침출수_4합계['발생유량'] = 매립계_침출수_4합계['발생유량'].fillna(0)
매립계_침출수_4합계 = 매립계_침출수_4합계.groupby(
    ['연도', '단위유역', '시군']
)['발생유량'].sum().reset_index()

## 소계 계산, 단위유역/시군 순서 지정, 연도 기준 wide 포맷 변환
매립계_침출수_4합계 = subtotal(매립계_침출수_4합계)
매립계_침출수_4합계 = order_func(매립계_침출수_4합계, value_col="발생유량")

# 오염원 및 분류 열 추가
idx_basin_leachate = 매립계_침출수_4합계.columns.get_loc('단위유역')
매립계_침출수_4합계.insert(idx_basin_leachate + 1, '오염원', '매립계_침출수발생량')
매립계_침출수_4합계.insert(idx_basin_leachate + 2, '분류', '소계')

# 데이터 정수형 및 소수점 변환 처리(발생유량은 소수점 첫째자리까지 표현)
year_cols_leachate = [c for c in 매립계_침출수_4합계.columns if isinstance(c, int) or str(c).isdigit()]
매립계_침출수_4합계[year_cols_leachate] = 매립계_침출수_4합계[year_cols_leachate].fillna(0).round(1)


# %% [markdown]
# ******************************************************************************
# ###  전체 통합 자료 정리
    
#####  최종 데이터 정리 함수  --------------------------------------------------
def data_total(df):
    data = df.copy()
    
    # 1. 특정 분류 제외
    exclude_classes = ["1종", "2종", "3종", "4종", "5종", "가두리", "도전양식", "유수식", "지수식"]
    data = data[~data['분류'].isin(exclude_classes)]
    
    # 2. 팩터(Categorical) 레벨 설정
    source_levels = [
        "생활계", "축산계", "산업계_업소수", "산업계_폐수발생량", "산업계_폐수방류량",
        "토지계", "양식계_시설면적", "매립계_시설수", "매립계_침출수발생량"
    ]
    class_levels = [
        "인구", "물사용량", "젖소", "한우", "말", "돼지", "양, 사슴", "개", "가금",
        "전", "답", "과수원", "기타초지", "임야", "기타", "대지", "공장용지",
        "공공시설지역", "교통지역", "주유소용지", "체육용지", "유원지", "소계"
    ]
    
    data['오염원'] = pd.Categorical(data['오염원'], categories=source_levels, ordered=True)
    data['분류'] = pd.Categorical(data['분류'], categories=class_levels, ordered=True)
    
    # 3. 데이터 정렬 (arrange)
    data = data.sort_values(by=['시군', '단위유역', '오염원', '분류'], kind='stable').reset_index(drop=True)
    
    return data


#####  시군 기준 최종 데이터 합치기 --------------------------------------------

## 전체 계별 데이터 합치기
데이터통합 = pd.concat([
    생활계_인구_4합계, 생활계_물사용량_4합계, 축산계_4합계,
    산업계_4합계, 토지계_4합계, 양식계_4합계, 매립계_시설수_3합계,
    매립계_침출수_4합계
], ignore_index=True)

## 최종 데이터 정리
데이터통합 = data_total(데이터통합)


## 수질개선사업계획 추진실적 기준으로 정리(기타수계 및 시행계획 지역 제외)
exclude_sigun = ["강원도", "동해시", "속초시", "양양군"]
exclude_basin = ["기타", "소계", "북한D", "임진A"]

데이터통합_추진실적 = 데이터통합[
    (~데이터통합['시군'].isin(exclude_sigun)) & 
    (~데이터통합['단위유역'].isin(exclude_basin))
].copy()


### 엑셀 파일 내보내기_to_excel
out_path_1 = OUTPUT_DIR / "전국오염원조사 자료 정리(강원도전체시군기준)_py.xlsx"
out_path_2 = OUTPUT_DIR / "전국오염원조사 자료 정리(총량대상시군기준)_py.xlsx"

데이터통합.to_excel(out_path_1, index=False)
데이터통합_추진실적.to_excel(out_path_2, index=False)

print(f"엑셀 파일 저장 완료:\n - {out_path_1.name}\n - {out_path_2.name}")


# %% [markdown]
# ******************************************************************************
# ### 계별 대표값 정리
    
# 동적 연도 추출 (초기 설정된 years 기준)
first_year = years[0]
last_year = years[-1]
col_last_year = f"{last_year}년"
col_2030 = "2030년"
diff_years = last_year - first_year
증감율_col_name = f'증감율_{str(first_year)[-2:]}_{str(last_year)[-2:]}'

## 1. 축산계 주요 축종 (소(젖소 + 한우) + 돼지 + 가금)
mask_cow = 축산계_4합계['분류'].isin(["젖소", "한우"])
축산계_소 = 축산계_4합계[mask_cow].groupby(['단위유역', '시군', '오염원'], dropna=False)[years].sum().reset_index()
축산계_소.insert(축산계_소.columns.get_loc('오염원') + 1, '분류', '소')

mask_pig_poultry = 축산계_4합계['분류'].isin(["돼지", "가금"])
축산계_4합계_2 = pd.concat([축산계_소, 축산계_4합계[mask_pig_poultry]], ignore_index=True)

## 2. 산업계 폐수 방류량
mask_ind_discharge = (산업계_4합계['오염원'] == "산업계_폐수방류량") & (산업계_4합계['분류'] == "소계")
산업계_4합계_2_폐수방류 = 산업계_4합계[mask_ind_discharge].copy()
산업계_4합계_2_폐수방류['오염원'] = "산업계"
산업계_4합계_2_폐수방류['분류'] = "폐수방류량"

## 3. 오염원 전망 계별 대표값 정리
# 오염원전망 데이터는 이전 토지계 파트에서 로드된 상태여야 함
mask_prospect_1 = 오염원전망['분류'].isin(["인구", "물사용량", "돼지", "가금"]) | (오염원전망['오염원'] == "산업계_폐수방류량")
오염원전망_part1 = 오염원전망[mask_prospect_1].copy()

mask_prospect_cow = 오염원전망['분류'].isin(["젖소", "한우"])
year_cols_prospect = [c for c in 오염원전망.columns if str(c).endswith('년') and str(c)[:4].isdigit()]
오염원전망_cow = 오염원전망[mask_prospect_cow].groupby(['시군', '단위유역', '오염원'], dropna=False)[year_cols_prospect].sum().reset_index()
오염원전망_cow.insert(오염원전망_cow.columns.get_loc('오염원') + 1, '분류', '소')

오염원전망_계별대표값 = pd.concat([오염원전망_part1, 오염원전망_cow], ignore_index=True)

# 문자열 치환 및 정리
오염원전망_계별대표값['단위유역'] = np.where(오염원전망_계별대표값['단위유역'] == "합계", "소계", 오염원전망_계별대표값['단위유역'])
오염원전망_계별대표값['오염원'] = np.where(오염원전망_계별대표값['오염원'] == "산업계_폐수방류량", "산업계", 오염원전망_계별대표값['오염원'])
오염원전망_계별대표값['분류'] = np.where(오염원전망_계별대표값['분류'] == "소계", "폐수방류량", 오염원전망_계별대표값['분류'])

# 팩터(Categorical) 레벨 적용 (Step 1의 리스트 활용)
오염원전망_계별대표값['단위유역'] = pd.Categorical(오염원전망_계별대표값['단위유역'], categories=["소계"] + ORDER_BASIN, ordered=True)
오염원전망_계별대표값['시군'] = pd.Categorical(오염원전망_계별대표값['시군'], categories=["강원도"] + ORDER_CITY, ordered=True)

# 필요한 연도 열(당해 마지막 연도 및 2030년)만 추출
오염원전망_계별대표값 = 오염원전망_계별대표값[['시군', '단위유역', '오염원', '분류', col_last_year, col_2030]].copy()


## 4. 계별대표값 합친 후 정리 (시군 기준)
계별_base = pd.concat([
    생활계_인구_4합계, 생활계_물사용량_4합계, 축산계_4합계_2, 산업계_4합계_2_폐수방류
], ignore_index=True)

if '권역' in 계별_base.columns:
    계별_base = 계별_base.drop(columns=['권역'])

exclude_sigun = ["강원도", "동해시", "속초시", "양양군"]
계별대표값_시군 = 계별_base[
    (~계별_base['시군'].isin(exclude_sigun)) &
    (계별_base['단위유역'] != "기타")
].copy()

계별대표값_시군 = 계별대표값_시군.merge(
    오염원전망_계별대표값,
    on=['시군', '단위유역', '오염원', '분류'],
    how='left'
)

cat_order_rep = ["인구", "물사용량", "소", "돼지", "가금", "폐수방류량", "농경지"]
계별대표값_시군['분류'] = pd.Categorical(계별대표값_시군['분류'], categories=cat_order_rep, ordered=True)

# 결측치, 무한대 값 방어(0 처리) 함수
def safe_calc(series):
    return series.replace([np.inf, -np.inf], np.nan).fillna(0).round(4)

# 동적 수식 계산
계별대표값_시군['증감율'] = safe_calc((계별대표값_시군[last_year] - 계별대표값_시군[col_2030]) / 계별대표값_시군[col_2030])

# CAGR (연평균 증감율)
# 주의: 음수값이 없고 분모가 0일때 발생할 inf, nan 모두 safe_calc로 처리됨
cagr_val = (계별대표값_시군[last_year] / 계별대표값_시군[first_year]) ** (1 / diff_years) - 1
계별대표값_시군['CAGR'] = safe_calc(cagr_val)

계별대표값_시군[증감율_col_name] = safe_calc((계별대표값_시군[last_year] - 계별대표값_시군[first_year]) / 계별대표값_시군[first_year])

# 정렬
계별대표값_시군 = 계별대표값_시군.sort_values(by=['시군', '분류', '단위유역']).reset_index(drop=True)


## 5. 유역 기준 계별대표값
계별대표값_유역 = 계별_base[
    (~계별_base['시군'].isin(exclude_sigun)) &
    (~계별_base['단위유역'].isin(["기타", "소계"]))
].copy()

# 시군 단위 "소계" 추가 (단위유역, 오염원, 분류별 그룹 합계)
sub_basin = 계별대표값_유역.groupby(['단위유역', '오염원', '분류'], dropna=False)[years].sum(numeric_only=True).reset_index()
sub_basin['시군'] = '소계'
계별대표값_유역 = pd.concat([계별대표값_유역, sub_basin], ignore_index=True)

# 팩터 레벨 지정
cat_order_sigun = ["소계"] + ORDER_CITY
계별대표값_유역['시군'] = pd.Categorical(계별대표값_유역['시군'], categories=cat_order_sigun, ordered=True)
계별대표값_유역['분류'] = pd.Categorical(계별대표값_유역['분류'], categories=cat_order_rep, ordered=True)

# 동적 증감율 계산
계별대표값_유역[증감율_col_name] = safe_calc((계별대표값_유역[last_year] - 계별대표값_유역[first_year]) / 계별대표값_유역[first_year])

# 정렬
계별대표값_유역 = 계별대표값_유역.sort_values(['단위유역', '분류', '시군']).reset_index(drop=True)


### 엑셀 파일 내보내기
out_path_rep_sigun = OUTPUT_DIR / "전국오염원조사자료 계별 대표값 정리(시군기준)_py.xlsx"
out_path_rep_basin = OUTPUT_DIR / "전국오염원조사자료 계별 대표값 정리(유역기준)_py.xlsx"

계별대표값_시군.to_excel(out_path_rep_sigun, index=False)
계별대표값_유역.to_excel(out_path_rep_basin, index=False)

print(f"계별 대표값 엑셀 저장 완료:\n - {out_path_rep_sigun.name}\n - {out_path_rep_basin.name}")


# %% [markdown]
# ******************************************************************************
# ### 동리 기준 데이터 정리
    
###  동리 기준 최종 데이터 합치기

## 전체 계별 데이터 합치기
# 매립계는 동리 구분이 없으므로 시군 단위 데이터(매립계_시설수_3합계, 매립계_침출수_4합계)를 그대로 병합
데이터통합_동리 = pd.concat([
    생활계_인구_4합계_동리, 생활계_물사용량_4합계_동리, 축산계_4합계_동리,
    산업계_4합계_동리, 토지계_4합계_동리, 양식계_4합계_동리,
    매립계_시설수_3합계, 매립계_침출수_4합계
], ignore_index=True)

## 최종 데이터 정리
데이터통합_동리 = data_total(데이터통합_동리)

# 매립계의 읍면동, 리를 "소계"로 일괄 변경
mask_landfill = 데이터통합_동리['오염원'].astype(str).str.startswith("매립계")
데이터통합_동리.loc[mask_landfill, '읍면동'] = "소계"
데이터통합_동리.loc[mask_landfill, '리'] = "소계"

# ★ '소계'를 최상단에 두고 나머지는 완벽한 가나다순 정렬
# 1. 읍면동/리 고유값 추출 (소계 및 빈칸 제외)
unique_emd = [x for x in 데이터통합_동리['읍면동'].dropna().unique() if x != "소계"]
unique_ri = [x for x in 데이터통합_동리['리'].dropna().unique() if x not in ["소계", ""]]

# 2. 파이썬 내장 sorted() 함수를 이용해 명시적으로 가나다순 정렬
sorted_emd = sorted(unique_emd)
sorted_ri = sorted(unique_ri)

# 3. Categorical 레벨 설정 ("소계" -> 빈칸 -> 가나다순)
cat_eupmyeondong = ["소계"] + sorted_emd
cat_ri = ["소계", ""] + sorted_ri

데이터통합_동리['읍면동'] = pd.Categorical(데이터통합_동리['읍면동'], categories=cat_eupmyeondong, ordered=True)
데이터통합_동리['리'] = pd.Categorical(데이터통합_동리['리'], categories=cat_ri, ordered=True)

# 중복 제거 및 최종 정렬 (Categorical 순서가 강제 적용됨)
데이터통합_동리 = 데이터통합_동리.drop_duplicates().sort_values(
    by=['시군', '단위유역', '읍면동', '리', '오염원', '분류']
).reset_index(drop=True)

## 엑셀 파일 내보내기
out_path_dongri = OUTPUT_DIR / "전국오염원조사 자료 정리(동리기준)_py.xlsx"
데이터통합_동리.to_excel(out_path_dongri, index=False)
print(f"동리 기준 엑셀 저장 완료: {out_path_dongri.name}")


# %% [markdown]
# ******************************************************************************
# ### 철원군 추진실적 및 이행평가 오염원 조정
    
### 철원군 추진실적 및 이행평가 오염원 조정
cheorwon_ri = [
    "화지리", "사요리", "외촌리", "율이리", "내포리", "대마리",
    "중세리", "산명리", "가단리", "유정리", "홍원리", "독검리",
    "지포리", "강포리", "이평리", "상노리", "중강리"
]

철원군 = 데이터통합_동리[
    (데이터통합_동리['시군'] == "철원군") &
    (~데이터통합_동리['단위유역'].isin(["기타", "소계"])) &
    (데이터통합_동리['리'].isin(cheorwon_ri))
].copy()

# 권역 등 불필요한 열 제외 및 마지막 연도(last_year) 기준 피벗
index_cols_cheorwon = ['오염원', '분류', '리']
철원군_pivot = 철원군.pivot_table(
    index=index_cols_cheorwon,
    columns='단위유역',
    values=last_year,
    aggfunc='sum'
).reset_index()

# 결측치를 빈칸으로 처리
철원군_pivot = 철원군_pivot.fillna('')

# 열 순서 재배치 (한탄A를 맨 뒤로)
cols = list(철원군_pivot.columns)
if '한탄A' in cols:
    cols.remove('한탄A')
    cols.append('한탄A')
철원군_pivot = 철원군_pivot[cols]

# 리 순서 팩터 지정 및 정렬
철원군_pivot['리'] = pd.Categorical(철원군_pivot['리'], categories=cheorwon_ri, ordered=True)
철원군_pivot = 철원군_pivot.sort_values(by=['오염원', '분류', '리']).reset_index(drop=True)

## 엑셀 파일 내보내기
out_path_cheorwon = OUTPUT_DIR / "철원군_오염원조정자료_py.xlsx"
철원군_pivot.to_excel(out_path_cheorwon, index=False)
print(f"철원군 조정 자료 엑셀 저장 완료: {out_path_cheorwon.name}")


# %%
