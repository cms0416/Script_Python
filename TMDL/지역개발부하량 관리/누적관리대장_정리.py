# %% [markdown]
# # 지역개발부하량 관리 및 분석


# %% [markdown]
# ******************************************************************************
# ### 라이브러리 및 함수 설정


# %% ---------------------------------------------------------------------------
# 1. 라이브러리 로드 및 환경 설정
import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display # 데이터프레임 이쁘게 출력

# 경로 설정(실제 환경에 맞게 수정 필요)
BASE_DIR = Path("C:/Coding/TMDL/지역개발부하량 관리")
OUTPUT_DIR = BASE_DIR / "Output"
PLOT_DIR = OUTPUT_DIR / "Plot"

# 디렉토리 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 기준년도 설정
REF_YEAR = 2025

# 한글 폰트 설정 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(f"설정 완료. 기준년도: {REF_YEAR}")
print(f"작업 경로: {BASE_DIR.absolute()}")

# %% ---------------------------------------------------------------------------
# 2. 유틸리티 함수 정의

# 정렬 순서 상수 정의
ORDER_CITY = [
    "강원도", "강원도(한강)", "춘천시", "원주시", "강릉시", "태백시(낙동강)",
    "태백시(한강)", "삼척시", "홍천군", "횡성군", "영월군", "평창군",
    "정선군", "철원군", "화천군", "양구군", "인제군", "고성군"
]

ORDER_BASIN_INTERNAL = [
    "합계", "소계", "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A",
    "섬강A", "섬강B", "북한A", "북한B", "소양A", "인북A", "소양B", "북한C",
    "홍천A", "한탄A", "제천A", "한강B", "한강D", "북한D", "임진A", "한탄B",
    "낙본A"
]

ORDER_BASIN_PERFORMANCE = [
    "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A", "섬강A", "섬강B",
    "북한A", "북한B", "소양A", "인북A", "소양B", "북한C", "홍천A", "한탄A",
    "제천A", "한강B", "한강D", "북한D", "임진A", "한탄B", "낙본A",
    "소계", "합계"
]

def safe_numeric_converter(x):
    """R의 to_num 함수 대체: 문자열, 논리형 등을 숫자로 안전하게 변환"""
    if pd.isna(x) or x in ["", "/"]:
        return 0
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, bool):
        return 1 if x else 0
    
    s = str(x).strip().replace(",", "")
    # 숫자 형태(음수, 소수점 포함)인지 정규식으로 확인
    if re.match(r'^[-+]?\d+(\.\d+)?$', s):
        return float(s)
    return 0

def add_subtotal(df, group_cols, value_cols, label="소계", fill_col=None):
    """그룹별 소계 행 추가(janitor::adorn_totals 대체)"""
    subtotals = df.groupby(group_cols)[value_cols].sum().reset_index()
    if fill_col:
        subtotals[fill_col] = label
    return pd.concat([df, subtotals], ignore_index=True)

def sort_dataframe(df, city_col='시군', basin_col='단위유역', mode='internal'):
    """데이터프레임 정렬(Categorical 활용)"""
    df = df.copy()
    df[city_col] = pd.Categorical(df[city_col], categories=ORDER_CITY, ordered=True)
    basin_order = ORDER_BASIN_INTERNAL if mode == 'internal' else ORDER_BASIN_PERFORMANCE
    df[basin_col] = pd.Categorical(df[basin_col], categories=basin_order, ordered=True)
    return df.sort_values(by=[city_col, basin_col])

def load_cal(df):
    """부하량 자료 정리 및 피벗"""
    if df.empty:
        # 빈 데이터프레임 처리 (오류 방지)
        cols = ['시군', '대상물질', '단위유역', '소진_점', '소진_비점']
        return pd.DataFrame(columns=cols)

    cols_to_sum = ['BOD.소진_점', 'BOD.소진_비점', 'TP.소진_점', 'TP.소진_비점']
    
    # 1. 그룹별 합계
    grouped = df.groupby(['시군', '단위유역'])[cols_to_sum].sum().reset_index()
    
    # 2. Melt & Pivot
    melted = grouped.melt(id_vars=['시군', '단위유역'], value_vars=cols_to_sum, var_name='temp_var')
    melted[['대상물질', '항목']] = melted['temp_var'].str.split('.', expand=True)
    
    pivoted = melted.pivot_table(
        index=['시군', '단위유역', '대상물질'], 
        columns='항목', values='value', aggfunc='sum'
    ).reset_index()
    
    # 3. 소계 추가
    final_df = add_subtotal(
        pivoted, 
        group_cols=['시군', '대상물질'], 
        value_cols=['소진_점', '소진_비점'], 
        fill_col='단위유역'
    )
    
    # 4. 열 순서 조정
    final_df = final_df[['시군', '단위유역', '대상물질', '소진_점', '소진_비점']]
    
    # 5. 수치 데이터 반올림 (소수점 셋째 자리)
    numeric_cols = ['소진_점', '소진_비점']
    final_df[numeric_cols] = final_df[numeric_cols].round(3)
    
    return final_df

print("함수 정의 완료.")

# %% [markdown]
# ******************************************************************************
# ### 자료 전처리


# %% ---------------------------------------------------------------------------
# 3. 기초자료(Base) 로드 및 정리

base_path = BASE_DIR / "지역개발부하량.xlsx"

# 파일 존재 여부 확인
if not base_path.exists():
    print(f"[오류] 파일을 찾을 수 없습니다: {base_path}")
    print("경로를 확인하거나 파일을 해당 위치에 복사해주세요.")
else:
    base = pd.read_excel(base_path, sheet_name=0)
    if '항목정렬' in base.columns:
        base = base.drop(columns=['항목정렬'])
    
    print(f"기초자료 로드 완료. 크기: {base.shape}")

# base 마지막 4개 열 소수점 셋째자리 반올림
for col in base.columns[-4:]:
    if pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].round(3)
print("기초자료 소수점 반올림 완료.")

# 지역개발 연차별 관리계획 자료 로드
annual_plan = pd.read_excel(base_path, sheet_name="지역개발 연차별 관리계획")

# 지역개발 연차별 관리계획 컬럼명 정의
col_point = f"{REF_YEAR}_점"
col_nonpoint = f"{REF_YEAR}_비점"

# 지역개발 연차별 관리계획 필요한 열 선택
annual_plan = annual_plan[['시군', '단위유역', '대상물질', col_point, col_nonpoint]].copy()

# 기초자료(Base) 데이터 확인
base

# %% ---------------------------------------------------------------------------
# 4. 추진실적용 Base 데이터셋 구성

# 필터링: 북한D, 임진A 제외 / 강원도(합계) 제외
base_perf = base.copy()
base_perf = base_perf[
    (~base_perf['단위유역'].isin(["북한D", "임진A"])) & 
    (base_perf['시군'] != "강원도")
]

# 춘천시, 철원군 소계 행 처리
# 기존 소계 제외
mask_exclude = (base_perf['시군'].isin(["춘천시", "철원군"])) & (base_perf['단위유역'] == "소계")
base_perf = base_perf[~mask_exclude]

# 소계 재산정
chuncheon_cheorwon = base_perf[base_perf['시군'].isin(["춘천시", "철원군"])].copy()
subtotal_cc = chuncheon_cheorwon.groupby(['대상물질', '시군'])[['지역개발_점', '지역개발_비점', '기승인_점', '기승인_비점']].sum().reset_index()
subtotal_cc['단위유역'] = '소계'

# 원본 base에서 시군별 '시군정렬' 값을 중복 없이 추출
meta_info = base[['대상물질', '시군', '단위유역', '시군정렬']].drop_duplicates()

# 재산정된 소계 데이터프레임에 병합('대상물질', '시군', '단위유역' 기준)
subtotal_cc = pd.merge(subtotal_cc, meta_info, on=['대상물질', '시군', '단위유역'] , how='left')

# 재산정된 소계 병합
base_perf = pd.concat([base_perf, subtotal_cc], ignore_index=True)

# "시군정렬" 열 기준으로 정렬
base_perf = base_perf.sort_values(by=['시군정렬'])

# "시군정렬", "기승인_점", "기승인_비점" 열 제외
base_perf = base_perf.drop(columns=['시군정렬', '기승인_점', '기승인_비점'])

print(f"추진실적용 Base 구성 완료. 크기: {base_perf.shape}")

# 데이터 확인
base_perf

# %% ---------------------------------------------------------------------------
# 5. 누적관리대장(Raw Data) 파일 로드 및 통합

log_files = glob.glob(str(BASE_DIR / "누적관리대장*.xlsx"))
log_data_list = []

print(f"감지된 파일 개수: {len(log_files)}개")

for f in log_files:
    try:
        # 3행 건너뛰고 로드
        temp_df = pd.read_excel(f, header=None, skiprows=3)
        log_data_list.append(temp_df)
        print(f"- 로드 성공: {os.path.basename(f)}")
    except Exception as e:
        print(f"- [오류] {os.path.basename(f)} 로드 실패: {e}")

if log_data_list:
    raw_data = pd.concat(log_data_list, ignore_index=True)
    print(f"\n전체 통합 데이터 크기: {raw_data.shape}")
else:
    raw_data = pd.DataFrame()
    print("\n[경고] 누적관리대장 파일이 없습니다.")

display(raw_data.head(3))

# %% ---------------------------------------------------------------------------
# 6. 누적관리대장 전처리(숫자 변환 및 파생변수)

data_pre = raw_data.copy()

# BOD 삭감량 관련 인덱스 등
num_col_indices = [
    13, 18, 23, 28, 33,  # BOD 점
    40, 45, 50, 55, 60,  # BOD 비점
    89, 94, 99, 104, 109, # TP 점
    116, 121, 126, 131, 136, # TP 비점
    11, 63, 64, 69, 70, 87, 139, 140, 145, 146
]

# 삭감방법 컬럼 인덱스
cols_method_point = [12, 17, 22, 27, 32]
cols_method_nonpoint = [37, 42, 47, 52, 57]

# 1. 결측치 및 문자열 처리
for col in cols_method_point + cols_method_nonpoint:
    # 13(12)열, 38(37)열은 빈 문자열로, 나머지는 "/"로 채움
    fill_val = "" if col in [12, 37] else "/"
    data_pre[col] = data_pre[col].fillna(fill_val)

# 2. 숫자 변환 적용
for col_idx in num_col_indices:
    if col_idx < len(data_pre.columns):
        data_pre[col_idx] = data_pre[col_idx].apply(safe_numeric_converter)

# 3. 삭감방법 문자열 합치기
def concat_methods(row, indices):
    # 지정된 인덱스의 값을 쉼표로 연결
    vals = [str(row[i]) for i in indices]
    # 불필요한 문자 제거 및 앞뒤 공백 제거
    joined = ", ".join(vals).replace(", /", "").strip()
    if joined.startswith(","):
        joined = joined[1:].strip()
    return joined

data_pre['삭감방법_점'] = data_pre.apply(lambda x: concat_methods(x, cols_method_point), axis=1)
data_pre['삭감방법_비점'] = data_pre.apply(lambda x: concat_methods(x, cols_method_nonpoint), axis=1)

# 4. 삭감량 합계 계산
data_pre['BOD.삭감량_점'] = data_pre[[13, 18, 23, 28, 33]].sum(axis=1)
data_pre['BOD.삭감량_비점'] = data_pre[[40, 45, 50, 55, 60]].sum(axis=1)
data_pre['TP.삭감량_점'] = data_pre[[89, 94, 99, 104, 109]].sum(axis=1)
data_pre['TP.삭감량_비점'] = data_pre[[116, 121, 126, 131, 136]].sum(axis=1)

# 필요한 열 선택 및 최종 정리(Data Cleaning)
# 엑셀 열 위치 기반 선택 (주의: 엑셀 서식이 바뀌면 이 부분 수정 필요)
select_indices = [
    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 
    63, 64, 69, 70, 87, 
    139, 140, 145, 146,
    166, 167, 169, 170, 171, 172
]

final_cols = [
    "사업구분", "시군", "단위유역", "할당일자", "사업명", "사업위치", "착공연도",
    "준공예정연도", "준공여부", "BOD.삭감량_합계", "BOD.지역개발_점",
    "BOD.지역개발_비점", "BOD.소진_점", "BOD.소진_비점", "TP.삭감량_합계",
    "TP.지역개발_점", "TP.지역개발_비점", "TP.소진_점", "TP.소진_비점",
    "협의일자", "협의상태", "주관부서", "담당자명", "부서연락처", "사업종류"
]

# 인덱스 범위 체크 후 선택
available_indices = [i for i in select_indices if i < len(data_pre.columns)]
df_selected = data_pre.iloc[:, available_indices].copy()

# 컬럼 개수가 맞는지 확인 후 할당
if len(df_selected.columns) == len(final_cols):
    df_selected.columns = final_cols
else:
    print(f"[주의] 선택된 컬럼 수({len(df_selected.columns)})와 정의된 이름 수({len(final_cols)})가 다릅니다.")
    # 임시 방편: 개수에 맞춰 자르거나 추가 로직 필요

# 계산된 컬럼 붙이기
df_selected['삭감방법_점'] = data_pre['삭감방법_점']
df_selected['삭감방법_비점'] = data_pre['삭감방법_비점']
df_selected['BOD.삭감량_점'] = data_pre['BOD.삭감량_점']
df_selected['TP.삭감량_점'] = data_pre['TP.삭감량_점']
df_selected['BOD.삭감량_비점'] = data_pre['BOD.삭감량_비점']
df_selected['TP.삭감량_비점'] = data_pre['TP.삭감량_비점']

# 날짜 및 연도 처리
df_selected['협의일자'] = df_selected['협의일자'].fillna(df_selected['할당일자'])
for col in ['할당일자', '협의일자']:
    df_selected[col] = df_selected[col].astype(str).str.replace('.', '-', regex=False)
    df_selected[col] = pd.to_datetime(df_selected[col], errors='coerce')

df_selected['할당연도'] = df_selected['할당일자'].dt.year
df_selected['협의연도'] = df_selected['협의일자'].dt.year

# 최초 및 기승인 개발은 협의연도를 2020년으로 강제 설정
mask_plan = df_selected['사업구분'].isin(["기본계획_최초개발", "기본계획_기승인"])
df_selected.loc[mask_plan, '협의연도'] = 2020

# 시군 명칭 및 태백시 처리
df_selected['시군'] = df_selected['시군'].astype(str).str.replace("강원특별자치도 ", "").str.replace("강원도 ", "")
mask_taebaek_nak = (df_selected['시군'] == '태백시') & (df_selected['단위유역'] == '낙본A')
df_selected.loc[mask_taebaek_nak, '시군'] = '태백시(낙동강)'
mask_taebaek_han = (df_selected['시군'] == '태백시') & (df_selected['단위유역'] != '낙본A')
df_selected.loc[mask_taebaek_han, '시군'] = '태백시(한강)'

# 미협의 할당 및 보완 사업 제외
data_clean = df_selected[~df_selected['협의상태'].isin(["할당", "보완"])].copy()

print(f"데이터 정제 완료. 최종 데이터 건수: {len(data_clean)}")

display(data_clean.head(3))

# %% ---------------------------------------------------------------------------
# 7. 추진실적용 데이터(북한D, 임진A 제외)
data_perf = data_clean[~data_clean['단위유역'].isin(["북한D", "임진A"])].copy()

display(data_perf.head(3))

# %% [markdown]
# ******************************************************************************
# ### 추진실적용 자료 정리


# %% ---------------------------------------------------------------------------
# 8. [추진실적] 지역개발부하량 소진현황 산정

# 1. 협의부하량 (기준년도 이전)
consulted = data_perf[data_perf['협의연도'] < REF_YEAR].copy()
consulted_sum = load_cal(consulted)
consulted_sum = consulted_sum.rename(columns={'소진_점': '협의부하량_점', '소진_비점': '협의부하량_비점'})

# 2. 사용부하량 (기준년도)
used = data_perf[data_perf['협의연도'] == REF_YEAR].copy()
used_sum = load_cal(used)
used_sum = used_sum.rename(columns={'소진_점': '사용부하량_점', '소진_비점': '사용부하량_비점'})

# 3. 최종 병합 및 잔여량 계산
perf_status = base_perf.copy()
perf_status = perf_status.merge(consulted_sum, on=['시군', '대상물질', '단위유역'], how='left').fillna(0)

perf_status['협의가능량_점'] = perf_status['지역개발_점'] - perf_status['협의부하량_점']
perf_status['협의가능량_비점'] = perf_status['지역개발_비점'] - perf_status['협의부하량_비점']

# 사용 부하량 합치기
perf_status = perf_status.merge(used_sum, on=['시군', '대상물질', '단위유역'], how='left').fillna(0)

# 잔여부하량 계산
perf_status['잔여부하량_점'] = perf_status['협의가능량_점'] - perf_status['사용부하량_점']
perf_status['잔여부하량_비점'] = perf_status['협의가능량_비점'] - perf_status['사용부하량_비점']

# 누적부하량 계산(참고용)
perf_status['누적부하량_점'] = perf_status['협의부하량_점'] + perf_status['사용부하량_점']
perf_status['누적부하량_비점'] = perf_status['협의부하량_비점'] + perf_status['사용부하량_비점']

# '시군'을 첫번째 열로 이동
cols = perf_status.columns.tolist()
cols.insert(0, cols.pop(cols.index('시군')))
perf_status = perf_status[cols]

# 모든 수치 데이터 소수점 셋째 자리 반올림
numeric_cols = perf_status.select_dtypes(include=[np.number]).columns
perf_status[numeric_cols] = perf_status[numeric_cols].round(3)

print("추진실적 소진현황 계산 완료.")
perf_status


# %% ---------------------------------------------------------------------------
# 9. [추진실적] 협의 현황(사업 목록) 정리

# 필요한 열 선택
cols_to_select = [
    "시군", "단위유역", "사업구분", "사업명", "협의일자", "준공예정연도",
    "BOD.소진_점", "BOD.소진_비점", "TP.소진_점", "TP.소진_비점",
    "삭감방법_비점", "BOD.삭감량_비점", "TP.삭감량_비점",
    "삭감방법_점", "BOD.삭감량_점", "TP.삭감량_점", "BOD.삭감량_합계", "TP.삭감량_합계",
    "할당일자", "착공연도", "준공여부", "협의상태", "협의연도"
]

perf_list = data_perf[cols_to_select].copy()

# 협의건수 산정 (R의 add_count 대체)
perf_list['협의건수'] = perf_list.groupby(['시군', '단위유역', '사업명'])['사업명'].transform('size')

# 비고 조건 설정
# (주의: 숫자 비교를 위해 협의연도를 문자형으로 변환하기 전에 실행)
conditions = [
    (perf_list['사업구분'] == "기본계획_최초개발") & (perf_list['협의건수'] >= 2),
    (perf_list['사업구분'] == "기본계획_최초개발"),
    (perf_list['협의연도'] < REF_YEAR),
    (perf_list['협의연도'] == REF_YEAR)
]
choices = ["협의 부하량", "제외", "협의 부하량", "사용 부하량"]
perf_list['비고'] = np.select(conditions, choices, default="제외")

# 연도 관련 열 문자형으로 변환 (결측치는 빈 문자열 처리)
year_cols = ['준공예정연도', '착공연도', '협의연도']
for col in year_cols:
    perf_list[col] = perf_list[col].apply(
        lambda x: str(int(x)) if pd.notna(x) and str(x).replace('.0', '').isdigit() else str(x) if pd.notna(x) else ""
    )

# 삭감량 보정 (사업취소: 음수, 재협의: 0)
reduction_cols = [
    'BOD.삭감량_점', 'TP.삭감량_점', 'BOD.삭감량_비점', 'TP.삭감량_비점',
    'BOD.삭감량_합계', 'TP.삭감량_합계'
]
mask_cancel = perf_list['사업구분'] == "사업취소"
mask_re = perf_list['사업구분'] == "재협의"

for col in reduction_cols:
    perf_list.loc[mask_cancel, col] = perf_list.loc[mask_cancel, col] * -1
    perf_list.loc[mask_re, col] = 0

# 행 순서 조정을 위한 group_key 생성
perf_list['group_key'] = perf_list.groupby(['시군', '단위유역', '사업명'])['협의일자'].transform('max')

# 데이터 정렬
perf_list = sort_dataframe(perf_list, mode='performance')
perf_list = perf_list.sort_values(by=['시군', '단위유역', 'group_key', '사업명', '협의일자'])
perf_list = perf_list.drop(columns=['group_key'])

# 시군 및 유역별 소계 계산(모든 문자열 컬럼에 '소계' 채우기)

# 수치형 컬럼 식별
numeric_cols = perf_list.select_dtypes(include=[np.number]).columns.tolist()

# 그룹별 합계(소계) 계산(소수점 셋째자리 반올림)
subtotals = perf_list.groupby(['시군', '단위유역'])[numeric_cols].sum().reset_index()
subtotals[numeric_cols] = subtotals[numeric_cols].round(3)

# 비수치형(문자열 등) 컬럼 식별 및 '소계' 값 채우기
non_numeric_cols = [c for c in perf_list.columns if c not in numeric_cols and c not in ['시군', '단위유역']]

for col in non_numeric_cols:
    subtotals[col] = "소계"

# 정렬을 위한 임시 인덱스 부여 (원본은 0~N, 소계는 아주 큰 값)
perf_list['sort_idx'] = range(len(perf_list))
subtotals['sort_idx'] = 99999999  # 그룹의 맨 마지막에 위치하도록

# 병합 및 정렬
perf_list_combined = pd.concat([perf_list, subtotals], ignore_index=True)

# 시군/단위유역 순서 재지정(concat 시 카테고리 속성이 풀릴 수 있음)
perf_list_combined = sort_dataframe(perf_list_combined, mode='performance')

# 최종 정렬: 시군 > 단위유역 > (원본순서 vs 소계)
perf_list_combined = perf_list_combined.sort_values(by=['시군', '단위유역', 'sort_idx'])

# 임시 컬럼 제거 및 '협의건수' 위치 조정
perf_list_combined = perf_list_combined.drop(columns=['sort_idx'])
cols = perf_list_combined.columns.tolist()
if '협의건수' in cols:
    cols.remove('협의건수')
    cols.append('협의건수')
perf_list_combined = perf_list_combined[cols]

print("추진실적_사업목록(협의 현황) 계산 완료. 크기:", perf_list_combined.shape)
perf_list_combined

# %% ---------------------------------------------------------------------------
# 10. [추진실적] 준공현황 정리

# 1. 준공사업 배출부하량 정리(준공여부 <= 기준년도)
# 데이터 타입 안전 처리를 위해 임시 복사본 생성
temp_perf = data_perf.copy()

# 준공여부를 숫자로 변환(미준공, 문자열 등은 NaN 처리 후 필터링)
temp_perf['준공여부_num'] = pd.to_numeric(temp_perf['준공여부'], errors='coerce')

# 기준년도 이하인 사업 필터링
completed_projects = temp_perf[temp_perf['준공여부_num'] <= REF_YEAR].copy()

# 부하량 계산 (load_cal 함수 재사용)
perf_completed_sum = load_cal(completed_projects)
perf_completed_sum = perf_completed_sum.rename(columns={'소진_점': '준공_점', '소진_비점': '준공_비점'})

# 2. 지역개발 연차별 관리계획 정리
plan_perf = annual_plan.copy()

# 컬럼명 변경
plan_perf = plan_perf.rename(columns={
    f"{REF_YEAR}_점": "당해연도계획_점",
    f"{REF_YEAR}_비점": "당해연도계획_비점"
})

# 필요한 컬럼만 선택 (R 코드에는 없지만 명시적으로 선택하면 안전함)
plan_perf = plan_perf[['시군', '단위유역', '대상물질', '당해연도계획_점', '당해연도계획_비점']]

# 필터링: 북한D, 임진A 제외 / 강원도 제외
plan_perf = plan_perf[
    (~plan_perf['단위유역'].isin(["북한D", "임진A"])) & 
    (plan_perf['시군'] != "강원도")
]

# 춘천시, 철원군 소계 제외 (재산정을 위해)
mask_exclude_cc = (plan_perf['시군'].isin(["춘천시", "철원군"])) & (plan_perf['단위유역'] == "소계")
plan_perf = plan_perf[~mask_exclude_cc].copy()

# 조건부 값 변경: 한탄B 유역 & BOD -> 0 처리
mask_hantan = (plan_perf['단위유역'] == "한탄B") & (plan_perf['대상물질'] == "BOD")
plan_perf.loc[mask_hantan, ['당해연도계획_점', '당해연도계획_비점']] = 0

# 3. 춘천시, 철원군 소계 재산정 및 병합
# 해당 시군 데이터 추출
cc_data = plan_perf[plan_perf['시군'].isin(["춘천시", "철원군"])].copy()

# 그룹별 합계 계산
cc_subtotals = cc_data.groupby(['대상물질', '시군'])[['당해연도계획_점', '당해연도계획_비점']].sum().reset_index()
cc_subtotals['단위유역'] = "소계"

# 원본에 합치기
plan_perf = pd.concat([plan_perf, cc_subtotals], ignore_index=True)

# 4. 준공현황 및 연차별 관리계획 최종 병합 (Left Join)
# 연차별 계획을 기준으로 준공 실적을 붙임
perf_completion_status = pd.merge(
    plan_perf, 
    perf_completed_sum, 
    on=['시군', '단위유역', '대상물질'], 
    how='left'
)

# 결측치(NA)를 0으로 대체
perf_completion_status = perf_completion_status.fillna(0)

# 5. 정렬(시군, 대상물질, 단위유역)
# sort_dataframe을 호출하여 '시군', '단위유역'을 순서가 있는 Categorical 타입으로 변환
perf_completion_status = sort_dataframe(perf_completion_status, mode='performance')
# 시군, 대상물질, 단위유역 순으로 정렬
perf_completion_status = perf_completion_status.sort_values(by=['시군', '대상물질', '단위유역'])

# 모든 수치 데이터 소수점 셋째 자리 반올림
numeric_cols = perf_completion_status.select_dtypes(include=[np.number]).columns
perf_completion_status[numeric_cols] = perf_completion_status[numeric_cols].round(3)

print("준공현황 정리 완료. 크기:", perf_completion_status.shape)
display(perf_completion_status.head())

# %% ---------------------------------------------------------------------------
# 11. [추진실적] 준공 사업 내역 정리

# 필요한 열 선택
cols_completion_list = [
    "시군", "단위유역", "사업구분", "사업명", "협의일자", "준공여부",
    "BOD.소진_점", "BOD.소진_비점", "TP.소진_점", "TP.소진_비점", "사업위치",
    "BOD.삭감량_합계", "TP.삭감량_합계", "삭감방법_점", "삭감방법_비점"
]

# 준공여부 숫자 변환 확인을 위해 temp_perf 사용 (이미 숫자변환됨)
# 필터링: 준공 완료(<= 기준년도) AND 기본계획_최초개발 제외
completion_list = temp_perf[
    (temp_perf['준공여부_num'] <= REF_YEAR) & 
    (temp_perf['사업구분'] != "기본계획_최초개발")
].copy()

# 필요한 컬럼만 추출
completion_list = completion_list[cols_completion_list]

# 정렬
completion_list = sort_dataframe(completion_list, mode='performance')
completion_list = completion_list.sort_values(by=['시군', '단위유역', '준공여부', '사업명'])

# 시군 및 유역별 소계 계산 (group_modify + adorn_totals 대체)
# 1. 수치형 컬럼 식별
numeric_cols_comp = completion_list.select_dtypes(include=[np.number]).columns.tolist()

# 2. 소계 계산
comp_subtotals = completion_list.groupby(['시군', '단위유역'])[numeric_cols_comp].sum().reset_index()

# 3. 비수치형 컬럼 '소계' 채우기
non_numeric_cols_comp = [c for c in completion_list.columns if c not in numeric_cols_comp and c not in ['시군', '단위유역']]
for col in non_numeric_cols_comp:
    comp_subtotals[col] = "소계"

# 4. 정렬용 인덱스 생성 및 병합
completion_list['sort_idx'] = range(len(completion_list))
comp_subtotals['sort_idx'] = 99999999

completion_list_combined = pd.concat([completion_list, comp_subtotals], ignore_index=True)

# 5. 최종 정렬 및 정리
completion_list_combined = sort_dataframe(completion_list_combined, mode='performance')
completion_list_combined = completion_list_combined.sort_values(by=['시군', '단위유역', 'sort_idx'])
completion_list_combined = completion_list_combined.drop(columns=['sort_idx'])

# 6. 모든 수치 데이터 소수점 셋째 자리 반올림
# (소계 계산 후 최종 결과에 적용하여 오차 방지)
completion_list_combined[numeric_cols_comp] = completion_list_combined[numeric_cols_comp].round(3)

print("준공 사업 내역 정리 완료. 크기:", completion_list_combined.shape)
display(completion_list_combined.head())

# %% ---------------------------------------------------------------------------
# 12. [추진실적] 삭감 목표 정리

# 1. 엑셀 파일 로드 (R의 sheet=3 -> Python sheet_name=2)
# 경로: BASE_DIR / "지역개발부하량.xlsx"
try:
    reduction_target = pd.read_excel(base_path, sheet_name=2)
    print(f"삭감 목표 시트 로드 완료. 원본 크기: {reduction_target.shape}")
except Exception as e:
    print(f"[오류] 삭감 목표 시트(index=2) 로드 실패: {e}")
    # 시트 이름으로 시도할 경우를 대비해 예외 처리 가능

# 2. 동적 컬럼명 정의 (이미 정의되어 있지만 안전을 위해 재확인)
col_target_point = f"{REF_YEAR}_점"
col_target_nonpoint = f"{REF_YEAR}_비점"

# 3. 컬럼 선택 (R의 select(시군:삭감_비점, ...) 구현)
# '시군'부터 '삭감_비점'까지의 컬럼 슬라이싱
try:
    # 정적 컬럼 구간 선택
    static_cols_df = reduction_target.loc[:, '시군':'삭감_비점']
    
    # 동적 컬럼(당해연도 계획) 선택
    dynamic_cols_df = reduction_target[[col_target_point, col_target_nonpoint]]
    
    # 컬럼 병합
    reduction_target = pd.concat([static_cols_df, dynamic_cols_df], axis=1)
    
except KeyError as e:
    print(f"[오류] 컬럼 선택 중 에러 발생: {e}")
    print("엑셀 컬럼명을 확인해주세요:", reduction_target.columns.tolist())

# 4. 필터링 (R의 filter(!단위유역 %in% ...))
exclude_basins = ["북한D", "임진A", "소계", "합계"]
reduction_target = reduction_target[~reduction_target['단위유역'].isin(exclude_basins)].copy()

print("삭감 목표 정리 완료. 최종 크기:", reduction_target.shape)
reduction_target

# %% ---------------------------------------------------------------------------
# 13. [추진실적] 추진실적 자료 파일 내보내기

# 저장할 경로 설정
output_path = OUTPUT_DIR / "누적관리대장_정리(추진실적)_py.xlsx"

# 시트 이름과 저장할 데이터프레임 매핑
sheets_to_write = {
    "5-가. 개발사업 소진현황": perf_status,
    "5-나. 개발사업 협의현황": perf_list_combined,
    "5-다. 개발사업 준공현황": perf_completion_status,
    "5-라. 준공된 개발사업 내역": completion_list_combined,
    "6. 삭감계획 이행실적": reduction_target
}

try:
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_to_write.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"파일 저장 완료: {output_path}")

except PermissionError:
    print(f"[오류] 파일이 열려 있어 저장할 수 없습니다: {output_path}")
    print("엑셀 파일을 닫고 다시 실행해주세요.")
except Exception as e:
    print(f"[오류] 파일 저장 중 문제가 발생했습니다: {e}")

# %% [markdown]
# ******************************************************************************
# ### 내부 검토용 자료 정리

# %% ---------------------------------------------------------------------------
# 14. [내부검토] 소진 부하량 정리

# 소진 부하량 계산
internal_exhaust = load_cal(data_clean)

# 컬럼 재배치 (R: relocate(대상물질))
# '대상물질' 컬럼을 리스트의 맨 앞으로 이동시킵니다.
cols = internal_exhaust.columns.tolist()
if '대상물질' in cols:
    cols.remove('대상물질')
    cols.insert(0, '대상물질')
internal_exhaust = internal_exhaust[cols]

# 컬럼명 변경 (R: rename(소진량_점 = 소진_점, ...))
internal_exhaust = internal_exhaust.rename(columns={
    '소진_점': '소진량_점', 
    '소진_비점': '소진량_비점'
})

print("내부검토용 소진 부하량 정리 완료. 크기:", internal_exhaust.shape)
display(internal_exhaust.head())

# %% ---------------------------------------------------------------------------
# 15. [내부검토] 기승인 소진량 정리

# 1. 필터링 (사업구분 == "기본계획_기승인")
df_approved_filtered = data_clean[data_clean['사업구분'] == '기본계획_기승인'].copy()

# 2. 부하량 계산
internal_approved = load_cal(df_approved_filtered)

# 3. 컬럼 재배치 (대상물질을 맨 앞으로)
cols = internal_approved.columns.tolist()
if '대상물질' in cols:
    cols.remove('대상물질')
    cols.insert(0, '대상물질')
internal_approved = internal_approved[cols]

# 4. 컬럼명 변경
internal_approved = internal_approved.rename(columns={
    '소진_점': '기승인_점',
    '소진_비점': '기승인_비점'
})

print("내부검토용 기승인 소진량 정리 완료. 크기:", internal_approved.shape)
display(internal_approved.head())

# %% ---------------------------------------------------------------------------
# 16. [내부검토] 전년도까지 소진량 정리

# 1. 필터링 (기승인 제외 & 협의연도 < 기준년도)
df_prev_filtered = data_clean[
    (data_clean['사업구분'] != '기본계획_기승인') & 
    (data_clean['협의연도'] < REF_YEAR)
].copy()

# 2. 부하량 계산
internal_prev_year = load_cal(df_prev_filtered)

# 3. 컬럼 재배치 (대상물질을 맨 앞으로)
cols = internal_prev_year.columns.tolist()
if '대상물질' in cols:
    cols.remove('대상물질')
    cols.insert(0, '대상물질')
internal_prev_year = internal_prev_year[cols]

# 4. 컬럼명 변경
internal_prev_year = internal_prev_year.rename(columns={
    '소진_점': '이전_점', 
    '소진_비점': '이전_비점'
})

print("내부검토용 전년도까지 소진량 정리 완료. 크기:", internal_prev_year.shape)
display(internal_prev_year.head())

# %% ---------------------------------------------------------------------------
# 17. [내부검토] 당해연도 소진량 정리

# 1. 필터링 (기승인 제외 & 협의연도 == 기준년도)
df_current_filtered = data_clean[
    (data_clean['사업구분'] != '기본계획_기승인') & 
    (data_clean['협의연도'] == REF_YEAR)
].copy()

# 2. 부하량 계산
internal_current_year = load_cal(df_current_filtered)

# 3. 컬럼 재배치(대상물질을 맨 앞으로)
cols = internal_current_year.columns.tolist()
if '대상물질' in cols:
    cols.remove('대상물질')
    cols.insert(0, '대상물질')
internal_current_year = internal_current_year[cols]

# 4. 컬럼명 변경
internal_current_year = internal_current_year.rename(columns={
    '소진_점': '올해_점', 
    '소진_비점': '올해_비점'
})

print("내부검토용 당해연도 소진량 정리 완료. 크기:", internal_current_year.shape)
display(internal_current_year.head())

# %% ---------------------------------------------------------------------------
# 18. [내부검토] 자료 최종 정리 및 합계 계산

# 1. Base 데이터 준비 (불필요한 열 제거 및 필터링)
cols_to_drop = ['시군정렬', '기승인_점', '기승인_비점']
base_internal = base.drop(columns=[c for c in cols_to_drop if c in base.columns])
base_internal = base_internal[base_internal['시군'] != "강원도"].copy()

# 2. 데이터 병합 (Left Join 4회)
# 병합할 데이터프레임 목록
dfs_to_merge = [internal_exhaust, internal_approved, internal_prev_year, internal_current_year]

internal_result = base_internal
for df in dfs_to_merge:
    # 공통 키: ['대상물질', '시군', '단위유역']
    internal_result = pd.merge(
        internal_result, 
        df, 
        on=['대상물질', '시군', '단위유역'], 
        how='left'
    )

# 3. 결측치 0 처리
internal_result = internal_result.fillna(0)

# 4. 신규 및 잔여량 계산
internal_result['신규_점'] = internal_result['소진량_점'] - internal_result['기승인_점']
internal_result['신규_비점'] = internal_result['소진량_비점'] - internal_result['기승인_비점']
internal_result['잔여량_점'] = internal_result['지역개발_점'] - internal_result['소진량_점']
internal_result['잔여량_비점'] = internal_result['지역개발_비점'] - internal_result['소진량_비점']

# 5. 열 순서 조정
# 잔여량을 지역개발 뒤로 이동
cols = internal_result.columns.tolist()
# 기준 위치 찾기 (지역개발_비점)
ref_idx = cols.index('지역개발_비점') + 1
for col in ['잔여량_비점', '잔여량_점']: # 역순으로 insert
    if col in cols:
        cols.remove(col)
        cols.insert(ref_idx, col)
internal_result = internal_result[cols]

# 6. 강원도 전체 합계 및 한강수계 합계 계산
# 합산을 위한 수치형 컬럼 식별
cols_to_sum = internal_result.select_dtypes(include=[np.number]).columns.tolist()

# (1) 강원도 전체 합계 (단위유역 == '소계'인 행들의 합)
gw_total = internal_result[internal_result['단위유역'] == '소계'].groupby(['대상물질'])[cols_to_sum].sum().reset_index()
gw_total['시군'] = "강원도"
gw_total['단위유역'] = "합계" # R코드에서는 fill='소계'였으나, order_func에서 '합계'로 정렬되므로 맞춤

# (2) 강원도(한강) 합계 (태백(낙동강) 제외하고 단위유역 == '소계'인 행들의 합)
gw_han = internal_result[
    (internal_result['시군'] != "태백시(낙동강)") & 
    (internal_result['단위유역'] == '소계')
].groupby(['대상물질'])[cols_to_sum].sum().reset_index()
gw_han['시군'] = "강원도(한강)"
gw_han['단위유역'] = "합계"

# 7. 합계 행 병합
internal_result = pd.concat([internal_result, gw_total, gw_han], ignore_index=True)

# 8. 소진율 및 잔여율 계산 (합계 행 포함하여 일괄 계산)
# 0으로 나누는 경우(ZeroDivisionError) 방지를 위해 np.where 또는 replace 활용 가능하나, 여기선 바로 계산
internal_result['소진율.전체_점'] = (internal_result['소진량_점'] / internal_result['지역개발_점']).fillna(0).round(4)
internal_result['소진율.전체_비점'] = (internal_result['소진량_비점'] / internal_result['지역개발_비점']).fillna(0).round(4)

internal_result['잔여율_점'] = 1 - internal_result['소진율.전체_점']
internal_result['잔여율_비점'] = 1 - internal_result['소진율.전체_비점']

# 분모가 0일 경우(지역개발-기승인 = 0) Inf 발생 가능 -> 0 또는 NaN 처리 필요할 수 있음
denom_new_point = internal_result['지역개발_점'] - internal_result['기승인_점']
denom_new_nonpoint = internal_result['지역개발_비점'] - internal_result['기승인_비점']

# 안전한 나눗셈을 위해 0인 경우 NaN 처리 후 계산, 다시 0 채움
internal_result['소진율.신규_점'] = (internal_result['신규_점'] / denom_new_point.replace(0, np.nan)).fillna(0).round(4)
internal_result['소진율.신규_비점'] = (internal_result['신규_비점'] / denom_new_nonpoint.replace(0, np.nan)).fillna(0).round(4)

# 9. 열 재배치 및 정렬
# '올해_비점' 뒤로 이동할 4개 열
cols_to_move = ['소진율.전체_점', '소진율.전체_비점', '잔여율_점', '잔여율_비점']

# 기존 컬럼 리스트에서 이동할 4개 컬럼만 임시 제거
cols = [c for c in internal_result.columns if c not in cols_to_move]

# '올해_비점' 위치 찾기
if '올해_비점' in cols:
    insert_idx = cols.index('올해_비점') + 1
else:
    insert_idx = len(cols)

# 해당 위치에 이동 대상 컬럼 삽입
for i, col in enumerate(cols_to_move):
    cols.insert(insert_idx + i, col)
internal_result = internal_result[cols]

# 정렬
internal_result = sort_dataframe(internal_result, mode='internal')
# 정렬 순서: 대상물질(BOD/TP끼리) -> 시군(지정된 순서) -> 단위유역(지정된 순서)
internal_result = internal_result.sort_values(by=['대상물질', '시군', '단위유역'])

# 10. 수치 데이터 반올림 처리(소진율/잔여율: 4자리, 나머지: 3자리)
# 소수점 4자리 반올림 대상이 되는 전체 비율 열
rate_cols = ['소진율.전체_점', '소진율.전체_비점', '잔여율_점', '잔여율_비점', '소진율.신규_점', '소진율.신규_비점']

# 전체 수치형 컬럼 식별
all_numeric_cols = internal_result.select_dtypes(include=[np.number]).columns.tolist()

# 나머지 수치형 컬럼 식별 (전체 - 소진율/잔여율)
other_numeric_cols = [c for c in all_numeric_cols if c not in rate_cols]

# 반올림 적용
# (1) 나머지 수치 데이터 -> 3자리
internal_result[other_numeric_cols] = internal_result[other_numeric_cols].round(3)

# (2) 소진율/잔여율 컬럼 -> 4자리
internal_result[rate_cols] = internal_result[rate_cols].round(4)

print("내부검토용 최종 자료 정리 완료. 크기:", internal_result.shape)
display(internal_result.head(10))

# %% ---------------------------------------------------------------------------
# 19. [내부검토] 최종 자료 점/비점 행 기준으로 전환

# 1. Melt (Wide -> Long)
# 변환 대상 컬럼: 지역개발_점 부터 소진율.신규_비점 까지
# 데이터프레임의 컬럼 순서를 알기 어려우므로, 정규식으로 _점 또는 _비점 으로 끝나는 컬럼 선택
target_cols = [c for c in internal_result.columns if c.endswith('_점') or c.endswith('_비점')]
id_vars = [c for c in internal_result.columns if c not in target_cols]

melted = internal_result.melt(
    id_vars=id_vars,
    value_vars=target_cols,
    var_name='temp_var',
    value_name='value'
)

# 2. 구분 및 점비점 컬럼 분리
# 예: "지역개발_점" -> "지역개발", "점"
melted[['구분', '점비점']] = melted['temp_var'].str.rsplit('_', n=1, expand=True)

# 3. Pivot (Long -> Wide)
# 구분(지역개발, 소진량 등)을 컬럼으로 보냄
internal_result_row_base = melted.pivot_table(
    index=id_vars + ['점비점'],
    columns='구분',
    values='value',
    aggfunc='first' # 중복 없으므로 first
).reset_index()

# 4. 정렬 및 컬럼 순서 정리
internal_result_row_base = sort_dataframe(internal_result_row_base, mode='internal')

# 컬럼 순서 지정 (R 코드의 select 순서 반영)
# 대상물질:소진량, 소진율.전체:잔여율
desired_cols_order = [
    '대상물질', '시군', '단위유역', '점비점', 
    '지역개발', '잔여량', '소진량', '기승인', '이전', '올해',
    '소진율.전체', '잔여율', '신규', '소진율.신규' # 비율 컬럼
]

# 실제 존재하는 컬럼만 선택하여 순서 적용
final_cols = [c for c in desired_cols_order if c in internal_result_row_base.columns]
internal_result_row_base = internal_result_row_base[final_cols]

# 정렬
internal_result_row_base = sort_dataframe(internal_result_row_base, mode='internal')
# 정렬 순서: 대상물질(BOD/TP끼리) -> 시군(지정된 순서) -> 단위유역(지정된 순서)
# -> 점비점에서 점이 먼저 오도록 점비점은 순서가 있는 Categorical로 변환하여 정렬
internal_result_row_base['점비점'] = pd.Categorical(
    internal_result_row_base['점비점'], 
    categories=['점', '비점'], 
    ordered=True
)

internal_result_row_base = internal_result_row_base.sort_values(by=['대상물질', '시군', '단위유역', '점비점'])


print("점/비점 행 기준 전환 완료. 크기:", internal_result_row_base.shape)
display(internal_result_row_base.head(10))

# %% ---------------------------------------------------------------------------
# 20. [내부검토] 협의 현황 데이터 정리

# 1. 데이터 선택
cols_consult = [
    '사업구분', '시군', '단위유역', '사업명', '준공예정연도', '준공여부', 
    '협의상태', '협의연도', '협의일자', '사업종류'
]
consultation_list = data_clean[cols_consult].copy()

# 2. 사업구분 재매핑
# 기본계획_기승인 -> 기승인, 기본계획_최초개발 -> 최초개발, 시행계획_최초개발 -> 기승인
remap_dict = {
    "기본계획_기승인": "기승인",
    "기본계획_최초개발": "최초개발",
    "시행계획_최초개발": "기승인"
}
consultation_list['사업구분'] = consultation_list['사업구분'].replace(remap_dict)

# 3. 환경영향평가 여부 확인 (R: str_detect)
# 사업종류에 '환경영향평가' 문자열이 포함되어 있으면 "O", 아니면 "X"
consultation_list['환경영향평가'] = np.where(
    consultation_list['사업종류'].astype(str).str.contains("환경영향평가"), 
    "O", 
    "X"
)

print("협의 현황 데이터 정리 완료. 크기:", consultation_list.shape)
display(consultation_list.head())

# %% ---------------------------------------------------------------------------
# 21. [내부검토] 협의 건수 정리 함수 정의

def calculate_consultation_counts(data, base_df):
    """
    협의 건수 집계, 피벗, 합계 계산 함수
    data: 분석 대상 데이터 (consultation_list)
    base_df: 전체 시군/단위유역 구조를 잡기 위한 기초 데이터
    """
    # 1. 그룹별 개수 집계 및 피벗 (Long -> Wide)
    # R: pivot_wider(names_from = 사업구분, values_from = 개수, values_fill = 0)
    pivot_df = data.pivot_table(
        index=['시군', '단위유역'], 
        columns='사업구분', 
        aggfunc='size', 
        fill_value=0
    ).reset_index()
    
    # 2. 필요한 컬럼 존재 여부 확인 및 생성 (없으면 0으로 채움)
    target_cols = ['기승인', '신규', '기간외소진', '재협의', '사업취소']
    for col in target_cols:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
            
    # 컬럼 순서 지정 ('최초개발' 등 제외)
    pivot_df = pivot_df[['시군', '단위유역'] + target_cols].copy()
    
    # 3. 총사업건수 계산
    # 기승인 + 신규 + 기간외소진 - 사업취소
    pivot_df['총사업건수'] = (
        pivot_df['기승인'] + pivot_df['신규'] + pivot_df['기간외소진'] - pivot_df['사업취소']
    )
    
    # 4. 구조 보정 (빈 유역 포함)
    # base 데이터에서 'BOD' 항목만 추출하여 전체 시군/단위유역 뼈대 생성
    # R: base %>% filter(대상물질 == "BOD") %>% select(시군, 단위유역)
    # base 데이터에서 '소계', '합계', '강원도' 등을 제외하고 순수 유역만 추출
    structure = base_df[
        (base_df['대상물질'] == 'BOD') & 
        (base_df['단위유역'] != '소계') & 
        (base_df['단위유역'] != '합계') &
        (base_df['시군'] != '강원도') &
        (base_df['시군'] != '강원도(한강)')
    ][['시군', '단위유역']].drop_duplicates()
    
    # Merge로 빈 유역도 포함되도록 함
    result = pd.merge(structure, pivot_df, on=['시군', '단위유역'], how='left').fillna(0)
    
    # 5. 합계 계산을 위한 수치형 컬럼
    numeric_cols = target_cols + ['총사업건수']
    
    # (1) 시군별 소계 계산
    city_subtotals = result.groupby('시군')[numeric_cols].sum().reset_index()
    city_subtotals['단위유역'] = "소계"
    
    # (2) 강원도 전체 합계 계산
    province_total = result[numeric_cols].sum().to_frame().T
    province_total['시군'] = "강원도"
    province_total['단위유역'] = "합계"
    
    # 6. 병합 (원본 + 시군소계 + 도합계)
    final_result = pd.concat([result, city_subtotals, province_total], ignore_index=True)
       
    # 7. '총사업건수' 열 위치를 '단위유역' 열 뒤로 이동
    cols = final_result.columns.tolist()
    if '총사업건수' in cols:
        cols.remove('총사업건수')
        insert_idx = cols.index('단위유역') + 1
        cols.insert(insert_idx, '총사업건수')
        final_result = final_result[cols]
    
    # 8. 모든 수치 데이터 정수형(int)으로 변환
    final_result[numeric_cols] = final_result[numeric_cols].astype(int)
    
    return final_result

print("협의 건수 정리 함수(calculate_consultation_counts) 정의 완료.")

# %% ---------------------------------------------------------------------------
# 22. [내부검토] 시군 및 유역별 순서 정리 및 협의 건수 정리

# 1. 전체 협의 현황
# 함수 적용
consultation_status = calculate_consultation_counts(consultation_list, base)

# 정렬 (sort_dataframe 활용)
consultation_status = sort_dataframe(consultation_status, mode='internal')

# 2. 환경영향평가 대상 사업 협의 현황
# 필터링 후 함수 적용
consultation_status_eia = calculate_consultation_counts(
    consultation_list[consultation_list['환경영향평가'] == "O"], 
    base
)

# 정렬
consultation_status_eia = sort_dataframe(consultation_status_eia, mode='internal')

print("협의 현황 집계 완료.")
print(f"- 전체 협의 현황 크기: {consultation_status.shape}")
print(f"- 환경영향평가 협의 현황 크기: {consultation_status_eia.shape}")

display(consultation_status.head(10)) # 확인을 위해 head 출력

# %%
# %% ---------------------------------------------------------------------------
# 23. [내부검토] 준공 건수 정리

# 1. 데이터 필터링 및 전처리
# R: filter(준공여부 != "미준공", 준공여부 > 2020)
temp_df = data_clean.copy()

# 준공여부 숫자 변환 (미준공 등 문자는 NaN 처리)
temp_df['준공여부_num'] = pd.to_numeric(temp_df['준공여부'], errors='coerce')

# 2020년 초과 사업만 필터링 & 연도를 정수형으로 변환
target_data = temp_df[
    (temp_df['준공여부_num'].notna()) & 
    (temp_df['준공여부_num'] > 2020)
].copy()

target_data['준공연도'] = target_data['준공여부_num'].astype(int)

# 2. 피벗 (Pivot)
# R: pivot_wider(names_from = 준공여부, values_from = 개수, ...)
pivot_df = target_data.pivot_table(
    index=['시군', '단위유역'],
    columns='준공연도',
    aggfunc='size',
    fill_value=0
).reset_index()

# 3. 연도별 합계 계산 (Row Total)
# R: adorn_totals(where = "col", name = "합계")
# 연도 컬럼만 식별 (시군, 단위유역 제외)
year_cols = [c for c in pivot_df.columns if isinstance(c, int)]
pivot_df['합계'] = pivot_df[year_cols].sum(axis=1)

# 4. 구조 보정 (빈 유역 포함)
# base 데이터에서 순수 유역만 추출 (소계, 합계 등 제외)
structure = base[
    (base['대상물질'] == 'BOD') & 
    (base['단위유역'] != '소계') & 
    (base['단위유역'] != '합계') &
    (base['시군'] != '강원도') &
    (base['시군'] != '강원도(한강)')
][['시군', '단위유역']].drop_duplicates()

# Merge (Left Join)
result = pd.merge(structure, pivot_df, on=['시군', '단위유역'], how='left').fillna(0)

# 5. 시군별 소계 및 도 전체 합계 계산
numeric_cols = ['합계'] + sorted(year_cols)

# (1) 시군별 소계
city_subtotals = result.groupby('시군')[numeric_cols].sum().reset_index()
city_subtotals['단위유역'] = "소계"

# (2) 강원도 전체 합계
province_total = result[numeric_cols].sum().to_frame().T
province_total['시군'] = "강원도"
province_total['단위유역'] = "합계"

# 6. 병합 및 정렬
final_completion_counts = pd.concat([result, city_subtotals, province_total], ignore_index=True)

# 정렬 (sort_dataframe 사용)
final_completion_counts = sort_dataframe(final_completion_counts, mode='performance')

# 7. 열 순서 조정 및 정수형 변환
# 순서: 시군, 단위유역, 합계, 연도1, 연도2 ...
sorted_cols = ['시군', '단위유역', '합계'] + sorted(year_cols)
final_completion_counts = final_completion_counts[sorted_cols]

# 모든 수치 데이터 정수형(int) 변환
final_completion_counts[numeric_cols] = final_completion_counts[numeric_cols].astype(int)

print("준공 건수 정리 완료. 크기:", final_completion_counts.shape)
display(final_completion_counts.head(10)) # 확인을 위해 head 출력

# %% ---------------------------------------------------------------------------
# 24. [내부검토] 내부검토용 자료 파일 내보내기

# 1. 시군별 소진현황 요약표 생성 (Wide Format)
cols_summary_wide = [
    '대상물질', '시군', '단위유역', 
    '지역개발_점', '지역개발_비점', '잔여량_점', '잔여량_비점', '소진량_점', '소진량_비점',
    '소진율.전체_점', '소진율.전체_비점', '잔여율_점', '잔여율_비점'
]
cols_summary_wide = [c for c in cols_summary_wide if c in internal_result.columns]

city_exhaustion_status = internal_result[cols_summary_wide].copy()

# [수정] 강원도는 단위유역이 '합계'로 지정되어 있으므로 OR(|) 조건으로 포함
mask_wide = (
    (city_exhaustion_status['시군'] != "강원도(한강)") & 
    ((city_exhaustion_status['단위유역'] == "소계") | (city_exhaustion_status['시군'] == "강원도"))
)
city_exhaustion_status = city_exhaustion_status[mask_wide].copy()
city_exhaustion_status = city_exhaustion_status.drop(columns=['단위유역'])

# 2. 시군별 소진현황 요약표 생성 (Long/Row Format)
# R: select(대상물질:소진량, 소진율.전체:잔여율) %>% filter...
# R 코드의 select 범위에 포함되는 컬럼: 대상물질, 시군, 단위유역, 점비점, 지역개발, 잔여량, 소진량
cols_summary_row = [
    '대상물질', '시군', '단위유역', '점비점', 
    '지역개발', '잔여량', '소진량', 
    '소진율.전체', '잔여율'
]
# 실제 존재하는 컬럼만 선택
cols_summary_row = [c for c in cols_summary_row if c in internal_result_row_base.columns]

city_exhaustion_status_row = internal_result_row_base[cols_summary_row].copy()

# 강원도 포함 조건 적용
mask_row = (
    (city_exhaustion_status_row['시군'] != "강원도(한강)") & 
    ((city_exhaustion_status_row['단위유역'] == "소계") | (city_exhaustion_status_row['시군'] == "강원도"))
)
city_exhaustion_status_row = city_exhaustion_status_row[mask_row].copy()
city_exhaustion_status_row = city_exhaustion_status_row.drop(columns=['단위유역'])

# 3. 엑셀 파일 내보내기
output_path_internal = OUTPUT_DIR / "누적관리대장_정리(내부검토)_py.xlsx"

sheets_to_write_internal = {
    "시군별소진현황": city_exhaustion_status,
    "시군별소진현황_점비점행기준": city_exhaustion_status_row,
    "협의현황": consultation_status,
    "지역개발부하량": internal_result,
    "지역개발부하량_점비점행기준": internal_result_row_base,
    "협의사업목록": consultation_list,
    "협의현황_영향평가": consultation_status_eia,
    "준공건수": final_completion_counts
}

try:
    with pd.ExcelWriter(output_path_internal, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_to_write_internal.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"파일 저장 완료: {output_path_internal}")

except PermissionError:
    print(f"[오류] 파일이 열려 있어 저장할 수 없습니다: {output_path_internal}")
    print("엑셀 파일을 닫고 다시 실행해주세요.")
except Exception as e:
    print(f"[오류] 파일 저장 중 문제가 발생했습니다: {e}")

# %% ---------------------------------------------------------------------------
# 25. [내부검토] 미준공 사업 현황 및 내역 정리

# 1. 미준공 사업 내역 정리 (data_clean 활용)
# R: select(사업구분:단위유역, 사업명, 준공예정연도, 준공여부, 협의상태, 협의연도, 협의일자, 주관부서:삭감방법_비점)
cols_incomplete_details = [
    '사업구분', '시군', '단위유역', '사업명', '준공예정연도', '준공여부', 
    '협의상태', '협의연도', '협의일자', '주관부서', '담당자명', '부서연락처', 
    '사업종류', '삭감방법_점', '삭감방법_비점'
]

# 존재하는 컬럼만 안전하게 선택
cols_incomplete_details = [c for c in cols_incomplete_details if c in data_clean.columns]
incomplete_details = data_clean[cols_incomplete_details].copy()

# 준공예정연도를 숫자형으로 변환 (비교 연산을 위해)
incomplete_details['준공예정연도_num'] = pd.to_numeric(incomplete_details['준공예정연도'], errors='coerce')

# 필터링: 협의상태 == "협의" & 준공여부 == "미준공" & 준공예정연도 <= 기준년도
mask_details = (
    (incomplete_details['협의상태'] == "협의") & 
    (incomplete_details['준공여부'] == "미준공") & 
    (incomplete_details['준공예정연도_num'] <= REF_YEAR)
)
incomplete_details = incomplete_details[mask_details].copy()

# 비교용 임시 컬럼 삭제
incomplete_details = incomplete_details.drop(columns=['준공예정연도_num'])

# 정렬 (sort_dataframe 후 지정된 순서로 정렬)
incomplete_details = sort_dataframe(incomplete_details, mode='internal')
incomplete_details = incomplete_details.sort_values(by=['시군', '단위유역', '준공예정연도'])


# 2. 미준공 사업 현황 정리 (consultation_list 활용)
incomplete_status = consultation_list.copy()

# 준공예정연도를 숫자형으로 변환
incomplete_status['준공예정연도_num'] = pd.to_numeric(incomplete_status['준공예정연도'], errors='coerce')

# 필터링
mask_status = (
    (incomplete_status['협의상태'] == "협의") & 
    (incomplete_status['준공여부'] == "미준공") & 
    (incomplete_status['준공예정연도_num'] <= REF_YEAR)
)
incomplete_status = incomplete_status[mask_status].copy()

# 집계 (시군, 단위유역별 개수 산출)
incomplete_summary = incomplete_status.groupby(['시군', '단위유역']).size().reset_index(name='개수')

# 시군별 소계 계산
city_subtotals_inc = incomplete_summary.groupby('시군')['개수'].sum().reset_index()
city_subtotals_inc['단위유역'] = "소계"

# 병합
incomplete_summary = pd.concat([incomplete_summary, city_subtotals_inc], ignore_index=True)

# 정렬
incomplete_summary = sort_dataframe(incomplete_summary, mode='internal')
incomplete_summary = incomplete_summary.sort_values(by=['시군', '단위유역'])

print("미준공 건수 정리 완료. 크기:", incomplete_summary.shape)
display(incomplete_summary.head())

print("미준공 사업 내역 정리 완료. 크기:", incomplete_details.shape)
display(incomplete_details.head())

# %% ---------------------------------------------------------------------------
# 26. [내부검토] 미준공 사업 현황 엑셀 파일 내보내기
output_path_incomplete = OUTPUT_DIR / "미준공_사업_현황_py.xlsx"

sheets_to_write_inc = {
    "미준공 사업 현황": incomplete_summary,
    "미준공 사업 내역": incomplete_details
}

try:
    with pd.ExcelWriter(output_path_incomplete, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_to_write_inc.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"파일 저장 완료: {output_path_incomplete}")

except PermissionError:
    print(f"[오류] 파일이 열려 있어 저장할 수 없습니다: {output_path_incomplete}")
except Exception as e:
    print(f"[오류] 파일 저장 중 문제가 발생했습니다: {e}")

# %% [markdown]
# ******************************************************************************
# ### 그래프 생성 및 시각화

# %% ---------------------------------------------------------------------------
# 27. [그래프] 그래프용 데이터 정리

# 1. 대상 데이터 필터링
# R: filter(!(시군 %in% c("강원도", "강원도(한강)")), 단위유역 == "소계")
mask_plot = (
    (~internal_result_row_base['시군'].isin(["강원도", "강원도(한강)"])) & 
    (internal_result_row_base['단위유역'] == "소계")
)
plot_df = internal_result_row_base[mask_plot].copy()

# 2. 필요한 컬럼 선택
# R: select(대상물질, 시군, 점비점:소진량, 소진율.전체)
cols_to_keep = ['대상물질', '시군', '점비점', '지역개발', '잔여량', '기승인', '소진량', '소진율.전체']
cols_to_keep = [c for c in cols_to_keep if c in plot_df.columns]
plot_df = plot_df[cols_to_keep].copy()

# 3. 파생 변수 생성 및 시군 명칭 변경
# R: mutate(소진율 = ..., 잔여율 = ..., 잔여량_1 = ...)
plot_df['소진율'] = plot_df['소진율.전체'] * 100
plot_df['잔여율'] = 100 - plot_df['소진율']
plot_df['잔여량_1'] = plot_df['잔여량']

# 시군 문자열 전처리 (정규식을 이용하여 "시" 또는 "군" 제거, 한강/낙동강 축약)
plot_df['시군'] = (
    plot_df['시군'].astype(str)
    .str.replace(r'(시|군)', '', regex=True)
    .str.replace('낙동강', '낙', regex=False)
    .str.replace('한강', '한', regex=False)
)

# 4. 데이터 형태 변환 (Wide -> Long)
# R: pivot_longer(cols = 잔여량:소진량, names_to = "구분", values_to = "부하량")
id_vars = [c for c in plot_df.columns if c not in ['잔여량', '소진량']]

plot_df_long = plot_df.melt(
    id_vars=id_vars,
    value_vars=['잔여량', '소진량'],
    var_name='구분',
    value_name='부하량'
)

# 5. 비율 컬럼 생성 및 불필요한 컬럼 삭제
# R: mutate(비율 = case_when(구분 == "잔여량" ~ 잔여율, 구분 == "소진량" ~ 소진율))
plot_df_long['비율'] = np.where(
    plot_df_long['구분'] == '잔여량', 
    plot_df_long['잔여율'], 
    plot_df_long['소진율']
)

# 불필요한 열 삭제 (R: select(-c(소진율.전체:잔여율)))
cols_to_drop = ['소진율.전체', '소진율', '잔여율']
plot_df_long = plot_df_long.drop(columns=[c for c in cols_to_drop if c in plot_df_long.columns])

# 이름 변경 (R: rename(잔여량 = 잔여량_1))
plot_df_long = plot_df_long.rename(columns={'잔여량_1': '잔여량'})

# 6. 라벨 위치 계산
# R: mutate(라벨위치 = ifelse(구분 == "잔여량", 부하량 - 부하량 * 0.5, 잔여량 + 부하량 * 0.5))
plot_df_long['라벨위치'] = np.where(
    plot_df_long['구분'] == "잔여량",
    plot_df_long['부하량'] - (plot_df_long['부하량'] * 0.5),
    plot_df_long['잔여량'] + (plot_df_long['부하량'] * 0.5)
)

print("그래프용 데이터 정리 완료. 크기:", plot_df_long.shape)
display(plot_df_long.head())

# %% ---------------------------------------------------------------------------
# 28 [그래프] 통합 그래프 생성 및 저장 함수

def create_exhaustion_chart(data, substance, source_type, output_filename):
    """
    소진 및 잔여량 현황 막대그래프 생성 및 저장 함수
    params:
        data: plot_df_long 데이터프레임
        substance: 대상물질 ('BOD' or 'TP')
        source_type: 오염원 구분 ('점' or '비점')
        output_filename: 저장할 파일명 (확장자 포함)
    """
    # 1. 데이터 필터링
    df_filtered = data[
        (data['대상물질'] == substance) & 
        (data['점비점'] == source_type)
    ].copy()
    
    if df_filtered.empty:
        print(f"[경고] 데이터가 없습니다: {substance} - {source_type}")
        return

    # 2. 그래프 초기화
    fig, ax = plt.subplots(figsize=(12, 6))

    # 잔여량과 소진량 데이터 분리
    df_exhaust = df_filtered[df_filtered['구분'] == '소진량'].copy()
    df_remain = df_filtered[df_filtered['구분'] == '잔여량'].copy()

    # X축 데이터
    x_labels = df_exhaust['시군'].tolist()
    x = np.arange(len(x_labels))

    # 3. 누적 막대 그리기
    # 소진량 (아래)
    ax.bar(
        x, df_exhaust['부하량'], 
        color='#F8766D', label='소진량', edgecolor='none'
    )
    # 잔여량 (위)
    ax.bar(
        x, df_remain['부하량'], 
        bottom=df_exhaust['부하량'].values, 
        color='#00BFC4', label='잔여량', edgecolor='none'
    )

    # 4. 텍스트 라벨 추가
    total_loads = df_remain['지역개발'].values
    remain_ratios = df_remain['비율'].values

    # Y축 최대값 여유 공간 확보 (15%)
    if len(total_loads) > 0:
        ax.set_ylim(0, max(total_loads) * 1.15)

    for i in range(len(x)):
        tot = total_loads[i]
        ratio = remain_ratios[i]
        
        # TP이면서 비점일 때는 소수점 3자리, 그 외(TP 점 포함)는 정수형 반올림 등
        # 요청사항에 따라 TP는 소수점 3자리, BOD는 정수형으로 분기 처리
        if substance == 'TP':
            str_tot = f"{round(tot, 3)}"
        else:
            str_tot = f"{round(tot)}"

        # 총 지역개발부하량 (막대 위)
        ax.annotate(
            str_tot,
            xy=(x[i], tot),
            xytext=(0, 14), 
            textcoords="offset points",
            ha='center', va='bottom',
            color='blue', fontsize=14, weight='bold'
        )
        
        # 잔여율 (막대 바로 위)
        ax.annotate(
            f"({round(ratio)}%)",
            xy=(x[i], tot),
            xytext=(0, 2), 
            textcoords="offset points",
            ha='center', va='bottom',
            color='blue', fontsize=13
        )

    # 5. 축, 제목 및 레이블 설정
    # 축 테두리
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    # 라벨 폰트
    ax.set_ylabel('부하량 (kg/일)', fontsize=16, weight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    # 타이틀 (필요 시 주석 해제)
    # ax.set_title(f'시군별 {substance} {source_type}오염원 소진 및 잔여량 현황', fontsize=16, pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=14)

    # 범례
    ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize=13)

    # 그리드
    ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # 6. 저장 및 출력
    plt.tight_layout()
    save_path = PLOT_DIR / output_filename
    plt.savefig(save_path, dpi=300)
    print(f"그래프 저장 완료: {save_path}")
    plt.show()

# ---------------------------------------------------------------------------
# 함수 실행 (4가지 케이스 일괄 처리)

# Case 1: BOD 점
create_exhaustion_chart(plot_df_long, 'BOD', '점', "지역개발부하량_소진현황_그래프_BOD_점.png")

# Case 2: BOD 비점
create_exhaustion_chart(plot_df_long, 'BOD', '비점', "지역개발부하량_소진현황_그래프_BOD_비점.png")

# Case 3: TP 점
create_exhaustion_chart(plot_df_long, 'TP', '점', "지역개발부하량_소진현황_그래프_TP_점.png")

# Case 4: TP 비점
create_exhaustion_chart(plot_df_long, 'TP', '비점', "지역개발부하량_소진현황_그래프_TP_비점.png")

# %%
