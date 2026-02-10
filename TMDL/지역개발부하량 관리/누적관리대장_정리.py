# %% [markdown]
# # 지역개발부하량 관리 및 분석

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
BASE_DIR = Path("E:/Coding/TMDL/지역개발부하량 관리")
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
    """그룹별 소계 행 추가 (janitor::adorn_totals 대체)"""
    subtotals = df.groupby(group_cols)[value_cols].sum().reset_index()
    if fill_col:
        subtotals[fill_col] = label
    return pd.concat([df, subtotals], ignore_index=True)

def sort_dataframe(df, city_col='시군', basin_col='단위유역', mode='internal'):
    """데이터프레임 정렬 (Categorical 활용)"""
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
    return final_df

print("함수 정의 완료.")

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

# 데이터 확인
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

# 시군 명칭 및 태백시 처리
df_selected['시군'] = df_selected['시군'].astype(str).str.replace("강원특별자치도 ", "").str.replace("강원도 ", "")
mask_taebaek_nak = (df_selected['시군'] == '태백시') & (df_selected['단위유역'] == '낙본A')
df_selected.loc[mask_taebaek_nak, '시군'] = '태백시(낙동강)'
mask_taebaek_han = (df_selected['시군'] == '태백시') & (df_selected['단위유역'] != '낙본A')
df_selected.loc[mask_taebaek_han, '시군'] = '태백시(한강)'

# 유효한 데이터만 필터링
data_clean = df_selected[~df_selected['협의상태'].isin(["할당", "보완"])].copy()

print(f"데이터 정제 완료. 최종 데이터 건수: {len(data_clean)}")

data_clean

# %% ---------------------------------------------------------------------------
# 추진실적용 데이터 (북한D, 임진A 제외)
data_perf = data_clean[~data_clean['단위유역'].isin(["북한D", "임진A"])].copy()

data_perf

# %% ---------------------------------------------------------------------------
# 8. [분석] 추진실적 소진현황 산정

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
perf_status = perf_status.merge(used_sum, on=['시군', '대상물질', '단위유역'], how='left').fillna(0)

# 계산
perf_status['협의가능량_점'] = perf_status['지역개발_점'] - perf_status['협의부하량_점']
perf_status['협의가능량_비점'] = perf_status['지역개발_비점'] - perf_status['협의부하량_비점']
perf_status['잔여부하량_점'] = perf_status['협의가능량_점'] - perf_status['사용부하량_점']
perf_status['잔여부하량_비점'] = perf_status['협의가능량_비점'] - perf_status['사용부하량_비점']

# 정렬 적용
perf_status = sort_dataframe(perf_status, mode='performance')

print("추진실적 소진현황 계산 완료.")
display(perf_status.head())

# %% ---------------------------------------------------------------------------
# 9. [분석] 내부검토용 자료 산정

# 전체 소진량
internal_exhaust = load_cal(data_clean)
internal_exhaust = internal_exhaust.rename(columns={'소진_점': '소진량_점', '소진_비점': '소진량_비점'})

# 기승인 소진량
internal_approved = load_cal(data_clean[data_clean['사업구분'] == '기본계획_기승인'])
internal_approved = internal_approved.rename(columns={'소진_점': '기승인_점', '소진_비점': '기승인_비점'})

# 병합
internal_result = base.copy()
internal_result = internal_result[internal_result['시군'] != "강원도"]
internal_result = internal_result.merge(internal_exhaust, on=['대상물질', '시군', '단위유역'], how='left')
internal_result = internal_result.merge(internal_approved, on=['대상물질', '시군', '단위유역'], how='left')
internal_result = internal_result.fillna(0)

# 신규 및 잔여량 계산
internal_result['신규_점'] = internal_result['소진량_점'] - internal_result['기승인_점']
internal_result['신규_비점'] = internal_result['소진량_비점'] - internal_result['기승인_비점']
internal_result['잔여량_점'] = internal_result['지역개발_점'] - internal_result['소진량_점']
internal_result['잔여량_비점'] = internal_result['지역개발_비점'] - internal_result['소진량_비점']

# 정렬
internal_result = sort_dataframe(internal_result, mode='internal')

print("내부검토용 자료 계산 완료.")
display(internal_result.head())

# %% ---------------------------------------------------------------------------
# 10. 엑셀 파일로 내보내기

output_file_name = OUTPUT_DIR / f"누적관리대장_정리_{datetime.now().strftime('%Y%m%d')}.xlsx"

try:
    with pd.ExcelWriter(output_file_name, engine='xlsxwriter') as writer:
        perf_status.to_excel(writer, sheet_name="5-가. 개발사업 소진현황", index=False)
        internal_result.to_excel(writer, sheet_name="지역개발부하량(내부)", index=False)
        # 필요한 경우 data_clean 등 raw 데이터도 시트로 추가 가능
        
    print(f"파일 저장 완료: {output_file_name}")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")

# %% ---------------------------------------------------------------------------
# 11. 그래프 시각화 (Seaborn)
# - 내부검토용 데이터 중 강원도 전체 및 한강수계 제외하고, 단위유역 '소계'인 데이터만 추출

plot_df = internal_result[
    (~internal_result['시군'].isin(["강원도", "강원도(한강)"])) & 
    (internal_result['단위유역'] == "소계")
].copy()

# 시각화를 위해 시군 명칭 줄이기
plot_df['시군_short'] = plot_df['시군'].str.replace("시", "").str.replace("군", "")
plot_df['시군_short'] = plot_df['시군_short'].str.replace("태백(낙동강)", "태백(낙)").str.replace("태백(한강)", "태백(한)")

# 예시: BOD 점오염원 소진/잔여 현황
target_material = "BOD"
subset = plot_df[plot_df['대상물질'] == target_material].set_index('시군_short')
subset_plot = subset[['소진량_점', '잔여량_점']]

plt.figure(figsize=(12, 6))
# Stacked Bar Chart 생성
subset_plot.plot(kind='bar', stacked=True, color=['orange', 'lightgray'], ax=plt.gca())

plt.title(f"{target_material} 점오염원 지역개발부하량 소진현황", fontsize=15, fontweight='bold')
plt.xlabel("시군")
plt.ylabel("부하량 (kg/일)")
plt.xticks(rotation=0)
plt.legend(["소진량", "잔여량"])
plt.tight_layout()

# 그래프 저장
save_img_path = PLOT_DIR / f"{target_material}_점_소진현황.png"
plt.savefig(save_img_path, dpi=300)
print(f"그래프 저장됨: {save_img_path}")
plt.show()

# %%