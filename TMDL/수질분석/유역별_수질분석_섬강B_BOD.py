# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: moonsoo1
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 유역별 수질 및 기상자료 분석
#

# %% [markdown]
# ### 1. 라이브러리 로드 및 계절 정의 함수

# %% ---------------------------------------------------------------------------
# 라이브러리 로드
import pandas as pd
import numpy as np
from datetime import datetime

# 파일 내보내기위한 라이브러리
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# %% ---------------------------------------------------------------------------
# 시각화 라이브러리 로드
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter , ScalarFormatter
import seaborn as sns

# 시각화를 위한 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'NanumGothic'  
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지


# %% ---------------------------------------------------------------------------
# 분석 대상 유역 및 기상대
단위유역 = "섬강B"
기상대 = "횡성"

# 분석 기간 설정
시작연도 = 2015
종료연도 = 2024


# %% ---------------------------------------------------------------------------
# 계절 정의 함수 (월 기준)
def add_season(df, month_column='월'):
    """월 정보를 바탕으로 계절 컬럼 추가"""
    df = df.copy()
    conditions = [
        df[month_column].between(3, 5),
        df[month_column].between(6, 8),
        df[month_column].between(9, 11)
    ]
    seasons = ['봄', '여름', '가을']
    df['계절'] = np.select(conditions, seasons, default='겨울')
    df['계절'] = pd.Categorical(df['계절'], categories=['봄', '여름', '가을', '겨울'], ordered=True)
    return df



# %% [markdown]
# ### 2. 기상자료 정리

# %% ---------------------------------------------------------------------------
# 기상자료(일단위) 정리
기상자료 = (
    pd.read_excel("E:/Coding/TMDL/수질분석/기상자료/기상자료_강수량_기온.xlsx")
    .query("지점명.str.contains(@기상대)", engine='python')  # regex 검색
    .drop(columns=["지점", "지점명"])
    .rename(columns={
        "일시": "일자",
        "평균기온(°C)": "평균기온",
        "일강수량(mm)": "일강수량"
    })
)

기상자료['일자'] = pd.to_datetime(기상자료['일자'])

# 연도, 월 컬럼 추가
기상자료['연도'] = 기상자료['일자'].dt.year
기상자료['월'] = 기상자료['일자'].dt.month

기상자료

# %% ---------------------------------------------------------------------------
# 기상자료(시단위) 정리
기상자료_시단위 = (
    pd.read_excel("E:/Coding/TMDL/수질분석/기상자료/기상자료_강수량_기온_시단위.xlsx")
    .query("지점명.str.contains(@기상대)", engine='python')  # regex 검색
    .drop(columns=["지점", "지점명"])
    .rename(columns={
        "기온(°C)": "기온",
        "강수량(mm)": "강수량"
    })
)

기상자료_시단위['일시'] = pd.to_datetime(기상자료_시단위['일시'])
기상자료_시단위['일자'] = 기상자료_시단위['일시'].dt.date
기상자료_시단위['연도'] = 기상자료_시단위['일시'].dt.year
기상자료_시단위['월'] = 기상자료_시단위['일시'].dt.month

기상자료_시단위['고강도강우'] = 기상자료_시단위['강수량'] >= 20
기상자료_시단위['매우고강도강우'] = 기상자료_시단위['강수량'] >= 30

기상자료_시단위

# %% [markdown]
# ### 3. 강수량 분석

# %% ---------------------------------------------------------------------------
## 연월별 강수량 합계

# 연월별 강수량 합계
강수량_연월별_합계 = (
    기상자료.groupby(['연도', '월'])['일강수량']
    .sum()
    .reset_index(name='월강수량')
)

# 연도별 '소계' 행 생성
강수량_연월별_합계_소계 = (
    강수량_연월별_합계
    .groupby('연도')
    .apply(lambda df: pd.concat([
        df,
        pd.DataFrame([{
            '연도': df['연도'].iloc[0],
            '월': '소계',
            '월강수량': df['월강수량'].sum()
        }])
    ], ignore_index=True))
    .reset_index(drop=True)
)

강수량_연월별_합계_소계

# %% ---------------------------------------------------------------------------
## 월별 강수량 평균
# '소계' 제외 후 월별 평균
강수량_월별_평균 = (
    강수량_연월별_합계
    .groupby('월')['월강수량']
    .mean()
    .round(1)
    .reset_index()
    .sort_values('월')
)

강수량_월별_평균

# %% ---------------------------------------------------------------------------
## 일별 최대 시간강우량 및 강우강도 여부
# 일별 요약
# 고강도 강우(강한비) 확인 : 시간당 강수량 20mm이상(한강홍수통제소 기준)
기상지표_일별 = (
    기상자료_시단위
    .dropna(subset=['강수량'])  # 전부 NA인 날 제거
    .groupby('일자')
    .agg({
        '강수량': ['sum', 'max', lambda x: (x >= 20).any(), lambda x: (x >= 30).any()],
        '기온': 'mean'
    })
)

# 컬럼 이름 변경
기상지표_일별.columns = [
    '일강수량', '최대시간강우량', '고강도강우_발생여부', 
    '매우고강도강우_발생여부', '평균기온'
    ]
# 인덱스 리셋 및 강우일 여부 추가
기상지표_일별 = 기상지표_일별.reset_index()
# 강우일 여부 추가
기상지표_일별['강우일'] = 기상지표_일별['일강수량'] > 0
# 평균기온 소수점 첫째자리로 반올림
기상지표_일별['평균기온'] = 기상지표_일별['평균기온'].round(1)

# 고강도강우 여부(기존) → int 변환 후 rolling 합계
기상지표_일별['고강도강우_발생여부_3일누적'] = (
    기상지표_일별['고강도강우_발생여부']
    .astype(int)                          # True/False → 1/0
    .rolling(window=3, min_periods=1)     # 최근 3일(당일 포함)
    .sum()                                # 합계 (3일 중 몇 번 발생했는지)
    .gt(0)                                # 0보다 크면 True
)

# 누적 강수량 계산
기상지표_일별['누적강수_3일'] = 기상지표_일별['일강수량'].rolling(window=3, min_periods=1).sum()
기상지표_일별['누적강수_4일'] = 기상지표_일별['일강수량'].rolling(window=4, min_periods=1).sum()
기상지표_일별['누적강수_5일'] = 기상지표_일별['일강수량'].rolling(window=5, min_periods=1).sum()


# 연도, 월 컬럼 추가
기상지표_일별['일자'] = pd.to_datetime(기상지표_일별['일자'])
기상지표_일별['연도'] = 기상지표_일별['일자'].dt.year
기상지표_일별['월'] = 기상지표_일별['일자'].dt.month

기상지표_일별

# %% ---------------------------------------------------------------------------
## 연도별 기상 지표 요약
기상지표_연도별 = (
    기상지표_일별.groupby('연도')
    .agg(
        강수량_합계=pd.NamedAgg(column='일강수량', aggfunc='sum'),
        강수량_일평균=pd.NamedAgg(column='일강수량', aggfunc=lambda x: round(x.mean(), 1)),
        강우일수=pd.NamedAgg(column='강우일', aggfunc='sum'),
        고강도일수=pd.NamedAgg(column='고강도강우_발생여부', aggfunc='sum'),
        평균기온=pd.NamedAgg(column='평균기온', aggfunc=lambda x: round(x.mean(), 1))
    )
).reset_index()

# 고강도비율 추가
기상지표_연도별['고강도비율'] = (
    (기상지표_연도별['고강도일수'] / 기상지표_연도별['강우일수']) * 100
).round(1)

# 평균기온 열을 맨뒤로 이동(꺼내고 맨 뒤에 다시 추가)
평균기온 = 기상지표_연도별.pop('평균기온')
기상지표_연도별['평균기온'] = 평균기온

기상지표_연도별

# %% [markdown]
# ### 4. 수질데이터(총량측정망) 정리

# %% ---------------------------------------------------------------------------
# 목표수질 불러오기
목표수질 = pd.read_excel("E:/Coding/TMDL/수질분석/목표수질.xlsx")
목표수질_BOD = 목표수질.loc[목표수질['총량지점명'] == 단위유역, 'BOD_목표수질'].values[0]

목표수질_BOD

# %% ---------------------------------------------------------------------------
# 총량측정망 불러오기
총량측정망_원본 = (
    pd.read_excel("E:/Coding/TMDL/수질분석/총량측정망_전체_2007_2025.xlsx")
    .dropna(subset=['BOD'])  # BOD 결측 제거
    # 단위유역과 연도 필터링
    .query("총량지점명.str.contains(@단위유역) and 연도 >= @시작연도 and 연도 <= @종료연도")
)

# 계절 추가
총량측정망_원본 = add_season(총량측정망_원본)

# 목표수질 초과 여부
총량측정망_원본['달성여부'] = np.where(총량측정망_원본['BOD'] > 목표수질_BOD, "초과", "달성")

총량측정망_원본

# %% ---------------------------------------------------------------------------
## 총량측정망 자료 정리

# 1. 원본 데이터 복사
총량측정망 = 총량측정망_원본.copy()

# 2. BOD 부하량 계산
총량측정망['측정부하량_BOD'] = 총량측정망['유량'] * 총량측정망['BOD'] * 86.4
총량측정망['목표부하량_BOD'] = 총량측정망['유량'] * 목표수질_BOD * 86.4

# 3. 유량 기준 내림차순 정렬
총량측정망 = 총량측정망.sort_values(by='유량', ascending=False).reset_index(drop=True)

# 4. 유량 크기 순서(동일값은 동일한 순위 부여)
총량측정망['유량크기순서'] = 총량측정망['유량'].rank(method='min', ascending=False)

# 5. 유량 유효 개수 계산(유량 값이 있는 행의 개수)
유량_유효개수 = 총량측정망['유량'].notna().sum()

# 6. 백분율 계산(NA는 자동으로 NaN 유지됨)
총량측정망['유량백분율'] = 총량측정망['유량크기순서'] / 유량_유효개수 * 100

# 7. 유황구간 정의
bins = [0, 10, 40, 60, 90, 100]
labels = ['홍수기', '풍수기', '평수기', '저수기', '갈수기']
총량측정망['유황구간'] = pd.cut(
    총량측정망['유량백분율'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=True  # 경계 포함 여부, 필요시 False로 조정
)

# 8. 기상자료 병합(일자, 연도, 월 기준)
총량측정망 = pd.merge(총량측정망, 기상지표_일별, on=['일자', '연도', '월'], how='left')


# 9. '유량크기순서'열을 첫번째 열로 이동
# '유량크기순서' 열을 첫번째로 배치하고, 
# 나머지 모든 열의 이름들을 리스트로 만들어 그 뒤에 배치
총량측정망 = 총량측정망[['유량크기순서'] + [col for col in 총량측정망.columns if col != '유량크기순서']]

총량측정망

# %% ---------------------------------------------------------------------------
## 일별 기상자료와 총량측정망 자료 합치기
# 1. 총량측정망 원본 복사
수질측정망_총량 = 총량측정망_원본.copy()

# 2. 일별 기상지표 자료를 기준으로 left join
총량측정망_기상 = pd.merge(
    기상지표_일별,
    수질측정망_총량,
    on=['일자', '연도', '월'],
    how='left'
)

# 3. 계절 추가
총량측정망_기상 = add_season(총량측정망_기상)

총량측정망_기상

# %% [markdown]
# ### 5. 달성률

# %% ---------------------------------------------------------------------------
## 유황구간별 달성률
# 유황구간 순서 지정
유황구간_순서 = ['갈수기', '저수기', '평수기', '풍수기', '홍수기']

# 유황구간을 순서 있는 범주형(factor)으로 변환
총량측정망['유황구간'] = pd.Categorical(
    총량측정망['유황구간'],
    categories=유황구간_순서,
    ordered=True
)

# 유황구간별 달성률 계산
달성률_유황구간별 = (
    총량측정망.groupby(['유황구간', '달성여부'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

달성률_유황구간별['총계'] = 달성률_유황구간별['달성'] + 달성률_유황구간별['초과']
달성률_유황구간별['달성률'] = (달성률_유황구간별['달성'] / 달성률_유황구간별['총계'] * 100).round(1)

# 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
총계_열 = 달성률_유황구간별.pop('총계')
달성률_유황구간별.insert(달성률_유황구간별.columns.get_loc('달성'), '총계', 총계_열)

# 유황구간 순서 적용 정렬
달성률_유황구간별 = 달성률_유황구간별.sort_values('유황구간').reset_index(drop=True)

달성률_유황구간별

# %% ---------------------------------------------------------------------------
## 계절별 달성률
달성률_계절별 = (
    총량측정망.groupby(['계절', '달성여부'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
달성률_계절별['총계'] = 달성률_계절별['달성'] + 달성률_계절별['초과']
달성률_계절별['달성률'] = (달성률_계절별['달성'] / 달성률_계절별['총계'] * 100).round(1)

# 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
총계_열 = 달성률_계절별.pop('총계')
달성률_계절별.insert(달성률_계절별.columns.get_loc('달성'), '총계', 총계_열)

달성률_계절별

# %% ---------------------------------------------------------------------------
## 월별 달성률
달성률_월별 = (
    총량측정망.groupby(['월', '달성여부'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
달성률_월별['총계'] = 달성률_월별['달성'] + 달성률_월별['초과']
달성률_월별['달성률'] = (달성률_월별['달성'] / 달성률_월별['총계'] * 100).round(1)

# 월별 강수량 평균 병합
달성률_월별 = pd.merge(달성률_월별, 강수량_월별_평균, on='월', how='left')

# 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
총계_열 = 달성률_월별.pop('총계')
달성률_월별.insert(달성률_월별.columns.get_loc('달성'), '총계', 총계_열)

달성률_월별

# %% ---------------------------------------------------------------------------
## 연도 별 달성률 및 강우강도 지표
달성률_연도별 = (
    총량측정망.groupby('연도')
    .agg({
        'BOD': lambda x: round(x.mean(), 1),
        '유량': 'sum',
        '달성여부': lambda x: (x == '달성').sum(),
    })
    .rename(columns={'BOD': 'BOD_평균', '유량': '유량_합계', '달성여부': '달성'})
    .reset_index()
)

달성률_연도별['초과'] = 총량측정망.groupby('연도')['달성여부'].apply(lambda x: (x == '초과').sum()).values
달성률_연도별['총계'] = 달성률_연도별['달성'] + 달성률_연도별['초과']
달성률_연도별['달성률'] = (달성률_연도별['달성'] / 달성률_연도별['총계'] * 100).round(1)

# 열 순서 재정렬
달성률_연도별 = 달성률_연도별[['연도', '초과', '달성', '총계', '달성률', 'BOD_평균', '유량_합계']]

# 기상지표와 병합
달성률_강우강도_연도별 = pd.merge(달성률_연도별, 기상지표_연도별, on='연도', how='left')

달성률_강우강도_연도별

# %% ---------------------------------------------------------------------------
# 분석 결과 엑셀로 내보내기
output_path = Path(f"E:/Coding/TMDL/수질분석/Output/수질분석_{단위유역}_파이썬.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    총량측정망.to_excel(writer, sheet_name="수질현황", index=False)
    달성률_유황구간별.to_excel(writer, sheet_name="달성률_유황구간별", index=False)
    달성률_계절별.to_excel(writer, sheet_name="달성률_계절별", index=False)
    달성률_월별.to_excel(writer, sheet_name="달성률_월별", index=False)
    달성률_강우강도_연도별.to_excel(writer, sheet_name="달성률_강우강도_연도별", index=False)

# %% [markdown]
# ### 6. 유역내 측정지점 별 정리(수질측정망)

# %% ---------------------------------------------------------------------------
## 수질측정망 자료 불러오기
수질측정망 = (
    pd.read_excel("E:/Coding/TMDL/수질분석/수질측정망/섬강A_수질측정망_2014_2024.xlsx")
    .dropna(subset=['BOD'])  # BOD 결측 제거
)

# BOD 2 이상인 행 제거(섬강2 지점 이상수치 2.108 mg/L 제외)
#수질측정망 = 수질측정망.query("BOD < 2")

# 연도, 월 컬럼 추가
수질측정망['일자'] = pd.to_datetime(수질측정망['일자'])
수질측정망['연도'] = 수질측정망['일자'].dt.year
수질측정망['월'] = 수질측정망['일자'].dt.month

# 계절 추가
수질측정망 = add_season(수질측정망)

# 목표수질 초과 여부
수질측정망['달성여부'] = np.where(수질측정망['BOD'] > 목표수질_BOD, "초과", "달성")


수질측정망

# %% ---------------------------------------------------------------------------
# 유역내 측정지점 별 평균(최근 5년)
수질측정망_평균 = (
    수질측정망
    .query(f"연도 > {종료연도} - 5")             # ① 연도 필터
    [['측정소명', 'BOD']]                         # ② 필요한 열만 선택
    .groupby('측정소명', as_index=False)         # ③ 측정소별 그룹화
    .agg(BOD_평균=('BOD', lambda x: np.round(x.mean(skipna=True), 1)))  # ④ 평균 + 반올림
)

수질측정망_평균

# %% ---------------------------------------------------------------------------
# 유역내 측정지점 별 달성률(최근 5년)
수질측정망_달성률 = (
    수질측정망
    .query(f"연도 > {종료연도} - 5")             # 연도 필터
    .groupby(['측정소명', '달성여부'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# 총계와 달성률 계산
수질측정망_달성률['총계'] = 수질측정망_달성률['달성'] + 수질측정망_달성률['초과']
수질측정망_달성률['달성률'] = (수질측정망_달성률['달성'] / 수질측정망_달성률['총계'] * 100).round(1)

수질측정망_달성률

# %% ---------------------------------------------------------------------------
# 유역내 측정지점 별 평균수질, 달성률 정리
# 수질측정망_평균과 수질측정망_달성률의 달성률 열 병합
수질측정망_결과 = pd.merge(
    수질측정망_평균, 
    수질측정망_달성률[['측정소명', '달성률']],
    on='측정소명',
    how='left'
)

수질측정망_결과

# %% [markdown]
# ### 7. 통계 분석

# %% ---------------------------------------------------------------------------
# 통계 분석용 라이브러리 로드
from scipy.stats import pearsonr

# %% ---------------------------------------------------------------------------
## 수질항목 간의 상관관계 분석

# 분석 대상 컬럼
corr_cols = [
    'TP', 'SS', '유량', '누적강수_3일', '누적강수_4일', '누적강수_5일',
    '평균기온', '일강수량', 'COD', '수온', 'TOC', 'TN', 'BOD', 'pH', 'EC', 'DO'
]

# NaN 제거
df_corr = 총량측정망[corr_cols].dropna()

# 상관계수 & p-value 계산
corr_matrix = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)
pval_matrix = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)

for col1 in corr_cols:
    for col2 in corr_cols:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1.0
            pval_matrix.loc[col1, col2] = 0.0
        else:
            corr, pval = pearsonr(df_corr[col1], df_corr[col2])
            corr_matrix.loc[col1, col2] = corr
            pval_matrix.loc[col1, col2] = pval

# 결과 확인 (옵션)
display(corr_matrix.round(2))
display(pval_matrix.round(4))

# %%
# ✅ BOD와의 상관계수 가져오기
bod_corr = corr_matrix['BOD'].drop('BOD')

# ✅ X축: 높은 상관계수 → 낮은 순서 (BOD는 맨 왼쪽)
x_order = ['BOD'] + bod_corr.sort_values(ascending=False).index.tolist()

# ✅ Y축: 낮은 상관계수 → 높은 순서 (BOD는 맨 아래)
y_order = bod_corr.sort_values(ascending=True).index.tolist() + ['BOD']

# ✅ 순서 적용된 히트맵 생성
corr_sorted = corr_matrix.loc[y_order, x_order]

display(corr_sorted.round(2))

# %% ---------------------------------------------------------------------------
# 분석 결과 엑셀로 내보내기
with pd.ExcelWriter(f"E:/Coding/TMDL/수질분석/Output/{단위유역}_BOD_상관계수.xlsx") as writer:
    corr_matrix.to_excel(writer, sheet_name="상관계수")
    pval_matrix.to_excel(writer, sheet_name="p값")

# %% [markdown]
# ### 8. 데이터 시각화

# %% ---------------------------------------------------------------------------
## 상관계수 히트맵

# ✅ seaborn 스타일 설정
sns.set_theme(style="whitegrid", font='NanumGothic', font_scale=1.0)

# ✅ 유의확률(p-value) 기준으로 annot 텍스트 마스킹
annot_matrix = corr_sorted.copy()

# p-value 행렬에서 현재 corr_sorted 순서에 맞게 정렬
pval_sorted = pval_matrix.loc[annot_matrix.index, annot_matrix.columns]

# p-value > 0.05인 경우 annot을 빈 문자열로 대체
annot_matrix = annot_matrix.round(2).astype(str)
annot_matrix[pval_sorted > 0.05] = ""

# 📊 히트맵 시각화
plt.figure(figsize=(11, 10))
ax = sns.heatmap(
    corr_sorted,
    annot=annot_matrix, fmt="",  # 유의하지 않은 값은 빈칸
    cmap="coolwarm", center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"weight": "bold", "fontsize": 12}
)

# ✅ 컬러바 객체 가져오기 (가로 방향일 경우 마지막 axes)
cbar = ax.collections[0].colorbar

# ✅ 컬러바 제목 설정
cbar.ax.set_xlabel("상관계수", fontsize=14, weight='bold', labelpad=10)
cbar.ax.xaxis.set_label_position('top')  # 상단에 위치

# ✅ 축라벨 설정
plt.xticks(fontsize=13, weight='bold', rotation=45, ha='right')  # x축 라벨 45도 기울이기
plt.yticks(fontsize=13, weight='bold')  # y축 라벨은 수평 유지

# ✅ 그래프 여백 조정
plt.tight_layout(pad=1.5) # 상하좌우 여백 자동 조정  
plt.subplots_adjust(right=1.05)  # 우측 여백 조정

# 저장
plt.savefig("E:/Coding/TMDL/수질분석/Output/Plot/섬강A_상관계수.png", dpi=300)
plt.show()

# %% ---------------------------------------------------------------------------
# 🎨 계절별 색상 지정
season_palette = {
    '봄': '#F8766D',
    '여름': '#00BA38',
    '가을': '#619CFF',
    '겨울': '#C77CFF'
}

# %% ---------------------------------------------------------------------------
## 유량 / 수질(BOD) 그래프

# 데이터 준비
데이터 = 총량측정망.copy()

# 계절별 색상 팔레트 적용
season_colors = 데이터['계절'].map(season_palette)

# 그래프 생성
fig, ax1 = plt.subplots(figsize=(8.4, 5))
ax2 = ax1.twinx()  # 보조 y축 생성

# 🔵 유량 막대
bars = ax2.bar(
    데이터['유량백분율'], 데이터['유량'],
    width=0.3, color='steelblue', edgecolor='steelblue', alpha=1,
    label='유량'
)
ax2.set_ylim(0, 200)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))  # y축 레이블 포맷팅(x를 정수로 변환하고, 천 단위 쉼표 추가)
ax2.grid(False)  # 보조 y축 격자선 제거

# ⚫ 수질 점
scatter = ax1.scatter(
    데이터['유량백분율'], 데이터['BOD'],
    c=season_colors, edgecolor='black', s=40,
    alpha=0.6, label='수질'
)
ax1.set_yscale("log")
#ax1.set_ylim(0.006, 1)
# y축 로그 스케일 유지하되 눈금은 일반 숫자로
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.ticklabel_format(axis='y', style='plain')
ax1.set_xticks(range(0, 101, 10))  # x축 눈금 설정
ax1.set_xticklabels([f"{x}%" for x in range(0, 101, 10)])  # x축 레이블 설정
ax1.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

# 🔴 목표수질 선
ax1.axhline(y=목표수질_BOD, color='red', linestyle='dashed', linewidth=1.2, label='목표수질')

# 🔲 수직선 (유황구간 기준선)
for x in [0, 10, 40, 60, 90, 100]:
    ax1.axvline(x=x, linestyle='dashed', linewidth=1, color='gray')

# 📝 유황구간 라벨
유황라벨 = {
    5: '홍수기', 25: '풍수기', 50: '평수기',
    75: '저수기', 95: '갈수기'
}
for x, label in 유황라벨.items():
    ax1.text(x, 0.65, label, ha='center', va='center', fontsize=12, weight='bold')

# 📌 축 레이블 제목 설정
ax1.set_xlabel("유량 백분율(%)", fontsize=14, weight='bold')
ax1.set_ylabel("BOD (mg/L)", fontsize=14, weight='bold')
ax2.set_ylabel("유량(㎥/s)", fontsize=14, weight='bold')

# 축 눈금 설정
ax1.tick_params(axis='both', labelsize=12)  # x축 + 왼쪽 y축
ax2.tick_params(axis='y', labelsize=12)     # 오른쪽 y축
# x축 눈금 바깥쪽으로
ax1.xaxis.set_ticks_position('bottom')

# ax2의 zorder를 얻고, ax2의 zorder보다 큰 값을 ax1의 zorder로 지정
# zorder가 낮을수록 먼저 그려지고, zorder가 높을수록 나중에 그려짐
ax1.set_zorder(ax2.get_zorder() + 10)
# ax1의 배경을 투명하게 만들어 ax1이 앞으로 배치되었을 때 ax2의 내용이 가려지지 않게 함
ax1.patch.set_visible(False)

# 축 테두리선 색상 변경
for spine in ax1.spines.values():
    spine.set_color('dimgray')
for spine in ax2.spines.values():
    spine.set_color('dimgray')

# 📌 범례 구성(수질/목표수질/유량 + 계절)
legend_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None',
               color='black', markerfacecolor='white', label='수질', markersize=8),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
    plt.Line2D([0], [0], linestyle='-', color='steelblue', label='유량')
] + [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor=color, alpha=0.6, label=label, markersize=8)
    for label, color in season_palette.items()
]

# 1️⃣ 수질 / 목표수질 / 유량 범례
legend1 = ax1.legend(
    handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질, 유량
    loc='upper center',
    ncol=3,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.25, 1.13),
    fontsize=12,
    handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 2️⃣ 계절 범례
legend2 = ax1.legend(
    handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
    loc='upper center',
    ncol=4,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.75, 1.13),
    fontsize=12,
    title_fontsize=12,
    handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 첫 번째 범례를 그래프에 추가로 다시 등록
ax1.add_artist(legend1)

# 여백 조정
plt.subplots_adjust(top=0.82)
plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------------
## 강수 / 수질(BOD) 그래프

# 데이터 준비
데이터 = 총량측정망_기상.copy()
데이터['일자'] = pd.to_datetime(데이터['일자'])
데이터 = 데이터[데이터['연도'] >= 2020]

# 계절별 색상 팔레트 적용
season_colors = 데이터['계절'].map(season_palette)

# 그래프 생성
fig, ax1 = plt.subplots(figsize=(8.4, 5))
ax2 = ax1.twinx()  # 보조 y축 생성

# 🔵 강수량 막대 그래프
ax2.bar(
    데이터['일자'], 데이터['일강수량'],
    width=3, color='steelblue', edgecolor='steelblue', alpha=1.0, label='강수량'
)
#ax2.set_ylim(0, 225)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))  # y축 레이블 포맷팅(x를 정수로 변환하고, 천 단위 쉼표 추가)
ax2.grid(False)  # 보조 y축 격자선 제거

# ⚫ 수질 점
scatter = ax1.scatter(
    데이터['일자'], 데이터['BOD'],
    c=season_colors, edgecolor='black', s=40,
    label='수질(BOD)', alpha=0.7
)
ax1.set_yscale("log")
#ax1.set_ylim(0.006, 1)
# y축 로그 스케일 유지하되 눈금은 일반 숫자로
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.ticklabel_format(axis='y', style='plain')
ax1.xaxis.set_major_locator(mdates.YearLocator())  # 연도 단위로 x축 눈금 설정
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 연도 형식으로 표시
ax1.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

# 🔴 목표수질 선
ax1.axhline(y=목표수질_BOD, color='red', linestyle='dashed', linewidth=1.2, label='목표수질')

# 📌 축 레이블 제목 설정
ax1.set_xlabel("연도", fontsize=14, weight='bold')
ax1.set_ylabel("BOD (mg/L)", fontsize=14, weight='bold')
ax2.set_ylabel("강수량(mm)", fontsize=14, weight='bold')

# 축 눈금 설정
ax1.tick_params(axis='both', labelsize=12)  # x축 + 왼쪽 y축
ax2.tick_params(axis='y', labelsize=12)     # 오른쪽 y축
# x축 눈금 바깥쪽으로
ax1.xaxis.set_ticks_position('bottom')

# ax2의 zorder를 얻고, ax2의 zorder보다 큰 값을 ax1의 zorder로 지정
# zorder가 낮을수록 먼저 그려지고, zorder가 높을수록 나중에 그려짐
ax1.set_zorder(ax2.get_zorder() + 10)
# ax1의 배경을 투명하게 만들어 ax1이 앞으로 배치되었을 때 ax2의 내용이 가려지지 않게 함
ax1.patch.set_visible(False)

# 축 테두리선 색상 변경
for spine in ax1.spines.values():
    spine.set_color('dimgray')
for spine in ax2.spines.values():
    spine.set_color('dimgray')

# 📌 범례 구성(수질/목표수질/강수량 + 계절)
legend_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', 
               color='black', markerfacecolor='white', 
               label='수질', markersize=8, markeredgewidth=1.5),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
    plt.Line2D([0], [0], linestyle='-', color='steelblue', label='강수량')
] + [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor=color, alpha=0.6, label=label, markersize=8)
    for label, color in season_palette.items()
]

# 1️⃣ 수질 / 목표수질 / 강수량 범례
legend1 = ax1.legend(
    handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질, 강수량
    loc='upper center',
    ncol=3,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.25, 1.13),
    fontsize=12,
    handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 2️⃣ 계절 범례 (제목 포함)
legend2 = ax1.legend(
    handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
    loc='upper center',
    ncol=4,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.75, 1.13),
    fontsize=12,
    title_fontsize=12,
    handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 첫 번째 범례를 그래프에 추가로 다시 등록
ax1.add_artist(legend1)

# 상단 여백 확보
plt.subplots_adjust(top=0.82)
plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------------
## 연도 별 박스플롯

# 데이터 준비
데이터 = 총량측정망.copy()
데이터 = 데이터[데이터['BOD'] > 0]  # 로그 스케일이므로 0 이상만 사용

# 계절별 색상 팔레트 적용
season_colors = 데이터['계절'].map(season_palette)

# 연도 순서 고정 (박스/스트립/막대 모두 동일 사용)
years_order = sorted(데이터['연도'].unique())

# 그래프 생성
fig, ax = plt.subplots(figsize=(8.4, 5))

# 📦 박스플롯(이상치 표시 없이)
sns.boxplot(
    data=데이터, x='연도', y='BOD',
    ax=ax, showfliers=False, width=0.5,
    color='white', linewidth=1,
    linecolor='black'
)

# ⚫ Jittered scatter
sns.stripplot(
    data=데이터, x='연도', y='BOD',
    ax=ax, 
    jitter=0.2,  # 점들이 겹치지 않도록 약간의 무작위 이동
    size=6,
    edgecolor='black',
    linewidth=1,
    palette=season_palette,
    hue='계절',
    dodge=False,  # 계절별로 점을 분리하지 않음
    alpha=0.6, marker='o'
)

# 🔴 목표수질 수평선
ax.axhline(y=목표수질_BOD, color='red', linestyle='--', linewidth=1.2, label='목표수질')

# 🧪 y축 로그 스케일
ax.set_yscale("log")
#ax.set_ylim(0.005, 1)
# y축 로그 스케일 유지하되 눈금은 일반 숫자로
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='plain')
ax.set_ylabel("BOD (mg/L)", fontsize=14, weight='bold')
ax.set_xlabel("연도", fontsize=14, weight='bold')
ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

# 축 눈금 설정
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# 축 테두리선 색상 변경
for spine in ax.spines.values():
    spine.set_color('dimgray')

# 🔵 보조축: 연도별 유량 합계 막대 ------
# 연도별 유량합계 매핑 (달성률_강우강도_연도별['연도', '유량_합계'] 가정)
flow_map = 달성률_강우강도_연도별.set_index('연도')['유량_합계']
flow_vals = [flow_map.get(y, np.nan) for y in years_order]
xpos = np.arange(len(years_order))

ax2 = ax.twinx()
bars = ax2.bar(
    xpos, flow_vals, width=0.6,
    color='steelblue', edgecolor='none', alpha=0.30, zorder=0,  # 뒤에 깔림
)
ax2.set_ylabel("연 유량 합계", fontsize=12)   # 단위가 있으면 예: (10^6 m³)
ax2.grid(False)

# 박스/점이 막대 위로 보이도록
ax.set_zorder(2)
ax.patch.set_alpha(0.0)

# x축 눈금을 연도 라벨로 명시(혹시 matplotlib 버전에 따라 필요할 수 있음)
ax.set_xticks(xpos)
ax.set_xticklabels(years_order, rotation=0)


# 📌 범례 구성(수질/목표수질 + 계절)
# '유량 합계' 패치 추가
flow_patch = mpatches.Patch(color='steelblue', alpha=0.6, label='유량')

legend_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor='white', label='수질', markersize=8),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
    flow_patch
] + [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor=color, alpha=0.6, label=label, markersize=8)
    for label, color in season_palette.items()
]

# 1️⃣ 수질 / 목표수질 범례
legend1 = ax.legend(
    handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질
    loc='upper center',
    ncol=3,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.25, 1.13),  # 범례 위치 조정
    fontsize=12,
    handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 2️⃣ 계절 범례
legend2 = ax.legend(
    handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
    loc='upper center',  # 범례 위치
    ncol=4,  # 범례 항목 수
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.73, 1.13),  # 범례 위치 조정
    fontsize=12,
    title_fontsize=12,
    handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 첫 번째 범례를 그래프에 추가로 다시 등록
ax.add_artist(legend1)

# 🎨 기타 스타일
ax.tick_params(labelsize=11)  # 축 눈금 폰트 크기
ax.set_title("")  # 제목 없음

plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()

# %% ---------------------------------------------------------------------------
## 누적강수 3일 / 수질(BOD) 그래프

# 데이터 준비
데이터 = 총량측정망.copy()

# 그래프 생성
fig, ax = plt.subplots(figsize=(8.4, 5))

# ⚫ 수질 점
scatter = ax.scatter(
    데이터['누적강수_3일'], 데이터['BOD'],
    c='white', edgecolor='black', s=40,
    alpha=0.6, label='수질'
)
ax.set_yscale("log")
#ax.set_ylim(0.006, 1)
# y축 로그 스케일 유지하되 눈금은 일반 숫자로
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='plain')
ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

# 🔴 목표수질 선
ax.axhline(y=목표수질_BOD, color='red', linestyle='dashed', linewidth=1.2, label='목표수질')

# 📌 축 레이블 제목 설정
ax.set_xlabel("누적강수 3일(mm)", fontsize=14, weight='bold')
ax.set_ylabel("BOD (mg/L)", fontsize=14, weight='bold')

# 축 눈금 설정
ax.tick_params(axis='both', labelsize=12)  # x축 + 왼쪽 y축
# x축 눈금 바깥쪽으로
ax.xaxis.set_ticks_position('bottom')

# 축 테두리선 색상 변경
for spine in ax.spines.values():
    spine.set_color('dimgray')

# 📌 범례 구성(수질/목표수질)
legend_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor='white', label='수질', markersize=8),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
]

# 1️⃣ 수질 / 목표수질 범례
legend = ax.legend(
    handles=legend_handles,
    loc='upper center',  # 범례 위치
    ncol=2,  # 범례 항목 수
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.2, 1),  # 범례 위치 조정
    fontsize=12,
    handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 여백 조정
plt.subplots_adjust(top=0.82)
plt.tight_layout()
plt.show()

# %% ---------------------------------------------------------------------------
## 강우 이벤트별 수질 영향 분석

# 강우영향 샘플 구분(직전 3일 누적강수량이 20mm 이상이면 '강우영향')
#총량측정망['강우영향'] = 총량측정망['누적강수_3일'] >= 20
#총량측정망['강우영향_구분'] = 총량측정망['강우영향'].map({True: '강우영향', False: '평상시'})

# 라이브러리 로드
from scipy.stats import mannwhitneyu

# 스타일 설정
sns.set_theme(style="whitegrid", font="NanumGothic", font_scale=1.1)  # Windows 사용자 기준

# ===== 박스플롯: 최근 3일 고강도 강우 발생 여부 vs BOD (목표수질 + 범례 포함) =====
# 로그 스케일 대응: BOD>0, 결측 제거
plot_df = (
    총량측정망
    .dropna(subset=['BOD', '고강도강우_발생여부_3일누적'])
    .query('BOD > 0')
    .copy()
)

# x축 레이블을 보기 쉽게 한글 라벨로 변환
plot_df['최근3일_고강도발생'] = np.where(plot_df['고강도강우_발생여부_3일누적'], '발생', '미발생')

fig, ax = plt.subplots(figsize=(8.4, 5))

# 박스플롯
# - 색상은 pastel 팔레트 2색 사용(미발생/발생 순)
order = ['미발생', '발생']
palette = sns.color_palette('pastel', n_colors=2)

sns.boxplot(
    data=plot_df,
    x='최근3일_고강도발생', y='BOD',
    order=order,
    palette=palette,
    showfliers=True,
    width=0.6,
    ax=ax
)

# 🔴 목표수질 수평선
ax.axhline(y=목표수질_BOD, color='red', linestyle='--', linewidth=1.2, label='목표수질')

# 축/눈금/격자 스타일
ax.set_yscale("log")
#ax.set_ylim(0.005, 1)  # 필요 시 조정
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='plain')
ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)

ax.set_ylabel("BOD (mg/L)", fontsize=13, weight='bold')
ax.set_xlabel("최근 3일 고강도 강우 발생 여부", fontsize=13, weight='bold')

# 축 테두리선 색상
for spine in ax.spines.values():
    spine.set_color('dimgray')

# 축 눈금 위치
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# 🎯 범례 구성: '미발생', '발생' + '목표수질'
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=palette[0], edgecolor='black', label='미발생'),
    Patch(facecolor=palette[1], edgecolor='black', label='발생'),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
]

legend = ax.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=3,
    frameon=True,
    edgecolor='black',
    bbox_to_anchor=(0.5, 1.12),
    fontsize=11,
    handletextpad=0.5,
    columnspacing=1.2
)

plt.subplots_adjust(top=0.86)
plt.tight_layout()
plt.show()

# %% ---------------------------------------------------------------------------
# 📊 통계검정: Mann-Whitney U 검정 (비모수)
bod_rain = 총량측정망[총량측정망['고강도강우_발생여부_3일누적'] == True]['BOD'].dropna()
bod_dry = 총량측정망[총량측정망['고강도강우_발생여부_3일누적'] == False]['BOD'].dropna()

stat, p_value = mannwhitneyu(bod_rain, bod_dry, alternative='two-sided')

print(f"📌 Mann-Whitney U 검정 결과:\n - 통계량: {stat:.2f}\n - p-value: {p_value:.3e}")
if p_value < 0.05:
    print("➡ 두 그룹 간 BOD 농도 차이는 통계적으로 유의합니다.")
else:
    print("➡ 두 그룹 간 BOD 농도 차이는 유의하지 않습니다.")

# 두 그룹간 평균 비교
mean_rain = bod_rain.mean()
mean_dry = bod_dry.mean()
print(f"📌 평균 BOD 농도:\n - 고강도 강우 발생: {mean_rain:.4f} mg/L\n - 고강도 강우 미발생: {mean_dry:.4f} mg/L")   


# %% ---------------------------------------------------------------------------
## 유역 내 수질측정망 박스플롯

# 데이터 준비
데이터 = 수질측정망.copy()

# x축 순서 지정
order = ["계천2", "금계천2", "전천", "섬강1", "섬강2", "섬강3"]

# 계절별 색상 팔레트 적용
season_colors = 데이터['계절'].map(season_palette)

# 그래프 생성
fig, ax = plt.subplots(figsize=(8.5, 6))

# 📦 박스플롯 (이상치 표시 없이)
sns.boxplot(
    data=데이터, x='측정소명', y='BOD', order=order,
    ax=ax, showfliers=False, width=0.5,
    color='white', linewidth=1,
    linecolor='black'
)

# ⚫ Jittered scatter
sns.stripplot(
    data=데이터, x='측정소명', y='BOD', order=order,
    ax=ax, 
    jitter=0.2,  # 점들이 겹치지 않도록 약간의 무작위 이동
    size=6,
    edgecolor='black',
    linewidth=1,
    palette=season_palette,
    hue='계절',
    dodge=False,  # 계절별로 점을 분리하지 않음
    alpha=0.5, marker='o'
)

# 🔴 목표수질 수평선
ax.axhline(y=목표수질_BOD, color='red', linestyle='--', 
           linewidth=1.2, label='목표수질')

# 🧪 y축 로그 스케일
ax.set_yscale("log")
ax.set_ylim(0.0009, 1)
# y축 로그 스케일 유지하되 눈금은 일반 숫자로
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='plain')
ax.set_ylabel("BOD (mg/L)", fontsize=14, weight='bold')
ax.set_xlabel("연도", fontsize=14, weight='bold')
ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

# 축 눈금 설정
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# 축 테두리선 색상 변경
for spine in ax.spines.values():
    spine.set_color('dimgray')

# 📌 범례 구성(수질/목표수질 + 계절)
legend_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor='white', label='수질', markersize=8),
    plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
] + [
    plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
               markerfacecolor=color, alpha=0.6, label=label, markersize=8)
    for label, color in season_palette.items()
]

# 1️⃣ 수질 / 목표수질 범례
legend1 = ax.legend(
    handles=legend_handles[:2],  # 두번째 까지(0~1): 수질, 목표수질
    loc='upper center',
    ncol=2,
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.25, 1.13),  # 범례 위치 조정
    fontsize=12,
    handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 2️⃣ 계절 범례
legend2 = ax.legend(
    handles=legend_handles[2:],  # 세번째 부터: 계절(봄~겨울)
    loc='upper center',  # 범례 위치
    ncol=4,  # 범례 항목 수
    frameon=True,  # 범례 테두리 표시
    edgecolor='black',  # 범례 테두리 색상
    bbox_to_anchor=(0.65, 1.13),  # 범례 위치 조정
    fontsize=12,
    title_fontsize=12,
    handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
    columnspacing=1   # 범례 항목(열) 간 간격
)

# 첫 번째 범례를 그래프에 추가로 다시 등록
ax.add_artist(legend1)

# 🎨 기타 스타일
ax.tick_params(labelsize=11)  # x축 + y축 눈금 크기
ax.set_title("")  # 제목 없음

plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()


# %%
