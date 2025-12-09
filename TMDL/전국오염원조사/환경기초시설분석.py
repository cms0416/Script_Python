# %% ---------------------------------------------------------------------------
# 환경기초시설 "방류량" 시트 병합(2018년까지/2019년 이후 분기 + 상단 3행 제외)
# - 2018년까지: A~D, G~Q 사용 / G+H+I → "유량" (삽입 후 G, H, I 제거)
# - 2019년 이후: A~D, G~O 사용 / G = "유량"

# %% ---------------------------------------------------------------------------
# 라이브러리 불러오기
import numpy as np
import pandas as pd
from pathlib import Path
import re

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화를 위한 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'NanumGothic'  
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

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

# %% ---------------------------------------------------------------------------
# 기본 설정
# 데이터 폴더 경로 설정
DATA_DIR = Path(r"E:/Coding/TMDL/전국오염원조사/환경기초시설")

# 필요한 열 설정
USECOLS_PRE2019  = "A:D,G:Q"   # 2018년까지
USECOLS_2019PLUS = "A:D,G:O"   # 2019년 이후

# 열 이름 설정
FINAL_COLUMNS = ["시설명", "시설코드", "일자", "방류구번호",
                 "유량", "온도", "pH", "BOD", "COD", "SS", "TN", "TP", "TOC"]

# 분석 대상 유역 및 기간 설정
단위유역 = "섬강B"
시작연도 = 2015
종료연도 = 2024

# %% ---------------------------------------------------------------------------
# 파일명에서 연도 추출(예: "2018년기준_..." → 2018)
def year_from_filename(fp: Path) -> int:
    m = re.match(r"(\d{4})년", fp.name)
    return int(m.group(1))

# %% ---------------------------------------------------------------------------
# 파일별 시트 읽기 함수
def read_file(fp: Path) -> pd.DataFrame:
    year = year_from_filename(fp)
    usecols = USECOLS_PRE2019 if year <= 2018 else USECOLS_2019PLUS

    # 시트 읽기(컬럼명 없음, 상단 3행 건너뜀)
    df = pd.read_excel(
        fp, sheet_name="방류량", header=None, skiprows=3,
        usecols=usecols, engine="openpyxl"
    )

    # --- 2018년까지: G/H/I 합산 → "유량"
    if year <= 2018:
        # 현재 df는 A:D, G:Q 범위만 포함됨 → A-D(4열) 다음의 첫 3개 수치열이 유량 분해열
        flow_parts = df.iloc[:, 4:7].apply(pd.to_numeric, errors="coerce")
        # NaN 안전 합산(모두 NaN이면 NaN, 하나라도 값 있으면 합)
        flow = flow_parts.sum(axis=1, min_count=1)
        # "유량"을 A-D 다음(인덱스 4)에 삽입
        df.insert(4, "유량", flow)
        # 합산에 사용한 기존 3개 열(G,H,I에 해당)은 삽입 후 위치 5,6,7 → 삭제
        df.drop(df.columns[[5,6,7]], axis=1, inplace=True)

    # 공통: 열 이름 부여
    df.columns = FINAL_COLUMNS

    # 공통: 일자 파싱 및 숫자 변환(A~C 제외)
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    num_cols = df.columns[3:]  # D~ 끝까지
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 필요 시: 완전 공백행 제거 및 중복 제거
    df = df.dropna(how="all").drop_duplicates()

    # 분석 편의: 연도/월 추가
    df["연도"] = df["일자"].dt.year
    df["월"] = df["일자"].dt.month

    return df

# %% ---------------------------------------------------------------------------
# 전체 파일 불러오기 및 병합

# 파일 목록 불러오기
files = sorted([f for f in DATA_DIR.glob("*.xlsx") if not f.name.startswith("~$")])

# 전체 파일 방류량 자료 병합
방류량_원본 = pd.concat([read_file(fp) for fp in files], ignore_index=True)

print(f"[완료] 총 {방류량_원본.shape[0]}행 × {방류량_원본.shape[1]}열")

방류량_원본

# %% ---------------------------------------------------------------------------
# 원본 데이터 복사
방류량 = 방류량_원본.copy()

# 부하량 산정
방류량["BOD부하량"] = (방류량["유량"] * 방류량["BOD"] * 86.4).round(2)  # kg/day
방류량["TP부하량"] = (방류량["유량"] * 방류량["TP"] * 86.4).round(2)    # kg/day

# 계절 열 추가
방류량 = add_season(방류량)
방류량

# %% ---------------------------------------------------------------------------
# 환경기초시설 현황 파일 불러오기
시설현황 = pd.read_excel(
    DATA_DIR / "환경기초시설 현황/환경기초시설_현황.xlsx", engine="openpyxl"
)

# 환경기초시설 현황에서 시설명, 시설코드, 시군, 단위유역 추출
시설현황 = 시설현황[["시설명", "시설코드", "시군", "단위유역"]].drop_duplicates()
시설현황

# %% ---------------------------------------------------------------------------
# 방류량 자료에 시군, 유역 현황 병합
방류량 = pd.merge(방류량, 시설현황, on=["시설명", "시설코드"], how="left")

# 정렬(시군, 시설명, 일자)
방류량_원본 = 방류량_원본.sort_values(["시설명", "일자"], ignore_index=True)

방류량

# %% ---------------------------------------------------------------------------
# "방류량"에서 원주공공하수처리시설 필터 후 연도별 BOD 부하량 박스플롯 그리기
plt.figure(figsize=(10, 6))
sns.boxplot(data=방류량[방류량["시설명"] == "원주공공하수처리시설"], x="연도", y="BOD부하량")
plt.title("원주공공하수처리시설 연도별 BOD 부하량 분포")
plt.show()

# %% ---------------------------------------------------------------------------
# 목표수질 불러오기
목표수질 = pd.read_excel("E:/Coding/TMDL/수질분석/목표수질.xlsx")
목표수질_BOD = 목표수질.loc[목표수질['총량지점명'] == 단위유역, 'BOD_목표수질'].values[0]

목표수질_BOD

# %% ---------------------------------------------------------------------------
# 총량측정망 불러오기
총량측정망_원본 = (
    pd.read_excel("E:/Coding/TMDL/수질분석/총량측정망_2007_2024.xlsx")
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

# 8. '유량크기순서'열을 첫번째 열로 이동
# '유량크기순서' 열을 첫번째로 배치하고, 
# 나머지 모든 열의 이름들을 리스트로 만들어 그 뒤에 배치
총량측정망 = 총량측정망[['유량크기순서'] + [col for col in 총량측정망.columns if col != '유량크기순서']]

총량측정망

# %% -----------------------------------------------------------
# 필요 라이브러리
from scipy.stats import pearsonr, rankdata
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# 파라미터
MAX_LAG = 14         # 시차 탐색 범위
ALPHA_FDR = 0.10     # 다중비교 FDR 기준(10%)

# %% --------------------------------------
# 1) 하천(섬강B) 일자별 평균 BOD, 평균 유량
riv = (
    총량측정망
    .query("총량지점명.str.contains(@단위유역)", engine="python")
    .assign(일자=lambda d: pd.to_datetime(d["일자"]).dt.floor("D"))
    .groupby("일자", as_index=True)
    .agg(하천_BOD_평균=("BOD", "mean"),
         하천_유량_평균=("유량", "mean"))
    .sort_index()
)

# %% --------------------------------------
# 2) 시설별 일자합 BOD부하량 (섬강B만)
fac_daily = (
    방류량
    .query("단위유역 == @단위유역")
    .assign(일자=lambda d: pd.to_datetime(d["일자"]).dt.floor("D"))
    .groupby(["시설코드","시설명","일자"], as_index=False)["BOD부하량"].sum()
)

# 편의상 시설 목록
fac_list = fac_daily[["시설코드","시설명"]].drop_duplicates().to_numpy().tolist()

# --------------------------------------
# 부분상관 유틸(순위-잔차 방식: Spearman partial)
def partial_spearman(x, y, z):
    """
    x: 시설부하(래그 적용), y: 하천 BOD, z: 하천 유량(통제)
    1) x,y,z 순위화 → 2) x~z, y~z 선형회귀 → 3) 잔차 간 피어슨 r, p
    """
    # 배열화
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())

    xr = rankdata(x[mask])
    yr = rankdata(y[mask])
    zr = rankdata(z[mask])

    X = sm.add_constant(zr)
    rx = sm.OLS(xr, X).fit().resid
    ry = sm.OLS(yr, X).fit().resid

    r, p = pearsonr(rx, ry)
    return float(r), float(p), int(mask.sum())

# %% --------------------------------------
# 3) 시설별 최적 시차 탐색 + 부분상관 계산
rows = []
for code, name in fac_list:
    s = (
        fac_daily
        .query("시설코드 == @code")
        .set_index("일자")[["BOD부하량"]]
        .sort_index()
    )

    # 시차별 평가
    best = None
    for lag in range(0, MAX_LAG+1):
        s_lag = s.copy()
        s_lag["시설부하_lag"] = s_lag["BOD부하량"].shift(lag)

        # 날짜 병합(내부조인)
        merged = riv.join(s_lag[["시설부하_lag"]], how="inner")

        r, p, n = partial_spearman(
            merged["시설부하_lag"], merged["하천_BOD_평균"], merged["하천_유량_평균"]
        )

        rec = {
            "시설코드": code, "시설명": name, "lag": lag, "n": n,
            "rho_partial_spearman": r, "p_partial_spearman": p,
            "fac_median_load": float(merged["시설부하_lag"].median(skipna=True)) if n>0 else np.nan
        }
        # 최적(절댓값 최대) 갱신
        if best is None or (np.nan_to_num(abs(r), nan=-1) > np.nan_to_num(abs(best["rho_partial_spearman"]), nan=-1)):
            best = rec

    if best is not None:
        rows.append(best)

result = pd.DataFrame(rows)

# %% --------------------------------------
# 4) 다중비교 보정(FDR, BH)
mask = result["p_partial_spearman"].notna()
rej, qvals, _, _ = multipletests(result.loc[mask, "p_partial_spearman"], alpha=ALPHA_FDR, method="fdr_bh")
result.loc[mask, "q_partial_spearman"] = qvals
result.loc[mask, "유의(FDR≤{:.0%})".format(ALPHA_FDR)] = rej

# 정렬: 효과크기 우선
result_sorted = (
    result
    .assign(abs_r=lambda d: d["rho_partial_spearman"].abs())
    .sort_values(["유의(FDR≤{:.0%})".format(ALPHA_FDR), "abs_r"], ascending=[False, False])
    .drop(columns=["abs_r"])
    .reset_index(drop=True)
)

print("시설별 영향도(부분상관, 유량 통제, 최적 시차):")
print(result_sorted.head(20))


# %%
