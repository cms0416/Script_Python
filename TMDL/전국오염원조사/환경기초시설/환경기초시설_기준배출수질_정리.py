# %% [markdown]
# # 환경기초시설 기준배출수질 산정
#
# 정규성 검증(Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov) 및 기준배출수질 산정

# %% ---------------------------------------------------------------------------
# 1. 라이브러리 로드 및 환경 설정
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
import warnings

# 경고 메시지 무시 (연산 중 발생하는 런타임 경고 등)
warnings.filterwarnings('ignore')

# 경로 설정 (R 코드 원본과 동일하게 구성)
BASE_DIR = Path("C:/Coding/TMDL")
WORKING_DIR = BASE_DIR / "전국오염원조사"

# 출력 디렉토리 확인 및 생성
OUTPUT_DIR = WORKING_DIR / "기준배출수질/Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"작업 경로: {WORKING_DIR.absolute()}")

# %% ---------------------------------------------------------------------------
# 2. 정렬 상수 정의

# 카테고리(Factor) 정렬을 위한 순서 리스트
ORDER_CITY = [
    "춘천시", "원주시", "강릉시", "태백시", "삼척시", "홍천군",
    "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군",
    "양구군", "인제군", "고성군", "동해시", "속초시", "양양군"
]

ORDER_BASIN = [
    "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A",
    "섬강A", "섬강B", "북한A", "북한B", "소양A", "인북A", "소양B", "북한C",
    "홍천A", "한탄A", "한강B", "제천A", "한강D", "북한D", "한탄B", "임진A",
    "낙본A"
]

# %% ---------------------------------------------------------------------------
# 3. 함수 정의 (데이터 로드 및 정리, 정규성 검증, 기준배출수질 산정, 평균유량 산정)

# 3-1. 데이터 로드 및 전처리 함수 (1-1. 데이터 불러오기)

class DataProcessor:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        
    def load_data(self):
        """환경기초시설 현황 및 방류량 자료 로드"""
        
        # 1. 환경기초시설 현황 자료
        status_path = self.working_dir / "환경기초시설/환경기초시설 현황/환경기초시설_현황.xlsx"
        try:
            self.df_status = pd.read_excel(status_path)
            print("현황 자료 로드 성공")
        except Exception as e:
            print(f"[오류] 현황 자료 로드 실패: {e}")
            self.df_status = pd.DataFrame()
            
        # 2. 방류량 자료 파일 목록 
        data_dir = self.working_dir / "기준배출수질/환경기초시설 데이터"
        xls_files = list(data_dir.glob("*.xls*"))
        print(f"대상 폴더에서 {len(xls_files)}개의 엑셀 파일을 발견했습니다.")
        
        df_list = []
        for file in xls_files:
            try:
                # 3줄(skip=3) 건너뛰고 헤더 없이(header=None) 읽기
                temp_df = pd.read_excel(file, sheet_name="방류량", skiprows=3, header=None)
                
                if not temp_df.empty:
                    # 열 인덱스를 0, 1, 2... 순으로 일치시켜 concat 시 컬럼 밀림 방지
                    temp_df.columns = range(temp_df.shape[1])
                    df_list.append(temp_df)
                    print(f" - 로드 성공: {file.name} (크기: {temp_df.shape})")
                else:
                    print(f" - [경고] 데이터가 없는 빈 시트입니다: {file.name}")
                    
            except Exception as e:
                print(f" - [오류] 로드 실패 ({file.name}): {e}")
                
        if df_list:
            self.df_raw = pd.concat(df_list, ignore_index=True)
            print(f"\n방류량 데이터 병합 완료. 총 행 수: {len(self.df_raw)}")
        else:
            self.df_raw = pd.DataFrame()
            print("[경고] 방류량 데이터가 병합되지 않았습니다.")

        return self.df_status, self.df_raw
    
    def clean_data(self, df_raw, df_status):
        """데이터 정리 (변수 선택, 이름 변경, 병합 및 필터링)"""
        if df_raw.empty or df_status.empty:
            return pd.DataFrame()
            
        # 필요한 열 선택 (0-index 기반: R의 1, 7, 10, 14번째 열은 Python에서 0, 6, 9, 13)
        target_indices = [0, 6, 9, 13]
        available_indices = [i for i in target_indices if i < len(df_raw.columns)]
        
        df_clean = df_raw.iloc[:, available_indices].copy()
        
        # 열 이름 변경
        col_names = ["시설명", "유량", "BOD", "TP"]
        if len(df_clean.columns) == len(col_names):
            df_clean.columns = col_names
        
        # 현황 자료와 조인 (시군, 단위유역 추가)
        df_status_sub = df_status[['시설명', '시군', '단위유역']]
        df_merged = pd.merge(df_clean, df_status_sub, on='시설명', how='left')
        
        # '기타' 수계 제외
        df_merged = df_merged[df_merged['단위유역'] != '기타'].copy()
        
        # 숫자형 변환 (문자열 등 오류는 NaN 처리)
        for col in ['유량', 'BOD', 'TP']:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            
        return df_merged

# ------------------------------------------------------------------------------
# 3-2. 정규성 검증 및 분석 함수 (1-2. 정규성 검증)

def check_normality(series):
    """
    개별 그룹(시설)의 데이터를 받아 정규성 검증 결과를 반환 
     - Shapiro-wilk, Anderson-Darling, Kolmogorov-Smirnov 3가지 방법 중
       1개 이상 정규성이면 정규성으로 판단
     - Kolmogorov-Smirnov는 모집단 평균(μ), 표준편차(σ)를 “사전 고정”된 모수로 가정하므로,
       표본에서 추정한 μ̂, σ̂를 사용할 경우 p-value가 과대·과소 평가될 수 있음
     - 따라서 모수추정 보정을 포함한 Lilliefors 테스트를 사용
    """
    n = len(series)
    if n < 3 or series.std(ddof=1) == 0:
        return pd.Series({'KS': np.nan, 'SW': np.nan, 'AD': np.nan, 'Result': None, 'n': n})
        
    p_ks = np.nan
    p_sw = np.nan
    p_ad = np.nan  
    ad_pass = False 
    
    if n >= 5:
        try:
            _, p_ks = lilliefors(series)
        except:
            pass
            
    if 3 <= n <= 5000:
        try:
            _, p_sw = stats.shapiro(series)
        except:
            pass
            
    if n >= 8:
        try:
            res_ad = stats.anderson(series, dist='norm')
            if res_ad.statistic < res_ad.critical_values[2]: 
                ad_pass = True
                p_ad = 1.0  # 정규성 통과 (5% 유의수준)
            else:
                p_ad = 0.0  # 정규성 기각
        except:
            pass
            
    result = "비정규성"
    if (pd.notna(p_ks) and p_ks > 0.05) or (pd.notna(p_sw) and p_sw > 0.05) or ad_pass:
        result = "정규성"
        
    if pd.isna(p_ks) and pd.isna(p_sw) and pd.isna(p_ad):
        result = np.nan
        
    return pd.Series({'KS': p_ks, 'SW': p_sw, 'AD': p_ad, 'Result': result, 'n': n})

def run_normality_tests(df, raw_col):
    """그룹별(시설명) 정규성 검증 일괄 수행"""
    result_col = f"{raw_col}_정규성"
    
    df_valid = df[df[raw_col].notna() & (df[raw_col] > 0)].copy()
    df_valid['log_val'] = np.log(df_valid[raw_col])
    
    grouped = df_valid.groupby('시설명')['log_val'].apply(check_normality).unstack().reset_index()
    grouped = grouped.rename(columns={'Result': result_col})
    
    # 계산 과정에 필요한 열 모두 포함하여 반환
    return grouped[['시설명', 'n', 'KS', 'SW', 'AD', result_col]].copy()

# ------------------------------------------------------------------------------
# 3-3. 기준배출수질(가, 나) 산정 함수 (2-1, 2-2. 기준배출수질)

def calc_parametric_limit(df, raw_col):
    """기준배출수질(가) 방식 (로그정규분포)"""
    result_col = f"{raw_col}_가"
    mean_col = f"{raw_col}_mean"
    sd_col = f"{raw_col}_sd"
    
    df_valid = df[df[raw_col].notna() & (df[raw_col] != 0)].copy()
    df_valid['log_val'] = np.log(df_valid[raw_col])
    
    grouped = df_valid.groupby('시설명')['log_val'].agg(['mean', 'std']).reset_index()
    grouped = grouped.rename(columns={'mean': mean_col, 'std': sd_col})
    
    grouped[result_col] = np.exp(grouped[mean_col] + 1.645 * grouped[sd_col])
    
    # 평균 및 표준편차 열 포함하여 반환
    return grouped[['시설명', mean_col, sd_col, result_col]].copy()

def calc_nonparametric_limit(df, raw_col):
    """기준배출수질(나) 방식 (비정규분포) 내분점 산정"""
    result_col = f"{raw_col}_나"
    max_col = f"{raw_col}_최대"
    count_col = f"{raw_col}_개수"
    
    df_valid = df[df[raw_col].notna() & (df[raw_col] != 0)].copy()
    df_valid = df_valid.sort_values(by=['시설명', raw_col])
    
    results = []
    
    for name, group in df_valid.groupby('시설명'):
        n = len(group)
        max_val = group[raw_col].max()
        
        if n == 0:
            continue
            
        vals = group[raw_col].values
        
        pos = 1 + 0.95 * (n - 1)
        a = np.floor(pos)   
        b = pos - a         
        
        idx_a = int(a) - 1  
        idx_a1 = int(a)     
        
        if idx_a1 >= n:
            Xa = vals[idx_a]
            Xa1 = np.nan
            result_val = vals[idx_a]
        else:
            Xa = vals[idx_a]
            Xa1 = vals[idx_a1]
            result_val = (1 - b) * Xa + b * Xa1
            
        results.append({
            '시설명': name,
            count_col: n,
            'a': a,
            'b': b,
            'Xa': Xa,
            'Xa1': Xa1,
            result_col: result_val,
            max_col: max_val
        })
    
    # 중간 계산 열(a, b, Xa, Xa1) 모두 포함하여 순서대로 반환
    return pd.DataFrame(results)[['시설명', count_col, 'a', 'b', 'Xa', 'Xa1', result_col, max_col]]

# ------------------------------------------------------------------------------
# 3-4. 평균유량 산정 함수 (2-3. 평균유량 산정)

def calc_mean_flow(df):
    """평균유량 산출"""
    grouped = df.groupby(['시군', '단위유역', '시설명'])['유량'].mean().reset_index()
    grouped['평균유량'] = grouped['유량'].round(1)
    return grouped[['시군', '단위유역', '시설명', '평균유량']]


# %% ---------------------------------------------------------------------------
# 4. 메인 실행 블록 (3. 최종 정리)

processor = DataProcessor(WORKING_DIR)
df_status, df_raw = processor.load_data()

if not df_raw.empty and not df_status.empty:
    df_clean = processor.clean_data(df_raw, df_status)
    
    print("데이터 정리 완료 및 검증/산정 시작...")
    
    # 정규성 검증
    bod_normality = run_normality_tests(df_clean, "BOD")
    tp_normality = run_normality_tests(df_clean, "TP")
    
    # 기준배출수질 산정
    bod_parametric = calc_parametric_limit(df_clean, "BOD")
    tp_parametric = calc_parametric_limit(df_clean, "TP")
    bod_nonparametric = calc_nonparametric_limit(df_clean, "BOD")
    tp_nonparametric = calc_nonparametric_limit(df_clean, "TP")
    
    # 평균유량
    df_mean_flow = calc_mean_flow(df_clean)
    
    # 2030 계획 데이터 로드
    plan_path = WORKING_DIR / "기준배출수질/환경기초시설_2030년_계획.xlsx"
    try:
        df_plan = pd.read_excel(plan_path)
        df_plan = df_plan.drop(columns=['준공연도', '시설코드'], errors='ignore')
    except Exception as e:
        print(f"[오류] 2030년 계획 파일 로드 실패: {e}")
        df_plan = pd.DataFrame()
        
    print("최종 정리 작업 수행 중...")
        
    if not df_plan.empty:
        # 기준배출수질(가, 나) 등의 데이터프레임을 조인할 때 필요한 계산 열들이 자동으로 포함됨
        df_final = pd.merge(df_plan, df_mean_flow, on=['시군', '단위유역', '시설명'], how='outer')
        
        # 정규성 검증 전체 데이터 병합 (이전 코드에서는 ['시설명', 'BOD_정규성']만 선택했으나, 이제 전체 병합)
        df_final = pd.merge(df_final, bod_normality, on='시설명', how='left')
        df_final = pd.merge(df_final, bod_parametric, on='시설명', how='left')
        df_final = pd.merge(df_final, bod_nonparametric, on='시설명', how='left')
        
        df_final = pd.merge(df_final, tp_normality, on='시설명', how='left', suffixes=('', '_TP'))
        
        # 이름 충돌 방지를 위해 TP 병합 시 중복 컬럼(n 등) 정리
        if 'n_TP' in df_final.columns:
            df_final = df_final.rename(columns={'n_TP': 'n_TP'})
            
        df_final = pd.merge(df_final, tp_parametric, on='시설명', how='left')
        df_final = pd.merge(df_final, tp_nonparametric, on='시설명', how='left', suffixes=('', '_TP'))
        
        df_final = pd.merge(df_final, df_status[['시설명', '가동개시', '용량']], on='시설명', how='left')
        
        # 카테고리 설정
        df_final['시군'] = pd.Categorical(df_final['시군'], categories=ORDER_CITY, ordered=True)
        df_final['단위유역'] = pd.Categorical(df_final['단위유역'], categories=ORDER_BASIN, ordered=True)
        
        # 샘플 개수에 따른 정규성 분류
        cond_bod_small = df_final['BOD_개수'] < 30
        cond_bod_large = df_final['BOD_개수'] >= 347
        df_final['BOD_정규성'] = np.where(df_final['BOD_정규성'].isna(), np.nan,
                                    np.where(cond_bod_small, "n<30",
                                        np.where(cond_bod_large, "n>=347", df_final['BOD_정규성'])))
                                        
        cond_tp_small = df_final['TP_개수'] < 30
        cond_tp_large = df_final['TP_개수'] >= 347
        df_final['TP_정규성'] = np.where(df_final['TP_정규성'].isna(), np.nan,
                                    np.where(cond_tp_small, "n<30",
                                        np.where(cond_tp_large, "n>=347", df_final['TP_정규성'])))
                                        
        # BOD_기준 산정
        conds_bod = [
            df_final['BOD_정규성'] == "정규성",
            df_final['BOD_정규성'] == "비정규성",
            df_final['BOD_정규성'] == "n>=347",
            df_final['BOD_정규성'] == "n<30"
        ]
        choices_bod = [df_final['BOD_가'], df_final['BOD_나'], df_final['BOD_나'], df_final['BOD_최대']]
        df_final['BOD_기준'] = np.select(conds_bod, choices_bod, default=np.nan)
        df_final['BOD_기준'] = df_final['BOD_기준'].astype(float).round(1)
        
        # TP_기준 산정
        conds_tp = [
            df_final['TP_정규성'] == "정규성",
            df_final['TP_정규성'] == "비정규성",
            df_final['TP_정규성'] == "n>=347",
            df_final['TP_정규성'] == "n<30"
        ]
        choices_tp = [df_final['TP_가'], df_final['TP_나'], df_final['TP_나'], df_final['TP_최대']]
        df_final['TP_기준'] = np.select(conds_tp, choices_tp, default=np.nan)
        df_final['TP_기준'] = df_final['TP_기준'].astype(float).round(3)
        
        # 구분 결정
        conds_status = [
            df_final['구분'].isna(),
            (df_final['구분'] == "미준공") & df_final['가동개시'].isna(),
            df_final['가동개시'].isna()
        ]
        choices_status = ["신규", "제외", "폐쇄"]
        df_final['구분'] = np.select(conds_status, choices_status, default="기존")
        
        # 필터링 및 정렬
        df_final = df_final[
            (~df_final['단위유역'].isin(["북한D", "임진A"])) &
            (df_final['구분'] != "제외")
        ]
        df_final = df_final.sort_values(by=['시군', '단위유역', '구분', '가동개시', '시설명'])
        
        # 컬럼 재배치
        cols = df_final.columns.tolist()
        for c in ['가동개시', '용량', '평균유량', 'BOD_기준', 'TP_기준']:
            if c in cols: cols.remove(c)
            
        try:
            idx_tp30 = cols.index('TP_30년') + 1
            cols.insert(idx_tp30, 'TP_기준')
            cols.insert(idx_tp30, 'BOD_기준')
            cols.insert(idx_tp30, '평균유량')
        except ValueError:
            cols.extend(['평균유량', 'BOD_기준', 'TP_기준'])
            
        try:
            idx_gubun = cols.index('구분') + 1
            cols.insert(idx_gubun, '용량')
            cols.insert(idx_gubun, '가동개시')
        except ValueError:
            cols.extend(['가동개시', '용량'])
            
        df_final = df_final[cols]
        print("최종 데이터 정리 완료.")


# %% ---------------------------------------------------------------------------
# 5. 최종 정리 자료 엑셀 파일 내보내기

if 'df_final' in locals() and not df_final.empty:
    output_path = OUTPUT_DIR / "기준배출수질_2024년기준_py.xlsx"
    sheets = {
        "기준배출수질_최종": df_final,
        "BOD_정규성검증": bod_normality,
        "BOD_기준배출수질_가": bod_parametric,
        "BOD_기준배출수질_나": bod_nonparametric,
        "TP_정규성검증": tp_normality,
        "TP_기준배출수질_가": tp_parametric,
        "TP_기준배출수질_나": tp_nonparametric
    }
    
    try:
        with pd.ExcelWriter(output_path) as writer:
            for s_name, df_sheet in sheets.items():
                df_sheet.to_excel(writer, sheet_name=s_name, index=False)
        print(f"엑셀 출력 완료: {output_path}")
    except Exception as e:
        print(f"[오류] 엑셀 파일 출력 실패: {e}")
else:
    print("[경고] 최종 데이터가 없습니다. 엑셀 파일이 생성되지 않습니다.")

# %%
