# %% [markdown]
# ******************************************************************************
# # 수질오염총량제 목표수질 평가 및 현황 분석


# %% [markdown]
# ******************************************************************************
# ### 1. 라이브러리 및 설정 로드

# 라이브러리 로드
import sys
from pathlib import Path
import pandas as pd
from functools import reduce

# 프로젝트 경로 설정 (이 부분은 동일)
ROOT = Path(r"E:/Coding/Script_Python")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# [수정됨] 사용자 정의 모듈 로드 경로 변경 (wq_analysis -> target_wq_eval)
from TMDL.target_wq_eval.package import config
from TMDL.target_wq_eval.package import data_manager
from TMDL.target_wq_eval.package import evaluator
from TMDL.target_wq_eval.package import exporter

# 데이터 경로 정의 (데이터 저장 경로는 기존 E:/Coding/TMDL 유지)
DATA_PATH = Path(r"E:/Coding/TMDL")
OUTPUT_PATH = DATA_PATH / "수질분석/Output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True) 

# 평가 연도 설정
FINAL_YEAR = 2025
START_YEAR = FINAL_YEAR - 10
YEARS = range(START_YEAR + 2, FINAL_YEAR + 1)

# %% [markdown]
# ******************************************************************************
# ### 2. 데이터 로드 및 전처리(총량측정망자료_합치기)

print(">> 데이터 로드 및 전처리 시작...")

# 2.1. 원본 엑셀 파일 로드
raw_df, hantan_df = data_manager.load_and_merge_raw_data(DATA_PATH)

# 2.2. 데이터 전처리 (형식 변환, 한탄A 처리 등)
processed_df = data_manager.process_data(raw_df, hantan_df)

# 2.3. 전체 지점 필터링 및 정렬
total_df = data_manager.filter_by_station(processed_df)

# 2.4. 강원 지역 필터링
gw_df = data_manager.filter_by_station(processed_df, stations=config.GW_STATIONS)

# 결과 확인
processed_df

# %% ---------------------------------------------------------------------------
# 2.5. 전처리 결과 엑셀 저장(중간 산출물)
print(">> 중간 산출물 엑셀 저장 중...")

output_filepath = DATA_PATH / "수질분석/총량측정망_2007_2025.xlsx"

# 시트 순서 지정
data_to_save = {
    "강원도": gw_df,
    "단위유역전체": total_df,
    "소유역포함": processed_df
}

# 엑셀 저장 모듈 호출
exporter.save_formatted_excel(output_filepath, data_to_save)

print(f">> 파일 저장 완료: {output_filepath}")

# %% [markdown]
# ******************************************************************************
# ### 3. 목표수질 평가 데이터 준비

print(">> 목표수질 평가 데이터 병합 중...")

# 3.1. 목표수질 데이터 로드
target_df = pd.read_excel(DATA_PATH / "수질분석/목표수질.xlsx")

# 3.2. 측정자료와 목표수질 병합
obs_df = total_df.merge(target_df, on='총량지점명', how='left')

# 3.3. 분석 기간 및 불필요 컬럼 필터링
obs_df = obs_df[obs_df['연도'] >= START_YEAR].copy()
cols_to_keep = ['강원도', '권역', '총량지점명', '연도', '월', 'BOD', 'TP', 'TOC', 
                'BOD_목표수질', 'TP_목표수질', 'TOC_목표수질', '평가방식']
obs_df = obs_df[cols_to_keep]

# 결과 확인
obs_df

# %% [markdown]
# ******************************************************************************
# ### 4. 항목별 연평균 및 달성률 산정

print(">> 평가 지표 산정 중...")

def process_item(item_name):
    target_col = f"{item_name}_목표수질"
    
    # 4.1. 연평균 산정
    df_ymean = evaluator.calculate_annual_mean(obs_df, item_name, target_col)
    
    # 4.2. 달성률 산정
    ach_list = []
    for y in YEARS:
        ach_res = evaluator.calculate_achievement(obs_df, item_name, target_col, y)
        if not ach_res.empty:
            ach_list.append(ach_res)
    
    if ach_list:
        df_ach = reduce(lambda left, right: pd.merge(left, right, on='총량지점명', how='outer'), ach_list)
    else:
        df_ach = pd.DataFrame(columns=['총량지점명'])

    # 4.3. 평가수질 산정
    assess_list = []
    for y in YEARS:
        assess_res = evaluator.calculate_assessment_val(obs_df, item_name, y)
        if not assess_res.empty:
            assess_list.append(assess_res)
            
    if assess_list:
        df_assess = reduce(lambda left, right: pd.merge(left, right, on='총량지점명', how='outer'), assess_list)
    else:
        df_assess = pd.DataFrame(columns=['총량지점명'])

    # 4.4. 전체 합치기
    # (1) 달성률(df_ach)과 평가수질(df_assess)을 위아래로 합침 (R의 bind_rows 대응)
    combined_metrics = pd.concat([df_ach, df_assess], ignore_index=True)

    # (2) 연평균(df_ymean)에 위에서 합친 데이터를 Left Join
    final = df_ymean.merge(combined_metrics, on='총량지점명', how='left')
    
    # 지점명 정렬
    final['총량지점명'] = pd.Categorical(final['총량지점명'], categories=config.STATION_ORDER_EVAL, ordered=True)
    return final.sort_values('총량지점명')

# 각 항목별 처리
bod_total = process_item('BOD')
tp_total = process_item('TP')
toc_total = process_item('TOC')

# 결과 확인
tp_total


# %% [markdown]
# ******************************************************************************
# ### 5. 결과 내보내기 (업무보고용 양식 포함)

print(">> 최종 결과 저장 중...")

# 5.1. BOD, TP 통합 양식
exclude_cols = [str(y) for y in range(START_YEAR, FINAL_YEAR + 1)]

# 강원도 데이터 필터링
bod_gw = bod_total[bod_total['강원도'] == '강원도'].copy()
tp_gw = tp_total[tp_total['강원도'] == '강원도'].copy()

# 연도별 평균(연평균) 컬럼 제외
bod_part = bod_gw.drop(columns=[c for c in bod_gw.columns if str(c) in exclude_cols], errors='ignore')
tp_part = tp_gw.drop(columns=[c for c in tp_gw.columns if str(c) in exclude_cols], errors='ignore')

# 1) 먼저 병합 수행(Suffix 적용)
merged_df = pd.merge(
    bod_part, 
    tp_part[['강원도', '권역', '총량지점명', 'TP_목표수질'] + [c for c in tp_part.columns if '~' in str(c)]],
    on=['강원도', '권역', '총량지점명'],
    suffixes=('_BOD', '_TP')
)

# 2) 컬럼 정렬을 위한 리스트 생성
# 고정 컬럼(앞부분에 위치할 정보)
fixed_cols = ['강원도', '권역', '총량지점명', 'BOD_목표수질', 'TP_목표수질']

# 기간 컬럼(물결표 '~'가 포함된 컬럼 찾기)
period_cols = [c for c in merged_df.columns if '~' in str(c)]

# 3) 기간 컬럼 정렬
period_cols.sort()

# 4) 최종 컬럼 재배치
bod_tp_integrated = merged_df[fixed_cols + period_cols]

# 5.2. 엑셀 저장
with pd.ExcelWriter(OUTPUT_PATH / "강원도_수질현황_py.xlsx") as writer:
    bod_gw.to_excel(writer, sheet_name="BOD", index=False)
    tp_gw.to_excel(writer, sheet_name="TP", index=False)
    toc_total[toc_total['강원도'] == '강원도'].to_excel(writer, sheet_name="TOC", index=False)
    bod_tp_integrated.to_excel(writer, sheet_name="BOD_TP_통합", index=False)

with pd.ExcelWriter(OUTPUT_PATH / "한강수계전체_수질현황_py.xlsx") as writer:
    bod_total.to_excel(writer, sheet_name="BOD", index=False)
    tp_total.to_excel(writer, sheet_name="TP", index=False)
    toc_total.to_excel(writer, sheet_name="TOC", index=False)

print(">> 파일 저장 완료")

# %% [markdown]
# ******************************************************************************
# ### 6. 계절별 달성 현황 분석

print(">> 계절별 분석 수행 중...")

seasonal_df = evaluator.analyze_seasonal(obs_df)

with pd.ExcelWriter(OUTPUT_PATH / "총량측정망_계절별달성현황_py.xlsx") as writer:
    seasonal_df.to_excel(writer, index=False)

print(">> 모든 분석이 완료되었습니다.")
# %%
