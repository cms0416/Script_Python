# 수질 평가 로직(연평균, 달성률, 평가수질) 모듈

import numpy as np
import pandas as pd
from .utils import round_half_up

def calculate_annual_mean(df, item, target_col):
    """
    연평균 산정 함수
    """
    # 소수점 자리수 설정
    dp = 3 if item == 'TP' else 1
    
    # 그룹핑 기준 컬럼의 결측치(NaN) 채우기
    # pivot_table은 인덱스로 사용되는 컬럼에 NaN이 있으면 해당 행을 삭제하므로,
    # 미리 빈 문자열("")로 변환하여 데이터를 보존해야 함.
    temp_df = df.copy()
    temp_df[['강원도', '권역']] = temp_df[['강원도', '권역']].fillna("")
    
    # 그룹바이 연산
    res = temp_df.groupby(['강원도', '권역', '총량지점명', target_col, '연도'], observed=True, dropna=False)[item]\
            .mean().reset_index()
            
    res['mean_value'] = res[item].apply(lambda x: round_half_up(x, dp))
    
    # Pivot (Long to Wide)
    pivot_res = res.pivot_table(
        index=['강원도', '권역', '총량지점명', target_col], 
        columns='연도', 
        values='mean_value'
    ).reset_index()
    
    return pivot_res

def calculate_achievement(df, item, target_col, year):
    """
    달성률 산정 함수 (단일 연도 기준 3년치)
    """
    # 3년치 데이터 필터링
    target_years = range(year - 2, year + 1)
    sub_df = df[
        (df['평가방식'] == '달성률') & 
        (df['연도'].isin(target_years)) & 
        (df[item].notna())
    ].copy()
    
    if sub_df.empty:
        return pd.DataFrame()

    # 달성 여부 (수질 <= 목표수질)
    sub_df['달성여부'] = np.where(sub_df[item] <= sub_df[target_col], 1, 0)
    
    # 그룹별 통계
    # R 로직: rank(ties.method='first'), ceiling(Total * 0.625)
    def calc_group(g):
        total = len(g)
        if total == 0: return None
        
        # 달성률 만족 횟수 (기준 순위)
        req_count = int(np.ceil(total * 0.625))
        
        # 값을 오름차순 정렬하여 기준 순위에 해당하는 값 찾기 (0-index이므로 -1)
        sorted_vals = sorted(g[item])
        criteria_val = sorted_vals[req_count - 1]
        
        success_count = g['달성여부'].sum()
        achievement_rate = round_half_up(success_count / total, 3)
        
        return pd.Series({
            '달성기준수질': criteria_val,
            '달성률': achievement_rate
        })

    result = sub_df.groupby('총량지점명', observed=True)[[item, '달성여부']].apply(calc_group).reset_index()
    
    # 평가기간 컬럼 생성 (예: 23~25)
    period_name = f"{year-2002}~{year-2000}"
    result['평가기간'] = period_name
    
    # Pivot 형태 반환 (총량지점명, 기간, 달성률)
    final = result[['총량지점명', '평가기간', '달성률']].copy()
    final = final.pivot(index='총량지점명', columns='평가기간', values='달성률').reset_index()
    
    return final

def calculate_assessment_val(df, item, year):
    """
    평가수질(변환평균) 산정 함수
    """
    target_years = range(year - 2, year + 1)
    sub_df = df[
        (df['평가방식'] == '변환평균') & 
        (df['연도'].isin(target_years))
    ].copy()
    
    if sub_df.empty:
        return pd.DataFrame()

    # 자연로그 변환
    col_ln = f"{item}_ln"
    sub_df[col_ln] = np.log(sub_df[item])
    
    dp = 3 if item == 'TP' else 1
    
    def calc_log_mean(g):
        mean_ln = g[col_ln].mean()
        var_ln = g[col_ln].var(ddof=1)
        
        # exp(mean + var/2)
        val = np.exp(mean_ln + var_ln / 2)
        return round_half_up(val, dp)

    result = sub_df.groupby('총량지점명', observed=True)[[col_ln]].apply(calc_log_mean).reset_index(name='value')
    
    period_name = f"{year-2002}~{year-2000}"
    result = result.rename(columns={'value': period_name})
    
    return result

def analyze_seasonal(df):
    """
    계절별 달성 현황 분석
    """
    # 데이터 복사 및 계절 분류
    temp = df[df['강원도'] == '강원도'].copy()
    
    conditions = [
        (temp['월'] >= 3) & (temp['월'] <= 5),
        (temp['월'] >= 6) & (temp['월'] <= 8),
        (temp['월'] >= 9) & (temp['월'] <= 11)
    ]
    choices = ['봄', '여름', '가을']
    temp['계절'] = np.select(conditions, choices, default='겨울')
    
    # 달성 여부
    temp['BOD_달성'] = np.where(temp['BOD'] <= temp['BOD_목표수질'], 1, 0)
    temp['TP_달성'] = np.where(temp['TP'] <= temp['TP_목표수질'], 1, 0)
    temp['전체'] = 1
    
    # 계절별 집계
    seasonal_sum = temp[temp['TP'].notna()].groupby(['총량지점명', '연도', '계절'], observed=True)[['전체', 'BOD_달성', 'TP_달성']].sum().reset_index()
    
    # 소계(Total) 추가 로직은 Pandas에서 pivot_table margins=True 등 사용 가능하나,
    # 여기서는 지점별/연도별 합계를 별도로 구해 concat 하는 방식이 명확함.
    # (복잡도를 줄이기 위해 단순 집계 로직만 구현)
    
    seasonal_sum['BOD_달성률'] = (seasonal_sum['BOD_달성'] / seasonal_sum['전체']).apply(lambda x: round_half_up(x, 3))
    seasonal_sum['TP_달성률'] = (seasonal_sum['TP_달성'] / seasonal_sum['전체']).apply(lambda x: round_half_up(x, 3))
    
    return seasonal_sum