# %% 패키지 로드
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import product

# %% 설정 가능한 파라미터
INITIAL_CAPITAL = 10000  # 초기 원금
CONTRIBUTION = 250       # 매 싸이클 적립금
CYCLE_WEEKS = 2          # 싸이클 주기 (주 단위)
INVESTMENT_YEARS = 10    # 투자 기간 (년)
G_VALUE = 10             # 초기 G값
BAND_WIDTH = 0.15        # 밴드폭 (±15%)
POOL_USAGE_RATIO = 0.75  # Pool 사용 제한 비율 (적립식 VR 기준)
START_DATE = '2014-01-01'  # 백테스트 시작 날짜 (조정 가능)

# %% 투자 전략 유형
STRATEGY_TYPE = '적립식VR'  # '적립식VR', '거치식VR', '인출식VR'

# %% 백테스트 종료 날짜 계산
end_date = datetime.strptime(START_DATE, '%Y-%m-%d') + timedelta(days=INVESTMENT_YEARS * 365)

# %% 데이터 다운로드 함수
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval='1d')
    data = data['Adj Close']
    data = data.fillna(method='ffill').dropna()  # 추가적인 NaN 처리
    return data

# %% 성과 지표 계산 함수
def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / (periods / 52)) - 1

def calculate_mdd(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

# %% 백테스트 시뮬레이션 함수
def backtest_vr_strategy(data, initial_capital, contribution, cycle_weeks, g_value, band_width, pool_usage_ratio, strategy_type):
    cycle_days = cycle_weeks * 7
    num_cycles = len(data) // cycle_days
    
    capital = initial_capital
    pool = 0
    V = initial_capital  # V1 초기값 설정
    shares = 0
    history = []
    
    for cycle in range(num_cycles):
        start_idx = cycle * cycle_days
        end_idx = start_idx + cycle_days
        if end_idx > len(data):
            end_idx = len(data)
        
        cycle_data = data.iloc[start_idx:end_idx]
        if cycle_data.empty:
            break
        
        last_price = cycle_data[-1]
        E = shares * last_price
        V2 = V + pool / g_value + (E - V) / (2 * np.sqrt(g_value))
        
        # 전략 유형에 따른 Pool 관리
        if strategy_type == '적립식VR':
            pool += contribution
        elif strategy_type == '인출식VR':
            pool -= contribution
        
        # 최소/최대 밴드 계산
        min_band = V * (1 - band_width)
        max_band = V * (1 + band_width)
        
        # 평가금 계산
        current_evaluation = shares * last_price
        
        # 리밸런싱 결정
        if current_evaluation < min_band:
            while pool >= last_price:
                shares += 1
                pool -= last_price
                current_evaluation = shares * last_price
        elif current_evaluation > max_band:
            while shares > 0:
                shares -= 1
                pool += last_price
                current_evaluation = shares * last_price
        
        # 다음 싸이클 V값 계산
        V = V2
        
        # Pool 사용 제한
        pool_limit = initial_capital * pool_usage_ratio
        pool = min(pool, pool_limit)
        
        # 포트폴리오 가치 기록
        portfolio_value = shares * last_price + pool
        history.append({
            'Cycle': cycle,
            'Date': cycle_data.index[-1],
            'Shares': shares,
            'Pool': pool,
            'Portfolio Value': portfolio_value
        })
    
    history_df = pd.DataFrame(history)
    return history_df

# %% 최적화 탐색 함수
def optimize_parameters(data, initial_capital, contribution, cycle_weeks, strategy_type):
    g_values = [5, 10, 15]  # G 값 후보
    band_widths = [0.1, 0.15, 0.2]  # 밴드폭 후보
    pool_ratios = [0.5, 0.75, 1.0]  # Pool 사용 비율 후보

    best_cagr = -float('inf')
    best_params = None
    results = []

    for g, band, pool in product(g_values, band_widths, pool_ratios):
        strategy_history = backtest_vr_strategy(
            data=data,
            initial_capital=initial_capital,
            contribution=contribution,
            cycle_weeks=cycle_weeks,
            g_value=g,
            band_width=band,
            pool_usage_ratio=pool,
            strategy_type=strategy_type
        )
        final_value = strategy_history['Portfolio Value'].iloc[-1]
        cagr = calculate_cagr(initial_capital, final_value, INVESTMENT_YEARS * 52)
        mdd = calculate_mdd(strategy_history['Portfolio Value'])

        results.append({'G': g, 'Band Width': band, 'Pool Ratio': pool, 'CAGR': cagr, 'MDD': mdd})

        if cagr > best_cagr:
            best_cagr = cagr
            best_params = (g, band, pool)

    # 최적 파라미터 출력
    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"Best Parameters: G={best_params[0]}, Band Width={best_params[1]}, Pool Ratio={best_params[2]}")
    return results_df

# %% 메인 실행 함수
def main():
    tickers = ['TQQQ', 'QLD']
    data = download_data(tickers, START_DATE, end_date.strftime('%Y-%m-%d'))
    
    # 백테스트 전략 실행
    tqqq_data = data['TQQQ']
    qld_data = data['QLD']
    
    strategy_history = backtest_vr_strategy(
        data=tqqq_data,
        initial_capital=INITIAL_CAPITAL,
        contribution=CONTRIBUTION,
        cycle_weeks=CYCLE_WEEKS,
        g_value=G_VALUE,
        band_width=BAND_WIDTH,
        pool_usage_ratio=POOL_USAGE_RATIO,
        strategy_type=STRATEGY_TYPE
    )
    
    # CSV로 저장
    strategy_history.to_csv('vr_strategy_history.csv', index=False)
    print("백테스트 결과가 'vr_strategy_history.csv'에 저장되었습니다.")
    
    # 전략의 최종 가치 계산
    strategy_final_value = strategy_history['Portfolio Value'].iloc[-1]
    
    # CAGR 및 MDD 계산
    num_weeks = INVESTMENT_YEARS * 52
    strategy_cagr = calculate_cagr(INITIAL_CAPITAL, strategy_final_value, num_weeks)
    strategy_mdd = calculate_mdd(strategy_history['Portfolio Value'])
    
    # TQQQ 및 QLD의 성과 계산
    tqqq_start = tqqq_data.iloc[0]
    tqqq_end = tqqq_data.iloc[-1]
    tqqq_cagr = calculate_cagr(tqqq_start, tqqq_end, num_weeks)
    tqqq_mdd = calculate_mdd(tqqq_data / tqqq_data.iloc[0] * INITIAL_CAPITAL)
    
    qld_start = qld_data.iloc[0]
    qld_end = qld_data.iloc[-1]
    qld_cagr = calculate_cagr(qld_start, qld_end, num_weeks)
    qld_mdd = calculate_mdd(qld_data / qld_data.iloc[0] * INITIAL_CAPITAL)
    
    # 결과 출력
    print(f"--- 백테스트 결과 ({STRATEGY_TYPE}) ---")
    print(f"CAGR: {strategy_cagr*100:.2f}%")
    print(f"MDD: {strategy_mdd*100:.2f}%")
    print("\n--- 비교 대상 ---")
    print(f"TQQQ CAGR: {tqqq_cagr*100:.2f}%")
    print(f"TQQQ MDD: {tqqq_mdd*100:.2f}%")
    print(f"QLD CAGR: {qld_cagr*100:.2f}%")
    print(f"QLD MDD: {qld_mdd*100:.2f}%")
    
    # 성과 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(strategy_history['Date'], strategy_history['Portfolio Value'], label='VR Strategy', linewidth=2)
    plt.plot(data.index, data['TQQQ'] / data['TQQQ'].iloc[0] * INITIAL_CAPITAL, label='TQQQ Buy and Hold', linestyle='--')
    plt.plot(data.index, data['QLD'] / data['QLD'].iloc[0] * INITIAL_CAPITAL, label='QLD Buy and Hold', linestyle=':')
    plt.fill_between(
        strategy_history['Date'],
        strategy_history['Portfolio Value'].cummax(),
        strategy_history['Portfolio Value'],
        color='red',
        alpha=0.3,
        label='Drawdown'
    )
    plt.legend()
    plt.title('VR Strategy vs Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()

    # 최적화 실행
    optimize_parameters(
        data=tqqq_data,
        initial_capital=INITIAL_CAPITAL,
        contribution=CONTRIBUTION,
        cycle_weeks=CYCLE_WEEKS,
        strategy_type=STRATEGY_TYPE
    )

if __name__ == "__main__":
    main()
