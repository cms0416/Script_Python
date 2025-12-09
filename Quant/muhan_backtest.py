import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------------------------------------
# 0. 가격 데이터 로더 (yfinance + 재시도)
# -----------------------------------------------------------

def load_price_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str = None,
    max_retries: int = 3,
    sleep_sec: int = 5,
) -> pd.Series:
    """
    야후 파이낸스에서 종가(Adjusted Close)를 받아오는 함수.
    - Too Many Requests 등 오류 발생 시 재시도.
    - 반드시 Series(종가)만 반환.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Downloading {ticker} price data (try {attempt}/{max_retries})...")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # 분배금/액면분할 반영된 Adjusted 가격
                progress=False,
            )
            if df.empty:
                raise ValueError("Empty data returned from Yahoo Finance.")
            # 여기서 반드시 Series로 변환
            close = df["Close"].dropna()
            if close.empty:
                raise ValueError("No valid Close data returned.")
            print(f"[INFO] Download complete: {ticker}, rows={len(close)}")
            return close

        except Exception as e:
            last_err = e
            print(f"[WARN] Failed to download {ticker}: {e}")
            if attempt < max_retries:
                print(f"[INFO] Retry after {sleep_sec} seconds...")
                time.sleep(sleep_sec)
            else:
                print("[ERROR] Max retries reached. Aborting.")
                raise

    raise last_err  # 방어 코드 (논리상 도달 X)


# -----------------------------------------------------------
# 1. 유틸 함수: MDD, CAGR 계산
# -----------------------------------------------------------

def compute_mdd(equity: pd.Series) -> float:
    """
    최대낙폭(MDD) 계산.
    equity: 일별 자산 곡선 (Series)
    return: MDD (음수, 예: -0.25)
    """
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    return float(dd.min())


def compute_cagr(equity: pd.Series) -> float:
    """
    CAGR 계산.
    equity: 일별 자산 곡선 (Series)
    """
    if equity.empty:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0

    num_days = len(equity)
    years = num_days / 252.0  # 거래일 기준 대략
    if years <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / years) - 1


# -----------------------------------------------------------
# 2. 전략 결과 저장용 데이터클래스
# -----------------------------------------------------------

@dataclass
class StrategyResult:
    final_equity: float
    cagr: float
    mdd: float
    num_cycles: int
    params: Dict[str, float]
    equity_curve: pd.Series


# -----------------------------------------------------------
# 3. 무한매수법 단일 백테스트 함수 (단순화 버전)
# -----------------------------------------------------------

def backtest_infinite_buying(
    prices: pd.Series,
    splits: int,
    target_return: float,
    initial_capital: float = 10_000.0,
) -> StrategyResult:
    """
    단일 파라미터 조합에 대해 무한매수법 단순화 버전을 백테스트.

    prices        : 일별 종가 Series (index: DatetimeIndex)
    splits        : 분할수 N (예: 40, 80, ...)
    target_return : 목표수익률(예: 0.10 = 10%)
    initial_capital : 초기 원금(무한매수법 원금 개념)
    """

    # 만약 DataFrame이 들어오면 방어적으로 Close 또는 첫 번째 컬럼 선택
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            prices = prices["Close"]
        else:
            prices = prices.iloc[:, 0]

    prices = prices.dropna()
    if prices.empty:
        return StrategyResult(
            final_equity=initial_capital,
            cagr=0.0,
            mdd=0.0,
            num_cycles=0,
            params={"splits": splits, "target_return": target_return},
            equity_curve=pd.Series(dtype=float),
        )

    # 상태 변수
    internal_capital = initial_capital  # 무한매수 원금(사이클별 기준)
    external_reserve = 0.0             # 사이클마다 50% 인출되는 비상금 개념

    cash = internal_capital            # 현금 (내부 계좌)
    position = 0                       # 보유 주식 수
    avg_price = 0.0                    # 평단가
    total_buy_amount = 0.0             # 누적 매수금(원금 기준, 가격변동과 무관)

    in_cycle = False
    cycle_start_internal_capital = internal_capital
    num_cycles = 0

    equity_list = []

    # 메인 루프
    for date, price in prices.items():
        price = float(price)
        if price <= 0:
            equity_list.append(cash + position * price + external_reserve)
            continue

        one_buy = internal_capital / splits

        # ---------------------------------------------------
        # 1) 사이클 시작 (position==0 & in_cycle==False)
        # ---------------------------------------------------
        if (position == 0) and (not in_cycle):
            max_shares = math.floor(one_buy / price)
            if max_shares > 0 and cash >= max_shares * price:
                cost = max_shares * price
                cash -= cost
                position += max_shares
                avg_price = price
                total_buy_amount = cost
                in_cycle = True
                cycle_start_internal_capital = internal_capital

        # ---------------------------------------------------
        # 2) 사이클 진행 중인 경우 (매도 → 매수 순서)
        # ---------------------------------------------------
        elif in_cycle and position > 0:
            # T 계산 (누적 매수금 기준)
            T = math.ceil(total_buy_amount / one_buy) if one_buy > 0 else 0
            T = max(0, min(T, splits))

            # 매매 기준 (trade_basis)
            trade_basis = target_return * (1.0 - 2.0 * T / splits)

            # 2-1) 매도 로직
            takeprofit_price = avg_price * (1.0 + target_return)
            partial_price = avg_price * (1.0 + trade_basis)

            # (1) 전량 매도 조건: 목표수익률 도달
            if price >= takeprofit_price:
                cash += position * price
                position = 0
                avg_price = 0.0
                total_buy_amount = 0.0

            else:
                # (2) 부분 매도 조건: trade_basis에 따른 회수/손절 (1/4씩)
                part_qty = math.floor(position * 0.25)
                if part_qty > 0:
                    # trade_basis >= 0 인 경우: 수익 일부 실현
                    if trade_basis >= 0 and price >= partial_price:
                        cash += part_qty * price
                        position -= part_qty
                    # trade_basis < 0 인 경우: 손절 성격
                    elif trade_basis < 0 and price <= partial_price:
                        cash += part_qty * price
                        position -= part_qty

            # 2-2) 매수 로직 (전반전/후반전 구분)
            if position > 0:
                # T는 매도 후에도 동일 기준으로 사용
                T = math.ceil(total_buy_amount / one_buy) if one_buy > 0 else 0

                if T < splits / 2:
                    # 전반전: 평단 이하 혹은 평단+trade_basis 부근
                    buy_qty = 0
                    # (1) 평단 이하: 전체 one_buy
                    if price <= avg_price:
                        buy_qty = math.floor(one_buy / price)
                    # (2) 평단 < price <= avg_price*(1+max(trade_basis,0)) : 0.5 one_buy
                    else:
                        upper = avg_price * (1.0 + max(trade_basis, 0.0))
                        if price <= upper:
                            buy_qty = math.floor((one_buy * 0.5) / price)

                    if buy_qty > 0:
                        buy_cost = buy_qty * price
                        # 내부 원금 한도(누적 매수금 ≤ internal_capital) 체크
                        if total_buy_amount + buy_cost > internal_capital:
                            buy_cost = max(0.0, internal_capital - total_buy_amount)
                            buy_qty = math.floor(buy_cost / price)

                        if buy_qty > 0 and cash >= buy_qty * price:
                            buy_cost = buy_qty * price
                            new_cost_basis = avg_price * position + buy_cost
                            position += buy_qty
                            avg_price = new_cost_basis / position
                            cash -= buy_cost
                            total_buy_amount += buy_cost

                else:
                    # 후반전: 평단 대비 trade_basis 이하에서만 매수 (더 싼 가격 위주)
                    buy_qty = 0
                    buy_trigger_price = avg_price * (1.0 + trade_basis)
                    if price <= buy_trigger_price:
                        buy_qty = math.floor(one_buy / price)

                    if buy_qty > 0:
                        buy_cost = buy_qty * price
                        if total_buy_amount + buy_cost > internal_capital:
                            buy_cost = max(0.0, internal_capital - total_buy_amount)
                            buy_qty = math.floor(buy_cost / price)

                        if buy_qty > 0 and cash >= buy_qty * price:
                            buy_cost = buy_qty * price
                            new_cost_basis = avg_price * position + buy_cost
                            position += buy_qty
                            avg_price = new_cost_basis / position
                            cash -= buy_cost
                            total_buy_amount += buy_cost

        # ---------------------------------------------------
        # 3) 사이클 종료 처리 (position == 0이 된 경우)
        # ---------------------------------------------------
        if in_cycle and position == 0:
            cycle_end_internal = cash
            cycle_pnl = cycle_end_internal - cycle_start_internal_capital

            if cycle_pnl > 0:
                # 이익: 50% 재투입, 50% 외부 비상금
                half = cycle_pnl * 0.5
                internal_capital = cycle_start_internal_capital + half
                external_reserve += cycle_pnl - half
            else:
                # 손실: internal_capital 원상복구, 손익은 비상금에서 조정
                internal_capital = cycle_start_internal_capital
                external_reserve += cycle_pnl  # 음수 반영

            cash = internal_capital
            total_buy_amount = 0.0
            avg_price = 0.0
            in_cycle = False
            num_cycles += 1

        # ---------------------------------------------------
        # 4) 일별 자산(Eq) 기록: 내부 + 외부 비상금 모두 포함
        # ---------------------------------------------------
        equity = cash + position * price + external_reserve
        equity_list.append(equity)

    equity_series = pd.Series(equity_list, index=prices.index)
    final_equity = float(equity_series.iloc[-1])
    cagr = compute_cagr(equity_series)
    mdd = compute_mdd(equity_series)

    return StrategyResult(
        final_equity=final_equity,
        cagr=cagr,
        mdd=mdd,
        num_cycles=num_cycles,
        params={"splits": splits, "target_return": target_return},
        equity_curve=equity_series,
    )


# -----------------------------------------------------------
# 4. Monte-Carlo 엔진 (prices를 받아서 돌리는 버전)
# -----------------------------------------------------------

def monte_carlo_infinite_buying(
    prices: pd.Series,
    initial_capital: float = 10_000.0,
    n_iter: int = 1000,
    splits_range: Tuple[int, int] = (10, 100),
    tp_range: Tuple[float, float] = (0.05, 0.20),
    seed: int = 42,
) -> pd.DataFrame:
    """
    무한매수법 파라미터 Monte-Carlo 탐색.

    prices       : 일별 종가 Series (이미 다운로드된 데이터)
    initial_capital : 초기 원금
    n_iter       : 랜덤 파라미터 조합 수
    splits_range : (분할수 최소, 최대)
    tp_range     : (목표수익률 최소, 최대)  예: (0.05, 0.20)
    seed         : 난수 시드 (재현성 확보용)
    """

    random.seed(seed)
    np.random.seed(seed)

    results: List[Dict] = []

    for i in range(n_iter):
        splits = random.randint(splits_range[0], splits_range[1])
        tp = random.uniform(tp_range[0], tp_range[1])

        res = backtest_infinite_buying(
            prices=prices,
            splits=splits,
            target_return=tp,
            initial_capital=initial_capital,
        )

        results.append(
            {
                "splits": splits,
                "target_return": tp,
                "final_equity": res.final_equity,
                "cagr": res.cagr,
                "mdd": res.mdd,
                "num_cycles": res.num_cycles,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("final_equity", ascending=False).reset_index(drop=True)
    return df


# -----------------------------------------------------------
# 5. 사용 예시
# -----------------------------------------------------------

if __name__ == "__main__":
    # 예시: TQQQ에 대해 2015년 이후 Monte-Carlo 1000회
    ticker = "TQQQ"
    start_date = "2015-01-01"
    end_date = None  # 오늘까지

    # 1) 가격 데이터 1회 다운로드
    close_prices = load_price_data(
        ticker=ticker,
        start=start_date,
        end=end_date,
        max_retries=3,
        sleep_sec=5,
    )

    # 2) Monte-Carlo 파라미터 탐색
    mc_df = monte_carlo_infinite_buying(
        prices=close_prices,
        initial_capital=10_000.0,
        n_iter=1000,
        splits_range=(10, 100),
        tp_range=(0.05, 0.20),
        seed=42,
    )

    # 상위 20개 전략 출력
    print(mc_df.head(20))

    # CSV 저장
    out_name = f"mc_results_{ticker}.csv"
    mc_df.to_csv(out_name, index=False)
    print(f"Saved: {out_name}")
