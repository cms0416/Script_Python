import numpy as np
import pandas as pd

from .config import ProjectConfig
from TMDL.TMDL_common.계절정의함수 import add_season


def load_target_quality(cfg: ProjectConfig) -> float:
    """
    목표수질 엑셀에서 해당 단위유역/항목의 목표수질을 가져온다.
    """
    analyte = cfg.analyte
    df = pd.read_excel(cfg.path_목표수질)

    값 = (
        df
        .loc[df['총량지점명'] == cfg.단위유역, analyte.target_col]
        .iloc[0]
    )
    return float(값)


def load_tmdl_raw(cfg: ProjectConfig, 목표수질: float) -> pd.DataFrame:
    """
    총량측정망 원본 로드 및 기본 전처리.
    - 단위유역/기간 필터
    - 계절 추가
    - 달성여부(TP > 목표수질 등)
    """
    analyte = cfg.analyte

    df = (
        pd.read_excel(cfg.path_총량측정망)
        .query(
            "총량지점명.str.contains(@cfg.단위유역) and 연도 >= @cfg.시작연도 and 연도 <= @cfg.종료연도",
            engine="python",
        )
        .dropna(subset=[analyte.col])
        .copy()
    )

    df = add_season(df)

    df['달성여부'] = np.where(
        df[analyte.col] > 목표수질, "초과", "달성"
    )

    return df


def prepare_tmdl(
    cfg: ProjectConfig,
    총량측정망_원본: pd.DataFrame,
    목표수질: float,
    기상지표_일별: pd.DataFrame,
):
    """
    총량측정망 자료 정리 + 기상지표 결합.
    - 부하량 계산
    - 유량 크기순서 / 백분율 / 유황구간
    - 기상지표_일별 merge
    - 총량측정망_기상(기상→수질 left join) 생성
    """
    analyte = cfg.analyte

    # 1. 원본 데이터 복사
    df = 총량측정망_원본.copy()

    # 2. 부하량 계산
    df[analyte.load_col] = df['유량'] * df[analyte.col] * 86.4
    df[analyte.target_load_col] = df['유량'] * 목표수질 * 86.4

    # 3. 유량 기준 내림차순 정렬
    df = df.sort_values(by='유량', ascending=False).reset_index(drop=True)

    # 4. 유량 크기 순서(동일값은 동일 순위)
    df['유량크기순서'] = df['유량'].rank(method='min', ascending=False)

    # 5. 유효개수 계산
    유량_유효개수 = df['유량'].notna().sum()

    # 6. 유량 백분율
    df['유량백분율'] = df['유량크기순서'] / 유량_유효개수 * 100

    # 7. 유황구간 정의
    bins = [0, 10, 40, 60, 90, 100]
    labels = ['홍수기', '풍수기', '평수기', '저수기', '갈수기']
    df['유황구간'] = pd.cut(
        df['유량백분율'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    # 8. 기상자료 병합(일자, 연도, 월 기준)
    df = pd.merge(
        df,
        기상지표_일별,
        on=['일자', '연도', '월'],
        how='left'
    )

    # 9. '유량크기순서'를 첫번째 열로 이동
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index('유량크기순서')))
    df = df[cols]

    # --- 총량측정망_기상 (기상→수질 left join) ---
    수질측정망_총량 = 총량측정망_원본.copy()
    총량측정망_기상 = pd.merge(
        기상지표_일별,
        수질측정망_총량,
        on=['일자', '연도', '월'],
        how='left'
    )
    총량측정망_기상 = add_season(총량측정망_기상)

    return df, 총량측정망_기상


def load_monitoring(cfg: ProjectConfig, 목표수질: float):
    """
    유역 내 수질측정망 정리.
    - 분석항목 결측 제거
    - 상한값(max_valid) 필터
    - 연도/월/계절 추가
    - 달성여부, 최근 5년 기준 평균·달성률 계산
    """
    analyte = cfg.analyte

    # 1. 원본 로드 후 바로 복사
    df = pd.read_excel(cfg.path_수질측정망).copy()

    # 2. 분석 항목 결측 제거 + 복사
    df = df.loc[df[analyte.col].notna()].copy()

    # 3. 상한값 필터(TP < 2 등) + 복사
    if analyte.max_valid is not None:
        df = df.loc[df[analyte.col] < analyte.max_valid].copy()

    # 4. 연도, 월, 계절
    df.loc[:, '일자'] = pd.to_datetime(df['일자'])
    df.loc[:, '연도'] = df['일자'].dt.year
    df.loc[:, '월'] = df['일자'].dt.month
    df = add_season(df)

    # 5. 달성여부
    df.loc[:, '달성여부'] = np.where(
        df[analyte.col] > 목표수질, '초과', '달성'
    )

    # 6. 최근 5년 자료 (query 사용하지 않음 → numexpr/한글 이슈 제거)
    max_year = int(df['연도'].max())
    mask_recent = df['연도'] > max_year - 5
    최근자료 = df.loc[mask_recent].copy()

    # 7. 측정소별 평균
    수질측정망_평균 = (
        최근자료
        .groupby('측정소명', as_index=False)
        .agg(**{
            f"{analyte.name}_평균": (analyte.col, lambda x: np.round(x.mean(skipna=True), 3)),
        })
    )

    # 8. 측정소별 달성률
    수질측정망_달성률 = (
        최근자료
        .groupby(['측정소명', '달성여부'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # '달성', '초과' 컬럼이 없을 가능성을 대비한 보정
    for col in ['달성', '초과']:
        if col not in 수질측정망_달성률.columns:
            수질측정망_달성률[col] = 0

    수질측정망_달성률['총계'] = (
        수질측정망_달성률['달성'] + 수질측정망_달성률['초과']
    )
    수질측정망_달성률['달성률'] = (
        수질측정망_달성률['달성']
        / 수질측정망_달성률['총계'].replace(0, np.nan)
        * 100
    ).round(1)

    # 9. 평균 + 달성률 병합
    수질측정망_결과 = pd.merge(
        수질측정망_평균,
        수질측정망_달성률[['측정소명', '달성률']],
        on='측정소명',
        how='left'
    )

    # 10. 최종 반환
    return df, 수질측정망_평균, 수질측정망_결과
