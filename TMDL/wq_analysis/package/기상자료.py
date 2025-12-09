import pandas as pd

from .config import ProjectConfig
from TMDL.TMDL_common.계절정의함수 import add_season  # 현재는 사용 안 하지만 향후 확장 대비


def load_weather_hourly(cfg: ProjectConfig) -> pd.DataFrame:
    """
    기상자료(시단위) 정리.
    - 기상대 필터
    - 기본 컬럼 정리
    """
    기상자료_시단위 = (
        pd.read_excel(cfg.path_기상_시)
        .query("지점명.str.contains(@cfg.기상대)", engine='python')  # regex 검색
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

    # 시간 단위 강우 강도 플래그
    기상자료_시단위['고강도강우'] = 기상자료_시단위['강수량'] >= 20
    기상자료_시단위['매우고강도강우'] = 기상자료_시단위['강수량'] >= 30

    return 기상자료_시단위


def build_daily_indicators(기상자료_시단위: pd.DataFrame) -> pd.DataFrame:
    """
    시간단위 기상자료를 이용해 일단위 기상지표 생성.
    - 일강수량, 최대시간강우량, 고강도/매우고강도 발생 여부
    - 일평균기온
    - 강우일 여부
    - 고강도강우 3일 누적 여부
    - 누적강수(3/4/5일)
    - 연도, 월
    """
    기상지표_일별 = (
        기상자료_시단위
        .dropna(subset=['강수량'])  # 전부 NA인 날 제거
        .groupby('일자')
        .agg({
            '강수량': [
                'sum',          # 일강수량
                'max',          # 최대시간강우량
                lambda x: (x >= 20).any(),  # 고강도강우_발생여부
                lambda x: (x >= 30).any(),  # 매우고강도강우_발생여부
            ],
            '기온': 'mean',     # 일평균기온
        })
    )

    # 컬럼 이름 정리
    기상지표_일별.columns = [
        '일강수량',
        '최대시간강우량',
        '고강도강우_발생여부',
        '매우고강도강우_발생여부',
        '평균기온',
    ]

    # 인덱스 리셋 및 부가 변수
    기상지표_일별 = 기상지표_일별.reset_index()
    기상지표_일별['강우일'] = 기상지표_일별['일강수량'] > 0
    기상지표_일별['평균기온'] = 기상지표_일별['평균기온'].round(1)

    # 고강도강우 3일 누적 여부
    기상지표_일별['고강도강우_발생여부_3일누적'] = (
        기상지표_일별['고강도강우_발생여부']
        .astype(int)
        .rolling(window=3, min_periods=1)
        .sum()
        .gt(0)
    )

    # 누적 강수량 (3/4/5일)
    기상지표_일별['누적강수_3일'] = (
        기상지표_일별['일강수량'].rolling(window=3, min_periods=1).sum()
    )
    기상지표_일별['누적강수_4일'] = (
        기상지표_일별['일강수량'].rolling(window=4, min_periods=1).sum()
    )
    기상지표_일별['누적강수_5일'] = (
        기상지표_일별['일강수량'].rolling(window=5, min_periods=1).sum()
    )

    # 연도, 월 추가
    기상지표_일별['일자'] = pd.to_datetime(기상지표_일별['일자'])
    기상지표_일별['연도'] = 기상지표_일별['일자'].dt.year
    기상지표_일별['월'] = 기상지표_일별['일자'].dt.month

    return 기상지표_일별


def summarize_rainfall(기상지표_일별: pd.DataFrame):
    """
    (수정) 시간단위 기상자료 → 일단위 지표(기상지표_일별)를 이용하여
    연월별 강수량 합계, 월별 평균 강수량 계산.

    기존: 일단위 기상자료(기상자료)를 직접 사용
    변경: build_daily_indicators 결과(기상지표_일별)를 사용
    """
    # 연월별 강수량 합계
    강수량_연월별_합계 = (
        기상지표_일별
        .groupby(['연도', '월'])['일강수량']
        .sum()
        .reset_index(name='월강수량')
    )

    # 연도별 '소계' 행 생성
    강수량_연월별_합계_소계 = (
        강수량_연월별_합계
        .groupby('연도')
        .apply(
            lambda df: pd.concat(
                [
                    df,
                    pd.DataFrame({
                        '연도': [df['연도'].iloc[0]],
                        '월': ['소계'],
                        '월강수량': [df['월강수량'].sum()],
                    })
                ],
                ignore_index=True,
            )
        )
        .reset_index(drop=True)
    )

    # 월별 강수량 평균
    강수량_월별_평균 = (
        강수량_연월별_합계
        .groupby('월')['월강수량']
        .mean()
        .round(1)
        .reset_index()
        .sort_values('월')
    )

    return 강수량_연월별_합계_소계, 강수량_월별_평균


def summarize_weather_yearly(기상지표_일별: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 기상 지표 요약.
    - 강수량 합계, 일평균강수량
    - 강우일수, 고강도일수, 고강도비율
    - 평균기온
    """
    기상지표_연도별 = (
        기상지표_일별.groupby('연도')
        .agg(
            강수량_합계=pd.NamedAgg(column='일강수량', aggfunc='sum'),
            강수량_일평균=pd.NamedAgg(
                column='일강수량',
                aggfunc=lambda x: round(x.mean(), 1),
            ),
            강우일수=pd.NamedAgg(column='강우일', aggfunc='sum'),
            고강도일수=pd.NamedAgg(
                column='고강도강우_발생여부',
                aggfunc='sum',
            ),
            평균기온=pd.NamedAgg(
                column='평균기온',
                aggfunc=lambda x: round(x.mean(), 1),
            ),
        )
    ).reset_index()

    기상지표_연도별['고강도비율'] = (
        (기상지표_연도별['고강도일수'] / 기상지표_연도별['강우일수']) * 100
    ).round(1)

    # 평균기온 열을 맨 뒤로 이동
    평균기온 = 기상지표_연도별.pop('평균기온')
    기상지표_연도별['평균기온'] = 평균기온

    return 기상지표_연도별
