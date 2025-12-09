import pandas as pd


def compute_by_flow(총량측정망: pd.DataFrame) -> pd.DataFrame:
    """
    유황구간별 달성률 계산.
    원본 스크립트의 '달성률_유황구간별'.
    """
    # 유황구간 순서 지정
    유황구간_순서 = ['갈수기', '저수기', '평수기', '풍수기', '홍수기']

    # 유황구간을 순서 있는 범주형(factor)으로 변환
    총량측정망 = 총량측정망.copy()
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
    달성률_유황구간별['달성률'] = (
        달성률_유황구간별['달성'] / 달성률_유황구간별['총계'] * 100
    ).round(1)

    # 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
    총계_열 = 달성률_유황구간별.pop('총계')
    달성률_유황구간별.insert(
        달성률_유황구간별.columns.get_loc('달성'),
        '총계',
        총계_열
    )

    # 유황구간 순서 적용 정렬
    달성률_유황구간별 = 달성률_유황구간별.sort_values('유황구간').reset_index(drop=True)
    return 달성률_유황구간별


def compute_by_season(총량측정망: pd.DataFrame) -> pd.DataFrame:
    """
    계절별 달성률.
    """
    달성률_계절별 = (
        총량측정망.groupby(['계절', '달성여부'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    달성률_계절별['총계'] = 달성률_계절별['달성'] + 달성률_계절별['초과']
    달성률_계절별['달성률'] = (
        달성률_계절별['달성'] / 달성률_계절별['총계'] * 100
    ).round(1)

    # 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
    총계_열 = 달성률_계절별.pop('총계')
    달성률_계절별.insert(
        달성률_계절별.columns.get_loc('달성'),
        '총계',
        총계_열
    )
    return 달성률_계절별


def compute_by_month(총량측정망: pd.DataFrame, 강수량_월별_평균: pd.DataFrame) -> pd.DataFrame:
    """
    월별 달성률 + 월별 평균 강수량 병합.
    """
    달성률_월별 = (
        총량측정망.groupby(['월', '달성여부'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    달성률_월별['총계'] = 달성률_월별['달성'] + 달성률_월별['초과']
    달성률_월별['달성률'] = (
        달성률_월별['달성'] / 달성률_월별['총계'] * 100
    ).round(1)

    # 월별 강수량 평균과 병합
    달성률_월별 = pd.merge(
        달성률_월별,
        강수량_월별_평균,
        on='월',
        how='left'
    )

    # 열 순서 정리: '총계' 열을 꺼내고 '달성' 열의 위치 찾아서 그 위치에 삽입
    총계_열 = 달성률_월별.pop('총계')
    달성률_월별.insert(
        달성률_월별.columns.get_loc('달성'),
        '총계',
        총계_열
    )
    return 달성률_월별


def compute_by_year(총량측정망: pd.DataFrame, analyte_col: str, analyte_name: str) -> pd.DataFrame:
    """
    연도별 평균 수질, 유량합계, 달성/초과/총계/달성률 정리.
    analyte_col: 'TP', 'BOD' 등 컬럼명
    analyte_name: 'TP', 'BOD' 등 이름(평균 컬럼명 생성에 사용)
    """
    달성률_연도별 = (
        총량측정망.groupby('연도')
        .agg({
            analyte_col: lambda x: round(x.mean(), 3),
            '유량': 'sum',
            '달성여부': lambda x: (x == '달성').sum(),
        })
        .rename(columns={
            analyte_col: f'{analyte_name}_평균',
            '유량': '유량_합계',
            '달성여부': '달성'
        })
        .reset_index()
    )

    달성률_연도별['초과'] = (
        총량측정망.groupby('연도')['달성여부']
        .apply(lambda x: (x == '초과').sum())
        .values
    )
    달성률_연도별['총계'] = 달성률_연도별['달성'] + 달성률_연도별['초과']
    달성률_연도별['달성률'] = (
        달성률_연도별['달성'] / 달성률_연도별['총계'] * 100
    ).round(1)

    # 열 순서 재정렬
    cols = ['연도', '초과', '달성', '총계', '달성률', f'{analyte_name}_평균', '유량_합계']
    달성률_연도별 = 달성률_연도별[cols]
    return 달성률_연도별


def merge_with_weather(달성률_연도별: pd.DataFrame, 기상지표_연도별: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 달성률과 기상지표(강수량, 고강도일수 등) 병합.
    """
    달성률_강우강도_연도별 = pd.merge(
        달성률_연도별,
        기상지표_연도별,
        on='연도',
        how='left'
    )
    return 달성률_강우강도_연도별
