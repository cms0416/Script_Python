import numpy as np
import pandas as pd


def add_season(df: pd.DataFrame, month_column: str = '월') -> pd.DataFrame:
    """
    월 정보를 바탕으로 '계절' 컬럼을 추가한다.
    계절: 봄(3~5), 여름(6~8), 가을(9~11), 겨울(그 외)
    """
    df = df.copy()
    conditions = [
        df[month_column].between(3, 5),
        df[month_column].between(6, 8),
        df[month_column].between(9, 11),
    ]
    seasons = ['봄', '여름', '가을']
    df['계절'] = np.select(conditions, seasons, default='겨울')
    df['계절'] = pd.Categorical(
        df['계절'],
        categories=['봄', '여름', '가을', '겨울'],
        ordered=True,
    )
    return df
