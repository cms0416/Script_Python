import pandas as pd
from scipy.stats import pearsonr


def compute_corr(df: pd.DataFrame, cols: list[str]):
    """
    상관계수 및 p-value 행렬 계산.
    원본 스크립트의 corr_matrix, pval_matrix 계산 부분.
    """
    # NaN 제거
    df_corr = df[cols].dropna()

    # 상관계수 & p-value 계산
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
                pval_matrix.loc[col1, col2] = 0.0
            else:
                corr, pval = pearsonr(df_corr[col1], df_corr[col2])
                corr_matrix.loc[col1, col2] = corr
                pval_matrix.loc[col1, col2] = pval

    return corr_matrix, pval_matrix
