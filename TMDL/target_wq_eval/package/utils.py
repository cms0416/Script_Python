import numpy as np

def round_half_up(n, decimals=0):
    """
    반올림 함수(사사오입)
    """
    multiplier = 10 ** decimals
    return np.floor(n * multiplier + 0.5) / multiplier