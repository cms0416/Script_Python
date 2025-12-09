# %% ---------------------------------------------------------------------------
import pandas as pd
from .config import WQ_DIR
from .season import add_season

# %% ---------------------------------------------------------------------------
def load_weather_data(met_station: str):
    """기상대 명칭(부분일치)로 일/시단위 기상자료 로드."""
    daily_path = WQ_DIR / "기상자료" / "기상자료_강수량_기온.xlsx"
    hourly_path = WQ_DIR / "기상자료" / "기상자료_강수량_기온_시단위.xlsx"

    # 일단위
    daily = (
        pd.read_excel(daily_path)
        .query("지점명.str.contains(@met_station)", engine="python")
        .drop(columns=["지점", "지점명"])
        .rename(columns={
            "일시": "일자",
            "평균기온(°C)": "평균기온",
            "일강수량(mm)": "일강수량",
        })
    )
    daily["일자"] = pd.to_datetime(daily["일자"])
    daily["연도"] = daily["일자"].dt.year
    daily["월"] = daily["일자"].dt.month

    # 시단위
    hourly = (
        pd.read_excel(hourly_path)
        .query("지점명.str.contains(@met_station)", engine="python")
        .drop(columns=["지점", "지점명"])
        .rename(columns={
            "기온(°C)": "기온",
            "강수량(mm)": "강수량",
        })
    )
    hourly["일시"] = pd.to_datetime(hourly["일시"])
    hourly["일자"] = hourly["일시"].dt.date
    hourly["연도"] = hourly["일시"].dt.year
    hourly["월"] = hourly["일시"].dt.month

    return daily, hourly

# %% ---------------------------------------------------------------------------
def build_weather_indicators(daily: pd.DataFrame, hourly: pd.DataFrame):
    """연월별, 월별, 일별, 연도별 기상 지표 생성."""
    # 연월별 합계
    ym_rain = (
        daily.groupby(["연도", "월"])["일강수량"]
        .sum()
        .reset_index(name="월강수량")
    )

    # 월별 평균 강수량
    monthly_avg = (
        ym_rain.groupby("월")["월강수량"]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("월")
    )

    # 시단위 → 일별 지표
    hourly = hourly.dropna(subset=["강수량"])
    daily_ind = (
        hourly.groupby("일자")
        .agg(
            일강수량=("강수량", "sum"),
            최대시간강우량=("강수량", "max"),
            고강도강우_발생여부=("강수량", lambda x: (x >= 20).any()),
            매우고강도강우_발생여부=("강수량", lambda x: (x >= 30).any()),
            평균기온=("기온", "mean"),
        )
        .reset_index()
    )
    daily_ind["강우일"] = daily_ind["일강수량"] > 0
    daily_ind["평균기온"] = daily_ind["평균기온"].round(1)

    # 3/4/5일 누적 및 고강도 3일 누적 발생여부
    daily_ind["고강도강우_발생여부_3일누적"] = (
        daily_ind["고강도강우_발생여부"]
        .astype(int)
        .rolling(window=3, min_periods=1)
        .sum()
        .gt(0)
    )
    for win in (3, 4, 5):
        daily_ind[f"누적강수_{win}일"] = (
            daily_ind["일강수량"].rolling(window=win, min_periods=1).sum()
        )

    # 날짜 컬럼 재정비
    daily_ind["일자"] = pd.to_datetime(daily_ind["일자"])
    daily_ind["연도"] = daily_ind["일자"].dt.year
    daily_ind["월"] = daily_ind["일자"].dt.month

    # 연도별 요약
    yearly = (
        daily_ind.groupby("연도")
        .agg(
            강수량_합계=("일강수량", "sum"),
            강수량_일평균=("일강수량", lambda x: round(x.mean(), 1)),
            강우일수=("강우일", "sum"),
            고강도일수=("고강도강우_발생여부", "sum"),
            평균기온=("평균기온", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
    )
    yearly["고강도비율"] = (
        yearly["고강도일수"] / yearly["강우일수"] * 100
    ).round(1)

    return {
        "강수량_연월별_합계": ym_rain,
        "강수량_월별_평균": monthly_avg,
        "기상지표_일별": daily_ind,
        "기상지표_연도별": yearly,
    }
