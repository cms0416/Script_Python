# -*- coding: utf-8 -*-
"""
섬강B · BOD 분석 실행 스크립트
실행 예) E:\Coding> py -m TMDL.wq_analysis.run_seomgangB_BOD
"""

from pathlib import Path
from TMDL.wq_analysis.package import (
    ProjectConfig,
    AnalyteSpec,
    PlotOverrides,
    run_full_pipeline,
)

ROOT = Path("E:/Coding").resolve()
TMDL_DIR = ROOT / "TMDL"
WQ_DIR = TMDL_DIR / "수질분석"
OUT_DIR = WQ_DIR / "Output"
PLOT_DIR = OUT_DIR / "Plot"

PATH_TMDL_MASTER = WQ_DIR / "RAW DATA" / "tmdl_master.xlsx"
PATH_TARGET_QUALITY = WQ_DIR / "RAW DATA" / "target_quality.xlsx"
PATH_WEATHER_DAILY = WQ_DIR / "RAW DATA" / "weather_daily.csv"
PATH_WEATHER_MONTHLY = WQ_DIR / "RAW DATA" / "weather_monthly.csv"

def make_config() -> ProjectConfig:
    analyte = AnalyteSpec(
        col="BOD",
        label="BOD",
        unit="mg/L",
        use_logscale=False,   # BOD는 선형축 선호
        corr_candidates=[
            "유량", "수온", "SS", "TOC", "TP",
            "강수량", "3일누적강수량", "7일누적강수량"
        ],
    )

    plot_overrides = PlotOverrides(
        flow_percent_ticks=(0, 10, 40, 60, 90, 100),
        season_order=("봄", "여름", "가을", "겨울"),
        x_year_min=None,
        x_year_max=None,
        y_limits=(None, None),
        scatter_size=30,
        scatter_alpha=0.7,
        bar_width=0.8,
        season_palette=None,
    )

    cfg = ProjectConfig(
        watershed_code="섬강B",
        watershed_name="섬강B",
        data_dir=WQ_DIR,
        output_dir=OUT_DIR,
        plot_dir=PLOT_DIR,
        path_tmdl_master=PATH_TMDL_MASTER,
        path_target_quality=PATH_TARGET_QUALITY,
        path_weather_daily=PATH_WEATHER_DAILY,
        path_weather_monthly=PATH_WEATHER_MONTHLY,
        analyte=analyte,
        plot=plot_overrides,
    )
    return cfg

def main():
    cfg = make_config()
    run_full_pipeline(cfg)

if __name__ == "__main__":
    main()
