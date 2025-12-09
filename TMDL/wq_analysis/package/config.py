from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class AnalyteSpec:
    """
    수질 항목별 메타데이터 정의.
    """
    name: str              # 'TP', 'BOD' 등
    label: str             # 그래프/표에 쓸 표시 이름 (예: 'T-P')
    unit: str              # 단위 (예: 'mg/L')
    col: str               # 원 데이터 컬럼명 (예: 'TP')
    target_col: str        # 목표수질 엑셀 컬럼명
    load_col: str          # 부하량 컬럼명
    target_load_col: str   # 목표부하량 컬럼명
    max_valid: Optional[float] = None   # 상한값 필터(없으면 None)
    #default_ylim: Optional[Tuple[float, float]] = None

    @classmethod
    def TP(cls) -> "AnalyteSpec":
        return cls(
            name='TP',
            label='T-P',
            unit='mg/L',
            col='TP',
            target_col='TP_목표수질',
            load_col='측정부하량_TP',
            target_load_col='목표부하량_TP',
            max_valid=None,
            #default_ylim=(0.0025, 0.3),
        )

    @classmethod
    def BOD(cls) -> "AnalyteSpec":
        return cls(
            name='BOD',
            label='BOD',
            unit='mg/L',
            col='BOD',
            target_col='BOD_목표수질',
            load_col='측정부하량_BOD',
            target_load_col='목표부하량_BOD',
            max_valid=None,
            #default_ylim=(0.0009, 1),
        )

    @classmethod
    def from_name(cls, name: str) -> "AnalyteSpec":
        key = name.strip().upper()
        if key == "TP":
            return cls.TP()
        if key == "BOD":
            return cls.BOD()
        # 필요 시 COD, TOC, TN 등 추가
        raise ValueError(f"지원하지 않는 수질항목: {name}")


@dataclass
class ProjectConfig:
    """
    유역별 분석 공통 설정.
    """
    단위유역: str
    기상대: str
    시작연도: int
    종료연도: int
    수질항목: str   # "TP", "BOD" 등 문자열로 설정

    @property
    def analyte(self) -> AnalyteSpec:
        return AnalyteSpec.from_name(self.수질항목)

    #  공통 상위 경로 정의(이 부분만 고치면 모든 경로가 일괄 변경됨)
    @property
    def base_path(self) -> Path:
        """분석 데이터가 위치한 최상위 루트 디렉토리"""
        return Path("E:/Coding/TMDL/수질분석")

    @property
    def path_기상_일(self) -> Path:
        return self.base_path / "기상자료" / "기상자료_강수량_기온.xlsx"

    @property
    def path_기상_시(self) -> Path:
        return self.base_path / "기상자료" / "기상자료_강수량_기온_시단위.xlsx"

    @property
    def path_목표수질(self) -> Path:
        return self.base_path / "목표수질.xlsx"

    @property
    def path_총량측정망(self) -> Path:
        return self.base_path / "총량측정망_전체_2007_2025.xlsx"

    @property
    def path_수질측정망(self) -> Path:
        # 예: 섬강A → E:/Coding/TMDL/수질분석/수질측정망/섬강A_수질측정망_2014_2024.xlsx
        return self.base_path / "수질측정망" / f"{self.단위유역}_수질측정망_2014_2024.xlsx"

    @property
    def path_output(self) -> Path:
        # E:/Coding/TMDL/수질분석/Output/수질분석_{단위유역}_파이썬.xlsx
        return self.base_path / "Output" / f"수질분석_{self.단위유역}_파이썬.xlsx"

    @property
    def path_corr_output(self) -> Path:
        # TP의 경우: E:/Coding/TMDL/수질분석/Output/{단위유역}_TP_상관계수.xlsx
        return self.base_path / "Output" / f"{self.단위유역}_{self.analyte.name}_상관계수.xlsx"

    @property
    def path_plot_output(self) -> Path:
        """그래프가 저장될 기본 폴더 경로만 정의"""
        return self.base_path / "Output" / "Plot"

@dataclass
class PlotConfig:
    """
    그래프 관련 설정(축 범위, 측정소 순서 등).
    유역·수질항목별로 실행파일에서 조정.
    """

    # 1) 유량 / 수질 그래프
    flow_quality_ylim: Optional[Tuple[float, float]] = None  # 수질 축
    flow_ylim: Optional[Tuple[float, float]] = None          # 유량 축

    # 2) 강수 / 수질 그래프
    rain_quality_ylim: Optional[Tuple[float, float]] = None  # 수질 축
    rain_ylim: Optional[Tuple[float, float]] = None          # 강수량 축
    rain_start_year: Optional[int] = None                    # 예: 2020년 이후만 표시

    # 3) 연도별 박스플롯
    box_quality_ylim: Optional[Tuple[float, float]] = None   # 수질 축
    box_flow_ylim: Optional[Tuple[float, float]] = None      # 연 유량 합계 축

    # 4) 누적강수 / 수질, 강우 이벤트 박스
    cumrain_quality_ylim: Optional[Tuple[float, float]] = None
    event_quality_ylim: Optional[Tuple[float, float]] = None

    # 5) 유역 내 측정소 박스플롯
    station_quality_ylim: Optional[Tuple[float, float]] = None   # 수질 축
    station_order: Optional[List[str]] = None                # None이면 자동 정렬
