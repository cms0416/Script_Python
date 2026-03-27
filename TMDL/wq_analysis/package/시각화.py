import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.patches import Patch

from .config import ProjectConfig, PlotConfig

# 한글 폰트 설정(전역 설정)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 계절 색상 팔레트 정의
season_palette = {
    '봄': '#F8766D',
    '여름': '#00BA38',
    '가을': '#619CFF',
    '겨울': '#C77CFF'
}

# ---------------------------------------------------------------------------
# [공통] 그래프 저장 헬퍼 함수
# ---------------------------------------------------------------------------
def save_figure(cfg: ProjectConfig, title: str):
    """
    현재 활성화된 figure를 지정된 경로에 저장합니다.
    파일명 형식: {단위유역}_{수질항목}_{제목}.png
    저장 경로: C:/Coding/TMDL/수질분석/Output/Plot/
    """
    # 저장 경로 설정
    save_dir = cfg.path_plot_output
    
    # 디렉토리가 없으면 생성
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성 (자동으로 수질항목 이름 포함)
    # 예: 섬강A_TP_상관계수_히트맵.png
    filename = f"{cfg.단위유역}_{cfg.analyte.name}_{title}.png"
    filepath = save_dir / filename
    
    # 그래프 저장 (해상도 300dpi, 여백 최소화)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f" >> 그래프 저장 완료: {filename}")


# ---------------------------------------------------------------------------
# 1. 상관계수 히트맵
# ---------------------------------------------------------------------------

def plot_corr_heatmap(
    cfg: ProjectConfig,
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
):
    """
    상관계수 히트맵.
    - corr_matrix: 피어슨 상관계수 행렬
    - pval_matrix: p-value 행렬
    - cfg.analyte.name (예: 'TP', 'BOD') 기준으로 정렬
    """
    analyte_name = cfg.analyte.name

    # 대상 항목과의 상관계수 시리즈
    target_corr = corr_matrix[analyte_name].drop(analyte_name)

    # X축: 높은 상관 → 낮은 순서 (분석 항목은 맨 왼쪽)
    x_order = [analyte_name] + target_corr.sort_values(ascending=False).index.tolist()
    # Y축: 낮은 상관 → 높은 순서 (분석 항목은 맨 아래)
    y_order = target_corr.sort_values(ascending=True).index.tolist() + [analyte_name]

    # 필터된 상관계수 행렬
    corr_sorted = corr_matrix.loc[y_order, x_order]

    # 필터된 p-value 행렬
    pval_sorted = pval_matrix.loc[corr_sorted.index, corr_sorted.columns]
    
    # p-value 기준으로 annot 텍스트 마스킹
    annot_matrix = corr_sorted.round(2).astype(str)
    annot_matrix[pval_sorted > 0.05] = ""  # p-value가 0.05보다 크면 빈칸 처리

    # seaborn 스타일
    sns.set_theme(style="whitegrid", font='NanumGothic', font_scale=1.0)

    # 📊 히트맵 시각화
    plt.figure(figsize=(11, 10))
    ax = sns.heatmap(
        corr_sorted,
        annot=annot_matrix, fmt="",  # 유의하지 않은 값은 빈칸
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,        # 컬러바 범위 -1 ~ 1 고정
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"weight": "bold", "fontsize": 12}
    )

    # ✅ 컬러바 객체 가져오기 (가로 방향일 경우 마지막 axes)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13) # 컬러바 눈금 폰트 크기

    # ✅ 컬러바 제목 설정
    cbar.ax.set_xlabel("상관계수", fontsize=14, weight='bold', labelpad=10)
    cbar.ax.xaxis.set_label_position('top')  # 상단에 위치

    # ✅ 축라벨 설정
    plt.xticks(fontsize=13, weight='bold', rotation=45, ha='right')  # x축 라벨 45도 기울이기
    plt.yticks(fontsize=13, weight='bold')  # y축 라벨은 수평 유지

    # ✅ 그래프 여백 조정
    plt.tight_layout(pad=1.5) # 상하좌우 여백 자동 조정  
    plt.subplots_adjust(right=1.05)  # 우측 여백 조정)
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "상관계수_히트맵")
    
    plt.show()


# ---------------------------------------------------------------------------
# 1-2. 상관계수 히트맵(해당 항목만)
# ---------------------------------------------------------------------------

def plot_target_corr_heatmap(
    cfg,
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
):
    """
    특정 항목(cfg.analyte.name)에 대한 상관계수만 1줄로 시각화 (Strip Heatmap)
    """
    analyte_name = cfg.analyte.name
    
    # 1. 정렬 순서 결정 (타겟 변수와의 상관계수가 높은 순서대로)
    target_corr = corr_matrix[analyte_name].drop(analyte_name)
    x_order = [analyte_name] + target_corr.sort_values(ascending=False).index.tolist()
    
    # 2. 데이터 슬라이싱 (1행 DataFrame으로 추출)
    corr_row = corr_matrix.loc[[analyte_name], x_order]
    pval_row = pval_matrix.loc[[analyte_name], x_order]

    # 3. p-value 마스킹
    annot_row = corr_row.round(2).astype(str)
    annot_row[pval_row > 0.05] = ""

    # seaborn 스타일 (기존 유지)
    sns.set_theme(style="whitegrid", font='NanumGothic', font_scale=1.0)

    # 📊 히트맵 시각화
    plt.figure(figsize=(12, 3))
    
    ax = sns.heatmap(
        corr_row,
        annot=annot_row, fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,        # 컬러바 범위 -1 ~ 1 고정
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.2,      # 컬러바 길이
            "aspect": 2.5        # 컬러바 두께를 두껍게 (값이 작을수록 두꺼움)
        },
        annot_kws={"weight": "bold", "fontsize": 12}, # 폰트 스타일
        square=True          # 셀을 정사각형으로 강제(ax.set_aspect('equal')과 동일 효과)
    )

    # ✅ 컬러바 객체 설정 
    cbar = ax.collections[0].colorbar
    #cbar.ax.set_xlabel("상관계수", fontsize=10, weight='bold', labelpad=10)
    #cbar.ax.xaxis.set_label_position('top')
    #cbar.ax.tick_params(labelsize=10) # 컬러바 눈금 폰트 크기

    # ✅ 축 라벨 설정
    plt.xticks(fontsize=13, weight='bold', rotation=45, ha='right')
    plt.yticks(fontsize=13, weight='bold', rotation=0) # y축은 수평

    # Y축 라벨(인덱스 이름) 제거
    plt.ylabel("")

    # 그래프 저장(제목 설정)
    save_figure(cfg, "상관계수_히트맵_해당항목")

    plt.show()


# ---------------------------------------------------------------------------
# 2. 유량 / 수질 그래프
# ---------------------------------------------------------------------------

def plot_flow_vs_quality(
    cfg: ProjectConfig,
    총량측정망: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
):
    """
    유량백분율(%) – [오른쪽] 유량(bar), [왼쪽] 수질(scatter, 로그스케일) 복합 그래프.
    - 유량 축 범위: plot_cfg.flow_ylim
    - 수질 축 범위: plot_cfg.flow_quality_ylim 또는 cfg.analyte.default_ylim
    """
    analyte = cfg.analyte

    데이터 = 총량측정망.copy()
    # 계절 색상
    season_colors = 데이터['계절'].map(season_palette)

    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    ax2 = ax1.twinx()

    # 유량 막대
    bars = ax2.bar(
        데이터['유량백분율'], 데이터['유량'],
        width=0.3, color='steelblue', edgecolor='steelblue', alpha=1.0,
        label='유량'
    )
    if plot_cfg.flow_ylim is not None:
        ax2.set_ylim(*plot_cfg.flow_ylim)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(False)

    # 수질 점 (로그 스케일)
    scatter = ax1.scatter(
        데이터['유량백분율'], 데이터[analyte.col],
        c=season_colors, edgecolor='black', s=40,
        alpha=0.6, label='수질'
    )
    ax1.set_yscale("log")
    if plot_cfg.flow_quality_ylim is not None:
        ax1.set_ylim(*plot_cfg.flow_quality_ylim)
    
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.set_xticks(range(0, 101, 10))  # x축 눈금 설정
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

    # 목표수질 수평선
    ax1.axhline(
        y=목표수질,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label='목표수질'
    )

    # 유황구간 기준선 및 라벨
    for x in [0, 10, 40, 60, 90, 100]:
        ax1.axvline(x=x, linestyle='dashed', linewidth=1, color='gray')

    유황라벨 = {
        5: '홍수기', 25: '풍수기', 50: '평수기',
        75: '저수기', 95: '갈수기'
    }
    for x, label in 유황라벨.items():
        ax1.text(x, 0.95, label, ha='center', va='center',
                 fontsize=12, weight='bold', transform=ax1.get_xaxis_transform())

    # 축 레이블
    ax1.set_xlabel("유량 백분율(%)", fontsize=14, weight='bold')
    ax1.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')
    ax2.set_ylabel("유량(㎥/s)", fontsize=14, weight='bold')

    # 축 눈금
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_ticks_position('bottom')

    # zorder 조정
    ax1.set_zorder(ax2.get_zorder() + 10)
    ax1.patch.set_visible(False)

    # 축 테두리선 색상
    for spine in ax1.spines.values():
        spine.set_color('black')

    # 범례 구성 (수질, 목표수질, 유량, 계절)
    flow_patch = Patch(facecolor='steelblue', edgecolor='steelblue', alpha=0.8, label='유량')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor='white', label='수질', markersize=8),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
        flow_patch
    ] + [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor=color, alpha=0.6, label=label, markersize=8)
        for label, color in season_palette.items()
    ]
    # 수질 / 목표수질 / 유량 범례
    legend1 = ax1.legend(
        handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질, 강수량
        loc='upper center',
        ncol=3,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.25, 1.13),
        fontsize=11,
        handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )
    
    # 계절 범례
    legend2 = ax1.legend(
        handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
        loc='upper center',
        ncol=4,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.73, 1.13),
        fontsize=11,
        handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )
    
    # 첫 번째 범례를 그래프에 추가로 다시 등록
    ax1.add_artist(legend1)

    ax1.set_title("")
    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "유량_수질")
    
    plt.show()


# ---------------------------------------------------------------------------
# 3. 강수 / 수질 그래프
# ---------------------------------------------------------------------------

def plot_rain_vs_quality(
    cfg: ProjectConfig,
    총량측정망_기상: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
):
    """
    - [오른쪽] 강수량(bar), [왼쪽] 수질(scatter, 로그스케일).
    - rain_start_year 이후 연도만 필터링 가능.
    - 축 범위는 PlotConfig에서 설정.
    """
    analyte = cfg.analyte

    데이터 = 총량측정망_기상.copy()
    데이터['일자'] = pd.to_datetime(데이터['일자'])

    if plot_cfg.rain_start_year is not None:
        데이터 = 데이터[데이터['연도'] >= plot_cfg.rain_start_year]

    # 계절별 색상 팔레트 적용
    season_colors = 데이터['계절'].map(season_palette)

    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    ax2 = ax1.twinx()

    # 강수량 막대
    ax2.bar(
        데이터['일자'], 데이터['일강수량'],
        width=3, color='steelblue', edgecolor='steelblue',
        alpha=1.0, label='강수량'
    )
    if plot_cfg.rain_ylim is not None:
        ax2.set_ylim(*plot_cfg.rain_ylim)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))   # y축 레이블 포맷팅(x를 정수로 변환하고, 천 단위 쉼표 추가)
    ax2.grid(False)  # 보조 y축 격자선 제거 

    # 수질 점 (로그)
    scatter = ax1.scatter(
        데이터['일자'], 데이터[analyte.col],
        c=season_colors, edgecolor='black', s=40,
        label='수질', alpha=0.7
    )
    ax1.set_yscale("log")
    if plot_cfg.rain_quality_ylim is not None:
        ax1.set_ylim(*plot_cfg.rain_quality_ylim)
    
    # y축 로그 스케일 유지하되 눈금은 일반 숫자로
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

    # 목표수질 수평선
    ax1.axhline(
        y=목표수질,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label='목표수질'
    )
    # 축 레이블 제목 설정
    ax1.set_xlabel("연도", fontsize=14, weight='bold')
    ax1.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')
    ax2.set_ylabel("강수량(mm)", fontsize=14, weight='bold')

    # 축 눈금 설정
    ax1.tick_params(axis='x', labelsize=11, rotation=0)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    # x축 눈금 바깥쪽으로
    ax1.xaxis.set_ticks_position('bottom')

    # ax2의 zorder를 얻고, ax2의 zorder보다 큰 값을 ax1의 zorder로 지정
    # zorder가 낮을수록 먼저 그려지고, zorder가 높을수록 나중에 그려짐
    ax1.set_zorder(ax2.get_zorder() + 10)
    # ax1의 배경을 투명하게 만들어 ax1이 앞으로 배치되었을 때 ax2의 내용이 가려지지 않게 함
    ax1.patch.set_visible(False)

    # 축 테두리선 색상
    for spine in ax1.spines.values():
        spine.set_color('black')

    # 범례 (수질/목표수질/강수량 + 계절)
    rain_patch = Patch(facecolor='steelblue', edgecolor='steelblue', label='강수량')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor='white', label='수질', markersize=8),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
        rain_patch
    ] + [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor=color, alpha=0.7, label=label, markersize=8)
        for label, color in season_palette.items()
    ]

    # 수질 / 목표수질 / 강수량 범례
    legend1 = ax1.legend(
        handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질, 강수량
        loc='upper center',
        ncol=3,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.26, 1.13),
        fontsize=11,
        handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 계절 범례 (제목 포함)
    legend2 = ax1.legend(
        handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
        loc='upper center',
        ncol=4,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.73, 1.13),
        fontsize=11,
        handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 첫 번째 범례를 그래프에 추가로 다시 등록
    ax1.add_artist(legend1)

    # 축 테두리선 색상 변경
    for spine in ax1.spines.values():
        spine.set_color('black')

    # 상단 여백 확보
    ax1.set_title("")
    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "강수량_수질")
    
    plt.show()


# ---------------------------------------------------------------------------
# 4. 연도 별 박스플롯 (수질 + 연 유량합계)
# ---------------------------------------------------------------------------

def plot_yearly_box(
    cfg: ProjectConfig,
    총량측정망: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
):
    """
    연도별 수질 박스플롯 + 연 유량합계 막대 (보조축).
    """
    analyte = cfg.analyte

    # 데이터 준비
    데이터 = 총량측정망.copy()
    데이터 = 데이터[데이터[analyte.col] > 0]  # 로그 스케일이므로 0 이상만 사용

    # 계절별 색상 팔레트 적용
    season_colors = 데이터['계절'].map(season_palette)

    # 연도 순서 고정 (박스/스트립/막대 모두 동일 사용)
    years_order = sorted(데이터['연도'].unique())

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8.4, 5))

    # 박스플롯
    sns.boxplot(
        data=데이터,
        x='연도',
        y=analyte.col,
        ax=ax,
        order=years_order,
        showfliers=False,
        width=0.5,
        color='white',
        linewidth=1,
        linecolor='black'
    )

    # 연도별 jitter (계절 색)
    sns.stripplot(
        data=데이터,
        x='연도',
        y=analyte.col,
        order=years_order,
        hue='계절',
        palette=season_palette,
        alpha=0.6,
        size=6,
        jitter=0.2,  # 점들이 겹치지 않도록 약간의 무작위 이동
        dodge=True,  # 계절별로 점을 분리
        edgecolor='black',
        linewidth=1,
        ax=ax
    )

    # 목표수질 라인
    ax.axhline(y=목표수질, color='red', linestyle='--', linewidth=1.2, label='목표수질')

    # 로그 스케일
    ax.set_yscale("log")
    if plot_cfg.box_quality_ylim is not None:
        ax.set_ylim(*plot_cfg.box_quality_ylim)
    
    # y축 로그 스케일 유지하되 눈금은 일반 숫자로
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

    # x축 눈금 바깥쪽으로
    ax.xaxis.set_ticks_position('bottom')

    # 연도별 유량 합계 막대(보조축)
    ax2 = ax.twinx()
    yearly_flow = (
        총량측정망.groupby('연도')['유량']
        .sum()
        .reindex(years_order)
    )
    xpos = np.arange(len(years_order))
    ax2.bar(
        xpos, yearly_flow,
        width=0.6,
        alpha=0.3,
        color='steelblue',
        edgecolor='none'
    )

    if plot_cfg.box_flow_ylim is not None:
        ax2.set_ylim(*plot_cfg.box_flow_ylim)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(False)

    # 박스/점이 막대 위로 보이도록
    ax.set_zorder(2)
    ax.patch.set_alpha(0.0)

    # x축 눈금을 연도 라벨로 명시(혹시 matplotlib 버전에 따라 필요할 수 있음)
    ax.set_xticks(xpos)
    ax.set_xticklabels(years_order, rotation=0)

    # 축 라벨
    ax.set_xlabel("연도", fontsize=14, weight='bold')
    ax.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')
    ax2.set_ylabel("연 유량 합계", fontsize=14, weight='bold')

    # 범례
    flow_patch = Patch(facecolor='steelblue', alpha=0.6, label='유량')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor='white', label='수질', markersize=8),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질'),
        flow_patch
    ] + [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black',
                   markerfacecolor=color, alpha=0.6, label=label, markersize=8)
        for label, color in season_palette.items()
    ]

    # 수질 / 목표수질 범례
    legend1 = ax.legend(
        handles=legend_handles[:3],  # 세번째 까지(0~2): 수질, 목표수질
        loc='upper center',
        ncol=3,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.25, 1.13),  # 범례 위치 조정
        fontsize=12,
        handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 계절 범례
    legend2 = ax.legend(
        handles=legend_handles[3:],  # 네번째 부터: 계절(봄~겨울)
        loc='upper center',  # 범례 위치
        ncol=4,  # 범례 항목 수
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.73, 1.13),  # 범례 위치 조정
        fontsize=12,
        title_fontsize=12,
        handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 첫 번째 범례를 그래프에 추가로 다시 등록
    ax.add_artist(legend1)

    # 축 테두리선 색상 변경
    for spine in ax.spines.values():
        spine.set_color('black')

    ax.tick_params(labelsize=11)
    ax.set_title("")
    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "연도별_수질_박스플롯")
    
    plt.show()


# ---------------------------------------------------------------------------
# 5. 누적강수 3일 / 수질 산점도
# ---------------------------------------------------------------------------

def plot_cumrain_vs_quality(
    cfg: ProjectConfig,
    총량측정망: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
):
    """
    누적강수_3일 - 수질 산점도(로그 스케일).
    """
    analyte = cfg.analyte

    # 데이터 준비
    데이터 = 총량측정망.copy()

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8.4, 5))

    # 수질 점(로그 스케일)
    scatter = ax.scatter(
        데이터['누적강수_3일'], 데이터[analyte.col],
        c='white', edgecolor='black', s=40,
        alpha=0.6, label='수질'
    )
    ax.set_yscale("log")
    if plot_cfg.cumrain_quality_ylim is not None:
        ax.set_ylim(*plot_cfg.cumrain_quality_ylim)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)

    # 목표수질 선
    ax.axhline(y=목표수질, color='red', linestyle='dashed', 
               linewidth=1.2, label='목표수질')

    ax.set_xlabel("최근 3일 누적강수량(mm)", fontsize=14, weight='bold')
    ax.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')

    # 축 눈금 설정
    ax.tick_params(labelsize=11)
    # x축 눈금 바깥쪽으로
    ax.xaxis.set_ticks_position('bottom')
    # 축 테두리선 색상
    for spine in ax.spines.values():
        spine.set_color('black')

    # 📌 범례 구성(수질/목표수질)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
                   markerfacecolor='white', label='수질', markersize=8),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
    ]

    # 수질 / 목표수질 범례
    legend = ax.legend(
        handles=legend_handles,
        loc='upper center',  # 범례 위치
        ncol=2,  # 범례 항목 수
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.2, 1),  # 범례 위치 조정
        fontsize=12,
        handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "누적강수3일_수질_산점도")
    
    plt.show()


# ---------------------------------------------------------------------------
# 6. 강우 이벤트별 수질 영향 박스플롯(고강도강우_발생여부_3일누적)
# ---------------------------------------------------------------------------

def plot_rain_event_box(
    cfg: ProjectConfig,
    총량측정망: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
    flag_col: str = '고강도강우_발생여부_3일누적',
):
    """
    최근 3일 고강도 강우 발생 여부(True/False) 별 수질 박스플롯.
    - flag_col: True/False 값이 들어있는 컬럼명.
    """
    analyte = cfg.analyte

    plot_df = (
        총량측정망
        .dropna(subset=[analyte.col, flag_col])
        .query(f"{analyte.col} > 0")
        .copy()
    )

    # x축 레이블을 보기 쉽게 한글 라벨로 변환
    plot_df['최근3일_고강도발생'] = np.where(plot_df[flag_col], '발생', '미발생')

    sns.set_theme(style="whitegrid", font="NanumGothic", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(8.4, 5))

    palette = sns.color_palette("pastel")[:2]

    sns.boxplot(
        data=plot_df,
        x='최근3일_고강도발생',
        y=analyte.col,
        order=['미발생', '발생'],
        palette=palette,
        showfliers=True,
        width=0.6,
        ax=ax
    )

    # 목표수질
    ax.axhline(y=목표수질, linestyle='--', color='red', linewidth=1.2)

    ax.set_yscale("log")
    if plot_cfg.event_quality_ylim is not None:
        ax.set_ylim(*plot_cfg.event_quality_ylim)
    
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)

    ax.set_xlabel("최근 3일 고강도 강우 발생 여부", fontsize=14, weight='bold')
    ax.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')

    # 축 눈금 위치
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # 축 테두리선 색상
    for spine in ax.spines.values():
        spine.set_color('black')

    # 범례 구성: '미발생', '발생' + '목표수질'
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=palette[0], edgecolor='black', label='미발생'),
        Patch(facecolor=palette[1], edgecolor='black', label='발생'),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        frameon=True,
        edgecolor='black',
        bbox_to_anchor=(0.5, 1.12),
        fontsize=11,
        handletextpad=0.5,
        columnspacing=1.2
    )

    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "고강도강우_수질_박스플롯")
    
    plt.show()


# ---------------------------------------------------------------------------
# 7. 유역 내 수질측정망 박스플롯
# ---------------------------------------------------------------------------

def plot_station_box(
    cfg: ProjectConfig,
    수질측정망: pd.DataFrame,
    목표수질: float,
    plot_cfg: PlotConfig,
):
    """
    유역 내 수질측정망 박스플롯 (측정소별, 계절색).
    """
    analyte = cfg.analyte
    데이터 = 수질측정망.copy()

    # x축 순서: 지정값이 있으면 사용, 없으면 자동 정렬
    if plot_cfg.station_order:
        order = plot_cfg.station_order
    else:
        order = sorted(데이터['측정소명'].unique())

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # 박스플롯
    sns.boxplot(
        data=데이터,
        x='측정소명',
        y=analyte.col,
        order=order,
        ax=ax,
        showfliers=False,
        width=0.5,
        color='white',
        linewidth=1,
        linecolor='black'
    )

    # Jittered scatter
    sns.stripplot(
        data=데이터,
        x='측정소명',
        y=analyte.col,
        order=order,
        ax=ax,
        hue='계절',  
        palette=season_palette,
        alpha=0.5,
        size=6,
        jitter=0.0,  # 점들이 겹치지 않도록 약간의 무작위 이동
        dodge=True,  # 계절별로 점을 분리
        edgecolor='black',
        linewidth=1  # 점 테두리 두께
    )

    # 목표수질 수평선
    ax.axhline(
        y=목표수질,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label='목표수질'
    )

    # y축 로그 스케일
    ax.set_yscale("log")
    if plot_cfg.station_quality_ylim is not None:
        ax.set_ylim(*plot_cfg.station_quality_ylim)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # 격자선 설정

    ax.set_xlabel("측정지점", fontsize=14, weight='bold')
    ax.set_ylabel(f"{analyte.label} ({analyte.unit})", fontsize=14, weight='bold')

    # 축 눈금 설정
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # 축 테두리선 색상 변경
    for spine in ax.spines.values():
        spine.set_color('black')

    # 범례 구성(수질/목표수질 + 계절)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
                   markerfacecolor='white', label='수질', markersize=8),
        plt.Line2D([0], [0], linestyle='--', color='red', label='목표수질')
    ] + [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='black', 
                   markerfacecolor=color, alpha=0.6, label=label, markersize=8)
        for label, color in season_palette.items()
    ]

    # 수질 / 목표수질 범례
    legend1 = ax.legend(
        handles=legend_handles[:2],  # 두번째 까지(0~1): 수질, 목표수질
        loc='upper center',
        ncol=2,
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.25, 1.13),  # 범례 위치 조정
        fontsize=12,
        handletextpad=0.3,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 계절 범례
    legend2 = ax.legend(
        handles=legend_handles[2:],  # 세번째 부터: 계절(봄~겨울)
        loc='upper center',  # 범례 위치
        ncol=4,  # 범례 항목 수
        frameon=True,  # 범례 테두리 표시
        edgecolor='black',  # 범례 테두리 색상
        bbox_to_anchor=(0.65, 1.13),  # 범례 위치 조정
        fontsize=12,
        title_fontsize=12,
        handletextpad=0.1,  # 범례 마커와 텍스트 간 간격
        columnspacing=1   # 범례 항목(열) 간 간격
    )

    # 첫 번째 범례를 그래프에 추가로 다시 등록
    ax.add_artist(legend1)

    # 기타 스타일
    ax.tick_params(labelsize=11)  # x축 + y축 눈금 크기
    ax.set_title("")  # 제목 없음

    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    
    # 그래프 저장(제목 설정)
    save_figure(cfg, "유역내_수질측정망_박스플롯")
    
    plt.show()
