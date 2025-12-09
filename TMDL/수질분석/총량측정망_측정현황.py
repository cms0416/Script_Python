# %% 라이브러리 로드
import os
import glob
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# %% 폰트 설정
font_path = Path("C:/Users/TG/AppData/Local/Microsoft/Windows/Fonts/NotoSansKR-Regular.otf")
if not font_path.exists():
    raise FileNotFoundError("Noto Sans KR font not found. Please download and provide the correct path.")

font_prop = font_manager.FontProperties(fname=str(font_path))
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly


# %% 데이터 불러오기 

# Define the path to the Excel files
data_path = Path("E:/Coding/TMDL/수질분석/4대강수계")
excel_files = glob.glob(str(data_path / "*.xls"))

# Function to extract watershed name from filename
def extract_watershed(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r'([가-힣]+)', basename)
    return match.group(1) if match else 'Unknown'

# Read and concatenate all Excel files
data_frames = []
for file in excel_files:
    df = pd.read_excel(file, skiprows=1)
    df['수계'] = extract_watershed(file)
    data_frames.append(df)

obs = pd.concat(data_frames, ignore_index=True)

# Drop unnecessary columns (4 to 7, 9 to 11, 13)
columns_to_drop = obs.columns[[3, 4, 5, 6, 8, 9, 10, 12]]
obs = obs.drop(columns=columns_to_drop)

# %%  데이터 정리  
# Rename columns
obs.columns = ["수계", "총량지점명", "일자", "BOD", "TP", "유량"]

# Convert 일자 to datetime and extract 연도, 월, 계절
obs['일자'] = pd.to_datetime(obs['일자'].str.replace('.', '-'), format='%Y-%m-%d')
obs['연도'] = obs['일자'].dt.year
obs['월'] = obs['일자'].dt.month

# Define 계절 based on 월
def get_season(month):
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 8:
        return "여름"
    elif 9 <= month <= 11:
        return "가을"
    else:
        return "겨울"

obs['계절'] = obs['월'].apply(get_season)
obs['계절'] = pd.Categorical(obs['계절'], categories=["봄", "여름", "가을", "겨울"], ordered=True)
obs['횟수'] = 1

# Filter out rows where TP is NaN
obs_1 = obs.dropna(subset=['TP'])

# ##############################################################
# ####################  계절별 평균 계산  ########################
# ##############################################################

# Group by 수계, 총량지점명, 연도, 계절 and sum 횟수
seasonal_sum = obs_1.groupby(['수계', '총량지점명', '연도', '계절'], as_index=False)['횟수'].sum()

# Then group by 수계 and 계절 to calculate mean 횟수
seasonal_avg = seasonal_sum.groupby(['수계', '계절'], as_index=False)['횟수'].mean().round(2)

# Pivot to wide format
seasonal_avg_wide = seasonal_avg.pivot(index='계절', columns='수계', values='횟수').reset_index()

# ##############################################################
# ####################  월별 평균 계산  ##########################
# ##############################################################

# Group by 수계, 총량지점명, 연도, 월 and sum 횟수
monthly_sum = obs_1.groupby(['수계', '총량지점명', '연도', '월'], as_index=False)['횟수'].sum()

# Then group by 수계 and 월 to calculate mean 횟수
monthly_avg = monthly_sum.groupby(['수계', '월'], as_index=False)['횟수'].mean().round(2)

# Pivot to wide format
monthly_avg_wide = monthly_avg.pivot(index='월', columns='수계', values='횟수').reset_index()

# ##############################################################
# ####################  월별 횟수 비율  ##########################
# ##############################################################

# Group by 수계 and 월, then sum 횟수
monthly_count = obs_1.groupby(['수계', '월'], as_index=False)['횟수'].sum()

# Pivot to wide format
monthly_count_wide = monthly_count.pivot(index='월', columns='수계', values='횟수').fillna(0)

# Calculate percentages per column
monthly_percentage = monthly_count_wide.div(monthly_count_wide.sum()).multiply(100).round(2)

# Add percentage signs
monthly_percentage_str = monthly_percentage.astype(str) + '%'

# ##############################################################
# ####################  한강수계 정리  ##########################
# ##############################################################

# Read 한강 데이터
han_data = pd.read_excel("수질분석/총량측정망_2007_2024.xlsx")

# Select required columns
han = han_data[['총량지점명', '일자', 'BOD', 'TP', '유량', '연도', '월']]

# Filter out 연도 2024
han = han[han['연도'] != 2024]

# Define 계절
han['계절'] = han['월'].apply(get_season)
han['계절'] = pd.Categorical(han['계절'], categories=["봄", "여름", "가을", "겨울"], ordered=True)
han['횟수'] = 1

# Filter out rows where TP is NaN
han = han.dropna(subset=['TP'])

# Group by 총량지점명, 연도, 계절 and sum 횟수
han_grouped = han.groupby(['총량지점명', '연도', '계절'], as_index=False)['횟수'].sum()

# Then group by 연도 and 계절 to calculate mean 횟수
han_avg = han_grouped.groupby(['연도', '계절'], as_index=False)['횟수'].mean().round(2)

# ##############################################################
# ####################  그래프 생성  ############################
# ##############################################################

# Create a PDF to save plots
pdf_path = "E:/Coding/TMDL/수질분석/4대강수계/총량측정망현황_그래프(8.4x4).pdf"
with PdfPages(pdf_path) as pdf:
    sns.set(style="whitegrid", font=font_prop.get_name(), font_scale=1.2)
    
    # 1. 계절별 측정 횟수 평균
    plt.figure(figsize=(8.4, 4))
    sns.barplot(data=seasonal_avg, x='계절', y='횟수', hue='수계', dodge=True)
    
    # Add labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    plt.xlabel("계절", fontweight='bold')
    plt.ylabel("횟수", fontweight='bold')
    plt.title("계절별 측정 횟수 평균", fontsize=14, fontweight='bold')
    plt.legend(title='수계')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # 2. 월별 측정 횟수 평균
    plt.figure(figsize=(8.4, 4))
    sns.barplot(data=monthly_avg, x='월', y='횟수', hue='수계', dodge=True)
    
    # Add labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    plt.xlabel("월", fontweight='bold')
    plt.ylabel("횟수", fontweight='bold')
    plt.title("월별 측정 횟수 평균", fontsize=14, fontweight='bold')
    plt.legend(title='수계')
    plt.xticks(ticks=range(0,12), labels=range(1,13))
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # 3. 한강 수계 연도별 겨울 측정 횟수
    han_winter = han_avg[han_avg['계절'] == "겨울"].copy()
    han_winter['연도'] = han_winter['연도'].astype(str).str.slice(-2).str.replace('', "'", 1)
    
    plt.figure(figsize=(8.4, 4))
    sns.barplot(data=han_winter, x='연도', y='횟수', color="deepskyblue")
    
    # Add labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    plt.xlabel("연도", fontweight='bold')
    plt.ylabel("횟수", fontweight='bold')
    plt.title("한강 수계 연도별 겨울 측정 횟수", fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"Graphs have been saved to {pdf_path}")
