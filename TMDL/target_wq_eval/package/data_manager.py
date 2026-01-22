# 데이터를 불러오고 전처리하는 모듈

import pandas as pd
from pathlib import Path
from . import config

def load_and_merge_raw_data(data_path: Path):
    """
    물환경정보시스템 엑셀 파일들을 읽어 병합하고 전처리하는 함수
    """
    # 1. 파일 목록 가져오기
    target_dir = data_path / "수질분석/물환경정보시스템"
    files = list(target_dir.glob("*.xlsx"))
    
    # 2. 데이터 읽기 및 병합
    df_list = []
    for f in files:
        temp_df = pd.read_excel(f, skiprows=1, header=0)
        df_list.append(temp_df)
    
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # 3. 한탄A 과거 자료 읽기
    hantan_path = target_dir / "한탄A_지점변경전/총량측정망_한탄A_0720.xlsx"
    hantan_df = pd.read_excel(hantan_path)
    
    return raw_df, hantan_df

def process_data(raw_df, hantan_df):
    """
    데이터 컬럼명 변경, 날짜 변환, 한탄A 데이터 교체 등 전처리 수행
    """
    # 1. 컬럼명 변경
    # config.py의 COL_MAPPING 리스트 사용
    raw_df.columns = config.COL_MAPPING
    
    # 2. 컬럼 순서 변경
    df = raw_df[config.COL_ORDER].copy()
    
    # 3. 날짜 형식 변경 및 연도/월 추가
    # 문자열 '.'을 '-'로 변경 후 datetime으로 변환
    df['일자'] = df['일자'].astype(str).str.replace('.', '-')
    df['일자'] = pd.to_datetime(df['일자'])
    df['연도'] = df['일자'].dt.year
    df['월'] = df['일자'].dt.month
    
    # 4. 데이터 필터링 및 한탄A 지점 변경전('21년 이전) 자료 삭제
    mask_remove_hantan = (df['총량지점명'] == '한탄A') & (df['연도'] < 2021)
    df = df[~mask_remove_hantan]
    df = df[df['연도'] > 2006]
    
    # 5. 한탄A 과거자료 병합
    # 한탄A 지점 변경전('21년 이전) 자료 기존 측정자료로 교체
    final_df = pd.concat([df, hantan_df], ignore_index=True)
    
    # 6. 정렬
    final_df = final_df.sort_values(by=['총량지점명', '일자'])
    
    return final_df

def filter_by_station(df, stations=None):
    """
    지점명 필터링 및 순서 적용
    """
    if stations is None:
        target_stations = config.STATION_ORDER
    else:
        target_stations = stations
        
    filtered_df = df[df['총량지점명'].isin(target_stations)].copy()
    
    # Categorical Type으로 변환하여 정렬 순서 지정
    filtered_df['총량지점명'] = pd.Categorical(
        filtered_df['총량지점명'], 
        categories=config.STATION_ORDER, 
        ordered=True
    )
    
    return filtered_df.sort_values(by=['총량지점명', '일자'])