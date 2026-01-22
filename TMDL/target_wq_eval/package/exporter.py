# 데이터 엑셀 저장기능 모듈

import pandas as pd

def save_formatted_excel(output_path, data_dict):
    """
    데이터프레임들을 지정된 서식으로 엑셀 파일에 저장하는 함수
    
    Args:
        output_path (Path): 저장할 파일 경로
        data_dict (dict): { '시트명': DataFrame } 형태의 딕셔너리 (순서대로 저장됨)
    """
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        
        # 딕셔너리의 순서대로 시트 생성
        for sheet_name, df in data_dict.items():
            # 1. 데이터 준비(복사본 생성 후 날짜형식 변환)
            temp_df = df.copy()
            
            if '일자' in temp_df.columns:
                temp_df['일자'] = temp_df['일자'].dt.date
            
            # 데이터 쓰기
            temp_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 2. 워크북 및 워크시트 객체 확보
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # 화면 눈금선 숨기기 (흰색 배경을 더 깔끔하게 보이게 함)
            worksheet.hide_gridlines(2) 
            
            # 3. 서식 정의
            # 배경색 흰색('#FFFFFF') 추가
            base_props = {
                'align': 'center', 
                'valign': 'vcenter', 
                'bg_color': '#FFFFFF' 
            }
            
            # 테두리 스타일(검은 실선)
            border_format = workbook.add_format({'border': 1})
            
            # 헤더 스타일(회색 배경, 굵게)
            header_format = workbook.add_format({
                'align': 'center', 'valign': 'vcenter', 
                'bold': True, 'border': 1, 'bg_color': '#F2F2F2'
            })
            
            # 데이터 포맷별 스타일 생성
            fmt_default = workbook.add_format(base_props)
            fmt_float_1 = workbook.add_format({**base_props, 'num_format': '0.0'})
            fmt_float_3 = workbook.add_format({**base_props, 'num_format': '0.000'})
            
            # 4. 열 너비 및 데이터 서식 적용
            max_row = len(temp_df)
            max_col = len(temp_df.columns) - 1
            
            for col_idx, col_name in enumerate(temp_df.columns):
                # (1) 열 너비 설정
                if col_name in ["총량지점명", "일자"]:
                    width = 10.5
                else:
                    width = 8.5
                
                # (2) 표시 형식 결정
                if col_name in ["BOD", "TOC", "pH", "DO", "COD", "CDO", "SS"]:
                    cell_format = fmt_float_1
                elif col_name in ["TP", "유량", "TN"]:
                    cell_format = fmt_float_3
                else:
                    cell_format = fmt_default
                
                # 열 서식 적용 (테두리 제외)
                worksheet.set_column(col_idx, col_idx, width, cell_format)
                
                # (3) 헤더 서식 덮어쓰기
                worksheet.write(0, col_idx, col_name, header_format)
            
            # 5. 테두리 적용 (데이터 영역만)
            # 조건부 서식을 이용해 값이 있는 곳(Formula: =TRUE)에만 테두리 적용
            worksheet.conditional_format(0, 0, max_row, max_col, {
                'type': 'formula',
                'criteria': '=TRUE',
                'format': border_format
            })