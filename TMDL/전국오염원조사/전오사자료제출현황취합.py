# %%
import pandas as pd
import os

# 파일들이 있는 폴더 경로
folder_path = 'E:/Coding/TMDL/전국오염원조사/전오사_제출현황/'  # 원하는 경로로 바꿔주세요

# 합친 데이터 저장용 리스트
merged_data = []

# 폴더 내 모든 파일 처리
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        # 오염원이름 추출
        try:
            pollution_source = filename.split('_')[1]
        except IndexError:
            print(f"파일명에서 오염원이름 추출 실패: {filename}")
            continue

        # 파일 경로
        file_path = os.path.join(folder_path, filename)

        # 엑셀 읽기
        df = pd.read_excel(file_path)

        # "오염원" 열 추가 (가장 앞에 삽입)
        df.insert(0, '오염원', pollution_source)

        # 리스트에 저장
        merged_data.append(df)

# 모든 데이터프레임을 하나로 병합
combined_df = pd.concat(merged_data, ignore_index=True)

# "시도" 열에서 "강원특별자치도"만 필터링
filtered_df = combined_df[combined_df['시도'] == '강원특별자치도']

# 필요한 열만 선택
df_long = filtered_df[["시군구", "오염원", "파일상태"]]

# pivot: 행은 시군구, 열은 오염원, 값은 파일상태
df_wide = df_long.pivot(index="시군구", columns="오염원", values="파일상태")

# 컬럼 순서 지정
columns_order = [
    "생활계",
    "축산계(가축분뇨현황)",
    "축산계(영업자현황)",
    "축산계(축산물이력제)",
    "축산계(국가가축방역통합시스템)",
    "산업계(산업계)",
    "산업계(세차장)",
    "양식계",
    "매립계",
    "환경기초시설",
    "기타수질오염원"
]

# 순서 맞추기 (없으면 NaN으로 채워짐)
df_wide = df_wide.reindex(columns=columns_order)

# 시군구를 첫 번째 열로 다시 복원(인덱스를 열로)
df_wide.reset_index(inplace=True)

# 시군 순서 지정
sigungu_order = [
    "춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시",
    "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군",
    "화천군", "양구군", "인제군", "고성군", "양양군"
]

# 순서대로 정렬
df_wide["시군구"] = pd.Categorical(df_wide["시군구"], categories=sigungu_order, ordered=True)
df_wide = df_wide.sort_values("시군구").reset_index(drop=True)

# 최종 결과 저장
output_path = 'E:/Coding/TMDL/전국오염원조사/전오사_제출현황/Output/전오사_제출현황.xlsx'
df_wide.to_excel(output_path, index=False)

print(f"완료! 파일 저장됨: {output_path}")

# %%
