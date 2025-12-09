# %%
import pandas as pd

# 파일 경로 설정
base_file = "E:/Coding/TMDL/수질분석/총량측정망_2007_2023.xlsx"
data_2024_file = "E:/Coding/TMDL/수질분석/자료조회_2024.xlsx"
output_file = "E:/Coding/TMDL/수질분석/총량측정망_2007_2024(파이썬테스트).xlsx"
output_file_kangwon = "E:/Coding/TMDL/수질분석/총량측정망_강원_2007_2024(파이썬테스트).xlsx"

# %% 총량지점명 리스트 정의
selected_sites = [
    "가평A", "경안A", "경안B", "골지A", "공릉A", "굴포A", "달천A", "달천B",
    "문산A", "복하A", "북한A", "북한B", "북한C", "북한D", "섬강A", "섬강B",
    "소양A", "소양B", "신천A", "안양A", "양화A", "영평A", "오대A", "옥동A",
    "왕숙A", "인북A", "임진A", "임진B", "제천A", "조종A", "주천A", "중랑A",
    "청미A", "탄천A", "평창A", "한강A", "한강B", "한강C", "한강D", "한강E",
    "한강F", "한강G", "한강H", "한강I", "한탄A", "한탄B", "홍천A", "흑천A",
    "낙본A"
]

kangwon_sites = [
    "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A", "섬강A", "섬강B",
    "북한A", "북한B", "소양A", "인북A", "소양B", "북한C", "홍천A", "한탄A",
    "제천A", "한강B", "한강D", "북한D", "임진A", "한탄B", "낙본A"
]

# %% 데이터 처리
# 2024년 데이터 불러오기 및 정리
data_2024_processed = (
    pd.read_excel(data_2024_file, skiprows=1)
    .set_axis([
        "총량지점명", "일자", "수온", "pH", "EC", "DO", "BOD", "COD",
        "SS", "TN", "TP", "TOC", "유량"
    ], axis=1)
    .reindex(columns=[
        "총량지점명", "일자", "BOD", "TP", "유량", "TOC", 
        "수온", "pH", "EC", "DO", "COD", "SS", "TN"
    ])
    .assign(
        일자 = lambda df: pd.to_datetime(df['일자'].str.replace('.', '-'), format='%Y-%m-%d'),
        연도 = lambda df: df['일자'].dt.year,
        월 = lambda df: df['일자'].dt.month
    )
    .query("총량지점명 in @selected_sites")
)

# %% 기존 데이터 불러오기 및 합치기
base = pd.read_excel(base_file)
data = pd.concat([base, data_2024_processed], ignore_index=True)

# %% 강원지역 포함 유역 자료 정리
category_order = [
    "골지A", "오대A", "주천A", "평창A", "옥동A", "한강A", "섬강A", "섬강B",
    "북한A", "북한B", "소양A", "인북A", "소양B", "북한C", "홍천A", "한탄A",
    "제천A", "한강B", "한강D", "북한D", "임진A", "한탄B", "낙본A"
]

data_kangwon = (
    data
    .query("총량지점명 in @kangwon_sites")
    .assign(
        총량지점명=lambda df: pd.Categorical(
            df['총량지점명'],
            categories=category_order,
            ordered=True
        )
    )
    .sort_values('총량지점명')
)

# %% 엑셀 파일 내보내기
data.to_excel(output_file, index=False)
data_kangwon.to_excel(output_file_kangwon, index=False)

# %%
