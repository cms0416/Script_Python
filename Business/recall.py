# %% 패키지 로드
import pandas as pd

# %% ---------------------------------------------------------------------------
# 1. 엑셀 파일 경로 설정 및 데이터 로드
# 파일 경로 설정
file_path = 'C:/Coding/Business/회수폐기공고목록_생약표기.xlsx'

# 엑셀 데이터 로드 (인코딩 옵션 불필요)
# 엔진은 openpyxl을 사용하며, 설치가 안 되어 있다면 pip install openpyxl 명령어로 설치하십시오.
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    # 파일이 열려 있거나 경로가 틀린 경우를 대비한 예외 처리

df

# %% ---------------------------------------------------------------------------
# 2. A열(구분)이 "생약"인 행만 필터링
# 열 이름이 '구분'과 '회수사유'라고 가정하며, 실제 파일의 열 인덱스(0, 4)를 기준으로 안전하게 선택합니다.
df_filtered = df[df.iloc[:, 0] == '생약'].copy()

df_filtered 

# %% ---------------------------------------------------------------------------
# 3. 필요한 열(A열: 구분, E열: 회수사유)만 선택
# iloc를 사용하여 첫 번째(0)와 다섯 번째(4) 열을 추출합니다.
df_final = df_filtered.iloc[:, [0, 4]].copy()
df_final.columns = ['구분', '회수사유']

df_final

# %% ---------------------------------------------------------------------------
# 4. 회수 사유 통합 카테고리 분류 함수 정의
def classify_reason(reason):
    if pd.isna(reason):
        return "데이터 없음"
    
    # 공백 제거 및 텍스트 표준화
    reason = str(reason).replace(" ", "")

    # 카테고리별 핵심 키워드 매핑 로직
    if any(k in reason for k in ['관능', '성상', '색택', '냄새', '맛', '형상', '외관']):
        return '성상(관능) 부적합'
    elif any(k in reason for k in ['중금속', '납', '비소', '수은', '카드뮴', 'Pb', 'As', 'Hg', 'Cd']):
        return '순도시험(중금속)'
    elif any(k in reason for k in ['농약', '잔류농약', '카벤다짐', '디클로르보스', '말라티온', '펜벤다짐']):
        return '순도시험(잔류농약)'
    elif any(k in reason for k in ['이산화황', '아플라톡신', '곰팡이독소', '이물', '미생물', '벤조피렌']):
        return '순도시험(기타)'
    elif '확인시험' in reason:
        return '확인시험 부적합'
    elif '회분' in reason:
        return '회분/산불용성회분'
    elif any(k in reason for k in ['정량', '함량', '지표성분', '엑스']):
        return '정량/함량 부적합'
    elif any(k in reason for k in ['건조감량', '수분']):
        return '건조감량 부적합'
    else:
        return '기타(기준외 등)'

# %% ---------------------------------------------------------------------------
# 5. 새로운 열(통합 카테고리) 추가
df_final['통합 카테고리'] = df_final['회수사유'].apply(classify_reason)

df_final

# %% ---------------------------------------------------------------------------
# 6. 분석 결과 요약 표 생성 (생약 회수 사유 통계)
# 빈도수 계산 및 데이터프레임 변환
df_summary = df_final['통합 카테고리'].value_counts().reset_index()
df_summary.columns = ['통합 카테고리', '빈도수']

# 순위 추가
df_summary.insert(0, '순위', range(1, len(df_summary) + 1))

# 주요 내용 매핑 (시각적 이해를 돕기 위한 설명 열 추가)
content_map = {
    '순도시험(중금속)': '납, 비소, 수은, 카드뮴 등 무기 오염물질',
    '성상(관능) 부적합': '외관, 색택, 냄새, 맛, 형상 등 오감 판정',
    '순도시험(기타)': '이산화황, 곰팡이독소, 이물, 미생물, 벤조피렌 등',
    '기타(기준외 등)': '기타 규격 기준 부적합 및 미분류 사유',
    '순도시험(잔류농약)': '살충제, 살균제 등 농약 성분 검출',
    '정량/함량 부적합': '지표성분, 엑스함량 등 유효성분 함량 미달',
    '회분/산불용성회분': '무기물질 잔류량(회분) 기준 초과',
    '확인시험 부적합': '기원 식물 및 약재 정체성 확인 시험 실패',
    '건조감량 부적합': '수분 함량 기준 초과 및 건조 상태 불량'
}
df_summary['주요 내용'] = df_summary['통합 카테고리'].map(content_map)

df_summary


# %% ---------------------------------------------------------------------------
# 결과 저장(Excel 파일로 저장)
output_path = 'C:/Coding/생약_회수사유_분석결과.xlsx'
# 엑셀로 저장 df_summary와 df_final 두 시트를 저장
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='회수사유_요약', index=False)  
    df_final.to_excel(writer, sheet_name='회수사유_상세', index=False)  
    
print(f"분석 결과가 '{output_path}'에 저장되었습니다.")

# %% [markdown]
# ******************************************************************************
# # 그래프

# %% ---------------------------------------------------------------------------
# 시각화 패키지 로드
import matplotlib.pyplot as plt
import pandas as pd

# %% ---------------------------------------------------------------------------
# 1. 한글 폰트 설정 (Windows 기준 나눔고딕 등 설치된 폰트 지정)
plt.rcParams['font.family'] = 'Malgun Gothic' # 맑은 고딕 설정
plt.rcParams['axes.unicode_minus'] = False

# %% ---------------------------------------------------------------------------
# 2. 가로 막대 그래프 시각화 (Horizontal Bar Chart)
# 위에서 생성한 df_summary 사용
df_sorted = df_summary.sort_values(by='빈도수', ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(df_sorted['통합 카테고리'], df_sorted['빈도수'], color='skyblue')

# 막대 끝에 빈도수 텍스트 표시
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f"{int(bar.get_width())}건", va='center', fontsize=11)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('생약 품질 부적합 회수 사유 빈도 현황', fontsize=15, weight='bold', pad=20)
plt.xlabel('빈도수 (건)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% ---------------------------------------------------------------------------
# 3. 파레토 차트 시각화 (Pareto Chart)
df_pareto = df_summary.sort_values(by='빈도수', ascending=False)
df_pareto['누적비율'] = df_pareto['빈도수'].cumsum() / df_pareto['빈도수'].sum() * 100

fig, ax1 = plt.subplots(figsize=(10, 6))

# 빈도수 막대 그래프
ax1.bar(df_pareto['통합 카테고리'], df_pareto['빈도수'], color='steelblue', alpha=0.8)
ax1.set_ylabel('빈도수(건)', fontsize=14, weight='bold')
plt.xticks(fontsize=13, rotation=45, ha='right')  # x축 라벨 45도 기울이기
plt.yticks(fontsize=13)

# 누적 비율 꺾은선 그래프
ax2 = ax1.twinx()
ax2.plot(df_pareto['통합 카테고리'], df_pareto['누적비율'], color='red', marker='o', ms=5, label='누적비율')
ax2.set_ylabel('누적 백분율(%)', fontsize=14, weight='bold')
ax2.set_ylim(0, 110)

# 80% 라인 표시 (집중 관리 포인트)
ax2.axhline(80, color='orange', linestyle='--', alpha=0.5)
ax2.text(len(df_pareto)-1, 82, '80%', color='orange', fontsize=13, fontweight='bold')
plt.yticks(fontsize=13)

plt.title('생약 회수 사유 파레토 분석(Pareto Analysis)', fontsize=15, weight='bold', pad=20)
plt.tight_layout()
plt.show()

# %%
