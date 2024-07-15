
import pandas as pd

# xlsx 파일 읽기
file_path = '선박입출항현황(210401-230331).xlsx'  # 파일 경로 설정
df = pd.read_excel(file_path)

# 출항일시 열을 datetime 형식으로 변환
df['출항일시'] = pd.to_datetime(df['출항일시'], errors='coerce')

# 유효한 날짜만 필터링
df = df.dropna(subset=['출항일시'])

# 출항일시에서 년도, 월, 일, 시간, 분 추출
df['출항년도'] = df['출항일시'].dt.year
df['출항월'] = df['출항일시'].dt.month
df['출항일'] = df['출항일시'].dt.day
df['출항시간'] = df['출항일시'].dt.hour
df['출항분'] = df['출항일시'].dt.minute

# 필요한 열만 선택하여 정리
df_sorted = df[['출항년도', '출항월', '출항일', '출항시간', '출항분', '출항일시']].sort_values(by='출항일시')

# 결과를 Excel 파일로 저장
output_file_path = 'ship_data.xlsx'
df_sorted.to_excel(output_file_path, index=False, sheet_name='출항일시 정리')

