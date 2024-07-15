import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '울산본항상하위15%_필터링_화물차_항만출입데이터_210401_230331.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# '입문시각'과 '출문시각'을 datetime 형식으로 변환합니다.
df['입문시각'] = pd.to_datetime(df['입문시각'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['출문시각'] = pd.to_datetime(df['출문시각'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# '입문시각'과 '출문시각'에 빈 값이 있는 행을 제거합니다.
#df = df.dropna(subset=['입문시각', '출문시각'])

# '체류시간' 열을 추가하여 '출문시각'과 '입문시각'의 차이를 초 단위로 계산합니다.
df['체류시간'] = ((df['출문시각'] - df['입문시각']).dt.total_seconds())/60

# 결과를 새로운 CSV 파일로 저장합니다.
output_file_path = '분단위울산본항상하위15%_필터링_화물차_항만출입데이터_210401_230331.csv'
df.to_csv(output_file_path, index=False, encoding='euc-kr')

print(f"Cleaned data with '체류시간' saved to {output_file_path}")
