import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '상하위15%제거_전처리_화물차_항만출입데이터_210401_230331.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# '입문시각'과 '출문시각'을 datetime 형식으로 변환합니다.
df['입문시각'] = pd.to_datetime(df['입문시각'])
df['출문시각'] = pd.to_datetime(df['출문시각'])

# '체류시간' 열을 추가하여 '출문시각'과 '입문시각'의 차이를 계산합니다.
df['체류시간'] = (df['출문시각'] - df['입문시각']).dt.total_seconds()

# 체류시간이 가장 긴 상위 10%를 구합니다.
threshold = df['체류시간'].quantile(0.85)

print(f"상위 10% 임계값 (초): {threshold}")

df_filtered = df[df['체류시간'] <= threshold]


# 결과를 새로운 CSV 파일로 저장합니다.
output_file_path = '2상하위15%제거_전처리_화물차_항만출입데이터_210401_230331.csv'
df_filtered.to_csv(output_file_path, index=False, encoding='euc-kr')

print(f"Cleaned data saved to {output_file_path}")
