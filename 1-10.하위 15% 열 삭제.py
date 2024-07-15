import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '전처리_화물차_항만출입데이터_210401_230331.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# '입문시각'과 '출문시각'을 datetime 형식으로 변환합니다.
df['입문시각'] = pd.to_datetime(df['입문시각'])
df['출문시각'] = pd.to_datetime(df['출문시각'])

df['체류시간'] = (df['출문시각'] - df['입문시각']).dt.total_seconds()

# 체류시간이 가장 짧은 하위 10%를 구합니다.
threshold = df['체류시간'].quantile(0.15)

print(f"하위 10% 임계값 (초): {threshold}")

df_filtered = df[df['체류시간'] > threshold]

# '체류시간' 열을 제거합니다.
df_filtered = df_filtered.drop(columns=['체류시간'])

output_file_path = '하위10%시간_전처리_화물차_항만출입데이터_210401_230331.csv'
df_filtered.to_csv(output_file_path, index=False, encoding='euc-kr')

