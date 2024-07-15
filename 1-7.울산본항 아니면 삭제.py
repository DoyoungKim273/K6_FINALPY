import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '상하위15%제거_전처리_화물차_항만출입데이터_210401_230331.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# '초소명' 열이 '울산본항'인 경우만 남깁니다.
df_filtered = df[df['초소명'] == '울산본항']

# 결과를 새로운 CSV 파일로 저장합니다.
output_file_path = '울산본항상하위15%_필터링_화물차_항만출입데이터_210401_230331.csv'
df_filtered.to_csv(output_file_path, index=False, encoding='euc-kr')

print(f"Filtered data saved to {output_file_path}")
