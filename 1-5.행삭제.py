import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '하위10%시간_전처리_화물차_항만출입데이터_210401_230331.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# '출문시각' 열과 '초소명' 열에 값이 없는 행을 제거합니다.
df_cleaned = df.dropna(subset=['차명'])

# 결과를 새로운 CSV 파일로 저장합니다.
output_file_path = '빈칸삭제하위10%시간_전처리_화물차_항만출입데이터_210401_230331.csv'
df_cleaned.to_csv(output_file_path, index=False, encoding='euc-kr')

print(f"Cleaned data saved to {output_file_path}")
