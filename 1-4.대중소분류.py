import pandas as pd
import numpy as np

# CSV 파일을 읽어옵니다.
file_path = '2진짜전처리_화물차_항만출입데이터_210401_230331_체류시간.csv'
df = pd.read_csv(file_path, encoding='euc-kr')

# 결측치(NaN) 처리
df.fillna('', inplace=True)  # NaN을 빈 문자열로 대체

# '대중소분류' 함수를 정의합니다.
def classify_vehicle_type(row):
    if '화물 소형(1t미만)' in row['차종'] or '2.5톤미만' in row['차종']:
        return '소'
    elif '화물 중형(1t이상~5t미만)' in row['차종'] or '2.5톤이상' in row['차종']:
        return '중'
    elif '화물 대형(5t이상)' in row['차종'] or '컨테이너차량' in row['차종'] or '특수차량(중장비)' in row['차종']:
        return '대'
    else:
        return None

# '차종' 열이 NaN인 경우 처리
df['차종'] = df['차종'].astype(str)  # '차종' 열을 문자열로 변환
df['차종'] = df['차종'].apply(lambda x: x.strip())  # 문자열 좌우 공백 제거

# '대중소분류' 열 추가하기
df['대중소분류'] = df.apply(classify_vehicle_type, axis=1)

# 결과를 CSV 파일로 저장합니다.
output_file_path = '3전처리_화물차_항만출입데이터_210401_230331.csv'
df.to_csv(output_file_path, index=False, encoding='euc-kr')

