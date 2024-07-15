import pandas as pd
from openpyxl import load_workbook

# 항목과 번호 사전 정의
ship_types = {
    'LNG 운반선': 1,
    'LPG 운반선': 2,
    '견인용예선': 3,
    '관공선': 4,
    '급유선': 5,
    '기타 예선': 6,
    '기타 유조선': 7,
    '기타선': 8,
    '모래운반선': 9,
    '산물선(벌크선)': 10,
    '석유제품 운반선': 11,
    '선박용도': 12,
    '세미(혼재)컨테이너선': 13,
    '시멘트운반선': 14,
    '신조선': 15,
    '압항 예선': 16,
    '여객선': 17,
    '원유운반선': 18,
    '일반화물선': 19,
    '자동차운반선': 20,
    '철강재 운반선': 21,
    '케미칼 운반선': 22,
    '케미칼가스 운반선': 23,
    '폐기물 운반선': 24,
    '풀컨테이너선': 25
}

# 엑셀 파일 불러오기
file_path = '선박filtered_data'  # 파일 경로 설정
wb = load_workbook(file_path)
sheet = wb.active

# 엑셀 데이터를 DataFrame으로 변환
data = sheet.values
columns = next(data)
df = pd.DataFrame(data, columns=columns)

# '선박용도' 열을 기준으로 'number' 열 추가
df['number'] = df['선박용도'].apply(lambda x: ship_types[x] if x in ship_types else None)

# 변경된 데이터프레임을 엑셀 파일에 저장
output_file = 'updated_ship_data.xlsx'
df.to_excel(output_file, index=False)

print(f'엑셀 파일이 성공적으로 업데이트되었습니다: {output_file}')

# 저장 후 워크북 닫기
wb.close()
