import pandas as pd
from sklearn.model_selection import train_test_split

# 엑셀 파일에서 데이터 불러오기
file_path = 'ship_data.xlsx'  
df = pd.read_excel(file_path)

# 데이터를 train/test로 분리 (80:20 비율)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=70)

train_df.to_excel('ship_test_data1.xlsx', index=False)
test_df.to_excel('ship_test_data2.xlsx', index=False)

