import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 불러오기 및 전처리
ship_data = pd.read_excel('ship_data.xlsx')  
truck_data = pd.read_excel('truck_data.xlsx')

# 필요한 열을 datetime 형식으로 변환
ship_data['입항일시'] = pd.to_datetime(ship_data['입항일시'])
ship_data['출항일시'] = pd.to_datetime(ship_data['출항일시'])
truck_data['입문시각'] = pd.to_datetime(truck_data['입문시각'])
truck_data['출문시각'] = pd.to_datetime(truck_data['출문시각'])

# 기간 필터링 (2021-04-01부터 2023-03-31)
ship_data = ship_data[(ship_data['입항일시'] >= '2021-04-01') & (ship_data['출항일시'] <= '2021-12-31')]
truck_data = truck_data[(truck_data['입문시각'] >= '2021-04-01') & (truck_data['출문시각'] <= '2021-12-31')]

# 그래프 설정
plt.figure(figsize=(14, 8))

# 입항일시, 출항일시 그래프
plt.subplot(2, 2, 1)
sns.histplot(ship_data['입항일시'], kde=True, color='blue', bins=50)
plt.title('배_입항일시')

plt.subplot(2, 2, 2)
sns.histplot(ship_data['출항일시'], kde=True, color='green', bins=50)
plt.title('배_출항일시')

# 입문시각, 출문시각 그래프
plt.subplot(2, 2, 3)
sns.histplot(truck_data['입문시각'], kde=True, color='orange', bins=50)
plt.title('차_입문시각')

plt.subplot(2, 2, 4)
sns.histplot(truck_data['출문시각'], kde=True, color='red', bins=50)
plt.title('차_출문시각')

plt.tight_layout()
plt.show()
