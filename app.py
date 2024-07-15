from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import joblib

# flask 애플리케이션 초기화
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# 모델 로드
model = torch.load('lstm_model.pth')

# 모델을 평가 모드로 전환
model.eval()

# 레이블 인코더 로드
label_encoder_차종 = joblib.load('label_encoder_차종.pkl')

# 예측 함수 정의
def predict_processing_time(year, month, day, hour, truck_type, ship_count):
    
    # 입력 데이터를 데이터프레임으로 생성
    input_data = pd.DataFrame([{
        '입문시각_연도': year,  
        '입문시각_월': month,   
        '입문시각_일': day,     
        '입문시각_시간': hour,  
        '차종': truck_type,     
        '선박_갯수': ship_count 
    }])
    
    # 차종에 인코딩 적용
    input_data['차종'] = label_encoder_차종.transform(input_data['차종'])
    
    # 데이터를 PyTorch 텐서로 변환
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
    # LSTM 입력 형태로 변환 (batch_size=1, seq_length=1, input_size)
    input_tensor = input_tensor.unsqueeze(1)  
    
    # 그래디언트 계산 비활성화 (메모리 절약)
    with torch.no_grad():  
        # 모델 예측 수행
        predicted_time = model(input_tensor)  
    
    # 예측 결과를 숫자로 반환
    return predicted_time.item() 

# 예측을 처리하는 엔드포인트 정의
# url은 변경을 하지 않아도 됨
@app.route('/predict', methods=['POST'])
def predict():
    # 요청으로부터 데이터 가져오기
    data = request.json
    year = data['year']
    month = data['month']
    day = data['day']
    hour = data['hour']
    truck_type = data['truck_type']
    ship_count = data['ship_count']

    # 예측 함수 호출
    prediction = predict_processing_time(year, month, day, hour, truck_type, ship_count)
    
    # 예측 결과를 JSON 형태로 변환
    return jsonify({'predicted_time': prediction})

# flask 앱 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)