from ev_classifier import EVClassifier
from ev_predict import EVPredictor
import json
import os
import warnings
import time  #  실행 시간 측정을 위한 모듈 추가

start_time = time.time()  #  시작 시간 기록
# 환경 변수 설정 (RDRAND 관련 경고 숨기기)
os.environ["PYTHONWARNINGS"] = "ignore"

# Python 경고 숨기기
warnings.filterwarnings("ignore")


# 차량 이미지 & JSON 경로
image_path = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG/01가5345_ice_20250221_104004.jpg"
json_path = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG_JSON/01가5345_ice_20250221_104004.json"
output_json_path = "ev_prediction_result.json"

#  실행 시간 측정 시작
start_time = time.time()

# 1️ JSON에서 crop 정보 가져오기
with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

if isinstance(json_data, list):
    json_data = json_data[0]  # 리스트일 경우 첫 번째 항목 선택

try:
    crop_list = [
        json_data['area']['x'],
        json_data['area']['y'],
        json_data['area']['width'],
        json_data['area']['height'],
        json_data['area']['angle']
    ]
except KeyError:
    print(" JSON 키 오류! 데이터 구조 확인 필요!")
    exit()

# 2️ HSV 특징 추출
classifier = EVClassifier("xgb_model.pkl", "lgbm_model.pkl")
features = classifier.feature_extraction_image(image_path, crop_list)

# 3️ 예측 실행
predictor = EVPredictor("xgb_model.pkl", "lgbm_model.pkl")
prediction = predictor.predict(features)

#  실행 시간 측정 종료
end_time = time.time()
elapsed_time = round(end_time - start_time, 4)

# 4️ 새로운 결과 JSON 객체 생성
new_result = {
    "area": json_data["area"],
    "text": json_data["text"],  # 차량번호 그대로 유지
    "ev": bool(prediction),  # 🔹 전기차 여부 (True/False)
    "elapsed": elapsed_time  # 🔹 실행 시간 추가
}

# 5️ 기존 JSON 파일이 있으면 불러오기 (누적 저장)
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]  # 기존 데이터가 리스트가 아니면 리스트로 변환
        except json.JSONDecodeError:
            existing_data = []  # 파일이 비어있거나 JSON 오류가 있으면 빈 리스트로 초기화
else:
    existing_data = []  # 파일이 없으면 빈 리스트로 시작

# 6️ 새로운 결과 추가
existing_data.append(new_result)

# 7️ 업데이트된 JSON 파일 저장
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, indent=4, ensure_ascii=False)

#  터미널에 EV 여부와 실행 시간 출력
ev_result = "EV" if prediction else "일반차"
print(f" 판별 결과: {ev_result}")
print(f" 실행 시간: {elapsed_time}초")
print(f" 결과가 '{output_json_path}'에 추가되었습니다!")