from ev_classifier_2 import EVClassifier
from ev_predict_2 import EVPredictor
import json
import os
import warnings
import time
import cv2
import logging
from ev_detector_v2 import EVDetector
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 경고 메시지 무시
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def process_single_image(image_path: str, json_path: str, output_json_path: str):
    """단일 이미지 처리 함수"""
    try:
        # 실행 시간 측정 시작
        start_time = time.time()

        # 1. JSON에서 crop 정보 가져오기
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        if isinstance(json_data, list):
            json_data = json_data[0]

        try:
            crop_list = [
                json_data['area']['x'],
                json_data['area']['y'],
                json_data['area']['width'],
                json_data['area']['height'],
                json_data['area']['angle']
            ]
        except KeyError:
            logger.error("JSON 키 오류! 데이터 구조 확인 필요!")
            return

        # 2. 이미지 로드 및 특징 추출
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return

        classifier = EVClassifier("xgb_model.pkl", "lgbm_model.pkl")
        
        # 이미지 전처리
        hsv_image = classifier.preprocess_image(image, crop_list[:4], crop_list[4])
        # 특징 추출
        features = classifier.extract_features(hsv_image)

        # 3. 예측 실행
        predictor = EVPredictor("xgb_model.pkl", "lgbm_model.pkl")
        prediction = predictor.predict_single(features, json_data["text"])

        # 실행 시간 측정 종료
        elapsed_time = round(time.time() - start_time, 4)

        # 4. 결과 저장
        new_result = {
            "area": json_data["area"],
            "text": json_data["text"],
            "ev": prediction.is_ev,
            "confidence": prediction.confidence,
            "elapsed": elapsed_time,
            "timestamp": prediction.timestamp.isoformat()
        }

        # 기존 결과 로드 및 추가
        if os.path.exists(output_json_path):
            with open(output_json_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(new_result)

        # 결과 저장
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        # 결과 출력
        ev_result = "EV" if prediction.is_ev else "일반차"
        logger.info(f"판별 결과: {ev_result}")
        logger.info(f"신뢰도: {prediction.confidence:.2f}")
        logger.info(f"실행 시간: {elapsed_time}초")
        logger.info(f"결과가 '{output_json_path}'에 추가되었습니다!")

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")

def test_realtime_processing():
    """실시간 처리 테스트 함수"""
    try:
        # 테스트용 이미지 경로
        test_image_path = "test_car.jpg"  # 실제 테스트 이미지 경로로 변경 필요
        
        if not os.path.exists(test_image_path):
            logger.warning(f"테스트 이미지가 없습니다: {test_image_path}")
            return

        # 테스트용 번호판 정보
        plate_info = {
            "area": {"x": 100, "y": 100, "width": 200, "height": 50, "angle": 0},
            "text": "01가1234"
        }

        # 이미지 로드
        frame = cv2.imread(test_image_path)
        if frame is None:
            logger.error("테스트 이미지를 로드할 수 없습니다.")
            return

        # 분류기 및 예측기 초기화
        classifier = EVClassifier("xgb_model.pkl", "lgbm_model.pkl")
        predictor = EVPredictor("xgb_model.pkl", "lgbm_model.pkl")

        # 예측 실행
        hsv_image = classifier.preprocess_image(frame, 
            [plate_info["area"]["x"], plate_info["area"]["y"], 
             plate_info["area"]["width"], plate_info["area"]["height"]], 
            plate_info["area"]["angle"])
        features = classifier.extract_features(hsv_image)
        
        result = predictor.predict_single(features, plate_info["text"])

        # 결과 저장
        predictor.save_results([result], "realtime_predictions.json")
        logger.info(f"실시간 예측 결과: {'EV' if result.is_ev else '일반차'}")

    except Exception as e:
        logger.error(f"실시간 처리 테스트 중 오류 발생: {str(e)}")

def evaluate_model(test_data_path: str):
    """모델 성능 평가"""
    try:
        # 테스트 데이터 로드
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        # 검출기 초기화
        detector = EVDetector()
        
        # 예측 결과 저장용 리스트
        y_true = []  # 실제 레이블
        y_pred = []  # 예측 레이블
        confidences = []  # 예측 신뢰도
        
        # 각 테스트 케이스에 대해
        for case in test_data:
            # 이미지 로드
            image = cv2.imread(case['image_path'])
            if image is None:
                logger.warning(f"이미지를 로드할 수 없습니다: {case['image_path']}")
                continue
                
            # 번호판 정보
            plate_info = case['plate_info']
            
            # 예측 수행
            result = detector.process_frame(image, plate_info)
            if result is None:
                logger.warning(f"예측 실패: {case['image_path']}")
                continue
                
            # 결과 저장
            y_true.append(1 if case['is_ev'] else 0)
            y_pred.append(1 if result.is_ev else 0)
            confidences.append(result.confidence)
            
            # 로깅
            logger.info(f"이미지: {case['image_path']}")
            logger.info(f"실제: {'전기차' if case['is_ev'] else '일반차'}")
            logger.info(f"예측: {'전기차' if result.is_ev else '일반차'}")
            logger.info(f"신뢰도: {result.confidence:.4f}")
            
        # 성능 지표 계산
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        
        # 결과 출력
        logger.info("\n=== 모델 성능 평가 결과 ===")
        logger.info(f"정확도 (Accuracy): {accuracy:.4f}")
        logger.info(f"정밀도 (Precision): {precision:.4f}")
        logger.info(f"재현율 (Recall): {recall:.4f}")
        logger.info(f"F1 점수: {f1:.4f}")
        
        # 신뢰도 통계
        logger.info("\n=== 신뢰도 통계 ===")
        logger.info(f"평균 신뢰도: {np.mean(confidences):.4f}")
        logger.info(f"최대 신뢰도: {np.max(confidences):.4f}")
        logger.info(f"최소 신뢰도: {np.min(confidences):.4f}")
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('혼동 행렬')
        plt.ylabel('실제 레이블')
        plt.xlabel('예측 레이블')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # 신뢰도 분포 시각화
        plt.figure(figsize=(8, 6))
        plt.hist(confidences, bins=50)
        plt.title('예측 신뢰도 분포')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.savefig('confidence_distribution.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 단일 이미지 처리 테스트
    image_path = "/home/ijh/combined/CENTRALCITY_EVMONITORING/EV_TEMP/01누5995_20250223_142333.jpg"
    json_path = "/home/ijh/combined/CENTRALCITY_EVMONITORING/EV_JSON/01누5995_20250223_142333.json"
    output_json_path = "ev_prediction_result.json"
    
    process_single_image(image_path, json_path, output_json_path)
    
    # 실시간 처리 테스트 (선택적)
    # test_realtime_processing()

    # 테스트 데이터 경로 설정
    test_data_path = "test_data.json"
    evaluate_model(test_data_path)

# 초기화
detector = EVDetector("xgb_model.pkl", "lgbm_model.pkl")

# 실시간 처리
frame = get_frame_from_ipcam()  # (1920, 1080, 3) 크기의 numpy array
plate_info = {
    "area": {"x": 100, "y": 100, "width": 200, "height": 50, "angle": 0},
    "text": "01가1234"
}

# 처리
result = detector.process_frame(frame, plate_info)

# 결과 확인
print(f"전기차 여부: {'전기차' if result.is_ev else '일반차'}")
print(f"신뢰도: {result.confidence:.2f}")