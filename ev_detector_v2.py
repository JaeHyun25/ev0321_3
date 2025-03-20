import cv2
import numpy as np
from ev_classifier_2 import EVClassifier
from ev_predict_2 import EVPredictor
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import time
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """검출 결과 데이터 클래스"""
    plate_number: str
    is_ev: bool
    confidence: float
    timestamp: datetime
    processing_time: float
    plate_area: Dict  # 번호판 위치 정보

class EVDetector:
    def __init__(self, xgb_model_path: str = "/home/ijh/auto_llm/LLMs/xgb_model.pkl", lgbm_model_path: str = "/home/ijh/auto_llm/LLMs/lgbm_model.pkl"):
        """초기화"""
        try:
            self.classifier = EVClassifier(xgb_model_path, lgbm_model_path)
            self.predictor = EVPredictor(xgb_model_path, lgbm_model_path)
            self.logger = logging.getLogger(__name__)
            logger.info("EVDetector 초기화 완료")
        except Exception as e:
            logger.error(f"EVDetector 초기화 중 오류 발생: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray, plate_info: Dict) -> DetectionResult:
        """단일 프레임 처리"""
        try:
            start_time = time.time()
            
            # plate_info 구조 로깅
            logger.info(f"입력된 plate_info 구조: {json.dumps(plate_info, indent=2, ensure_ascii=False)}")
            
            # 번호판 정보 추출
            if isinstance(plate_info, list) and len(plate_info) > 0:
                plate_info = plate_info[0]
            
            # area 정보 추출
            area = plate_info.get('area', {})
            if not area:
                raise ValueError("번호판 영역 정보를 찾을 수 없습니다.")
            
            crop_box = (area['x'], area['y'], area['width'], area['height'])
            
            # 이미지 전처리
            hsv_image = self.classifier.preprocess_image(frame, crop_box, area.get('angle', 0))
            
            # 특징 추출
            features = self.classifier.extract_features(hsv_image)
            
            # 번호판 텍스트 추출
            plate_text = plate_info.get('text', '')
            
            # 예측
            result = self.predictor.predict_single(features, plate_text)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                plate_number=plate_text,
                is_ev=result.is_ev,
                confidence=result.confidence,
                timestamp=datetime.now(),
                processing_time=processing_time,
                plate_area=area
            )
            
        except Exception as e:
            logger.error(f"프레임 처리 중 오류 발생: {str(e)}")
            raise

    def process_batch(self, frame: np.ndarray, plate_infos: List[Dict]) -> List[DetectionResult]:
        """여러 번호판 일괄 처리"""
        results = []
        for plate_info in plate_infos:
            result = self.process_frame(frame, plate_info)
            results.append(result)
        return results

    def save_results(self, results: List[DetectionResult], output_path: str):
        """결과 저장"""
        try:
            # 결과를 딕셔너리 리스트로 변환
            results_dict = [
                {
                    'plate_number': r.plate_number,
                    'is_ev': r.is_ev,
                    'confidence': r.confidence,
                    'processing_time': r.processing_time,
                    'timestamp': r.timestamp.isoformat(),
                    'plate_area': r.plate_area
                }
                for r in results
            ]
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4, ensure_ascii=False)
                
            logger.info(f"결과가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {str(e)}")
            raise

def test_detector():
    """테스트 함수"""
    try:
        # 테스트 이미지 로드
        frame = cv2.imread("test_car.jpg")
        if frame is None:
            logger.error("테스트 이미지를 로드할 수 없습니다.")
            return

        # 테스트용 번호판 정보
        plate_info = {
            "area": {"x": 100, "y": 100, "width": 200, "height": 50, "angle": 0},
            "text": "01가1234"
        }
        
        # 검출기 초기화 및 테스트
        detector = EVDetector()
        result = detector.process_frame(frame, plate_info)
        
        # 결과 출력
        logger.info(f"차량번호: {result.plate_number}")
        logger.info(f"전기차 여부: {'전기차' if result.is_ev else '일반차'}")
        logger.info(f"신뢰도: {result.confidence:.2f}")
        logger.info(f"처리 시간: {result.processing_time:.3f}초")
        
        # 결과 저장
        detector.save_results([result], "test_results.json")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_detector() 