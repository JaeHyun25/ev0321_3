import os
import yaml
import cv2
import json
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from src.detector.ev_detector import EVDetector
from src.detector.ev_detector_singleton import EVDetectorSingleton
from src.utils.logging_config import setup_logging

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_error_case(config: dict, frame: np.ndarray, plate_info: dict, error_msg: str):
    """에러 케이스 저장"""
    try:
        # 에러 케이스 저장 디렉토리 생성
        error_dir = os.path.join(config['paths']['error_cases_dir'], 
                               datetime.now().strftime('%Y%m%d'))
        os.makedirs(error_dir, exist_ok=True)
        
        # 현재 시간을 파일명으로 사용
        timestamp = datetime.now().strftime('%H%M%S_%f')
        
        # 이미지 저장 (설정에 따라)
        image_path = None
        if config['processing']['save_options']['save_error_image']:
            save_frame = frame.copy()
            if config['processing']['save_options']['resize_saved_image']:
                save_size = tuple(config['processing']['save_options']['saved_image_size'])
                save_frame = cv2.resize(save_frame, save_size)
            
            image_path = os.path.join(error_dir, f'{timestamp}.jpg')
            cv2.imwrite(image_path, save_frame)
        
        # 에러 정보 저장
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'plate_info': plate_info,
            'error_message': error_msg,
            'image_path': image_path
        }
        
        error_json_path = os.path.join(error_dir, f'{timestamp}.json')
        with open(error_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
            
        logging.info(f"에러 케이스 저장 완료: {error_json_path}")
        
    except Exception as e:
        logging.error(f"에러 케이스 저장 실패: {str(e)}")

def save_uncertain_case(config: dict, frame: np.ndarray, plate_info: dict, result: dict):
    """불확실한 판정 결과 저장"""
    try:
        # 신뢰도가 임계값보다 낮은 경우에만 저장
        if result['conf']['ev'] < config['processing']['confidence_threshold']:
            # 저장 디렉토리 생성
            uncertain_dir = os.path.join(config['paths']['uncertain_cases_dir'],
                                       datetime.now().strftime('%Y%m%d'))
            os.makedirs(uncertain_dir, exist_ok=True)
            
            # 현재 시간을 파일명으로 사용
            timestamp = datetime.now().strftime('%H%M%S_%f')
            
            # 이미지 저장 (설정에 따라)
            image_path = None
            if config['processing']['save_options']['save_uncertain_image']:
                save_frame = frame.copy()
                if config['processing']['save_options']['resize_saved_image']:
                    save_size = tuple(config['processing']['save_options']['saved_image_size'])
                    save_frame = cv2.resize(save_frame, save_size)
                
                image_path = os.path.join(uncertain_dir, f'{timestamp}.jpg')
                cv2.imwrite(image_path, save_frame)
            
            # 판정 정보 저장
            case_info = {
                'timestamp': datetime.now().isoformat(),
                'input_plate_info': plate_info,
                'detection_result': result,
                'image_path': image_path
            }
            
            json_path = os.path.join(uncertain_dir, f'{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(case_info, f, indent=2, ensure_ascii=False)
                
            logging.info(f"불확실한 판정 케이스 저장 완료: {json_path}")
            
    except Exception as e:
        logging.error(f"불확실한 판정 케이스 저장 실패: {str(e)}")

def validate_input(frame: np.ndarray, plate_info: dict, config: dict) -> tuple:
    """입력 데이터 검증
    Args:
        frame: np.ndarray - 이미지 데이터
        plate_info: dict - 번호판 정보
        config: dict - 설정 정보
    Returns:
        tuple - (is_valid: bool, error_message: str)
    """
    # 이미지 검증
    if frame is None:
        return False, "이미지 데이터가 없습니다."
    
    if not isinstance(frame, np.ndarray):
        return False, "이미지 데이터가 numpy array 형식이 아닙니다."
    
    if frame.shape != (1080, 1920, 3):
        return False, f"이미지 크기가 잘못되었습니다. 현재: {frame.shape}, 필요: (1080, 1920, 3)"
    
    # plate_info 검증
    required_fields = ['area', 'attrs', 'conf', 'text']
    for field in required_fields:
        if field not in plate_info:
            return False, f"필수 필드가 없습니다: {field}"
    
    return True, ""

def process_realtime_data(frame: np.ndarray, plate_info: dict, detector: EVDetector, config: dict) -> dict:
    """실시간 데이터 처리"""
    retry_count = config['realtime']['error_handling']['retry_count']
    retry_delay = config['realtime']['error_handling']['retry_delay']
    
    for attempt in range(retry_count):
        try:
            # 입력 데이터 검증
            is_valid, error_msg = validate_input(frame, plate_info, config)
            if not is_valid:
                save_error_case(config, frame, plate_info, error_msg)
                return None
                
            # area 정보를 detector가 이해할 수 있는 형식으로 변환
            area_info = {
                "plate_number": plate_info["text"],
                "area": [
                    plate_info["area"]["x"],
                    plate_info["area"]["y"],
                    plate_info["area"]["x"] + plate_info["area"]["width"],
                    plate_info["area"]["y"] + plate_info["area"]["height"]
                ],
                "angle": plate_info["area"]["angle"],
                "confidence": {
                    "ocr": plate_info["conf"]["ocr"],
                    "plate": plate_info["conf"]["plate"]
                }
            }
            
            # 이미지 처리
            result = detector.process_frame(frame, area_info)
            
            # 결과 생성
            detection_result = {
                "area": {
                    "angle": area_info["angle"],
                    "height": plate_info["area"]["height"],
                    "width": plate_info["area"]["width"],
                    "x": plate_info["area"]["x"],
                    "y": plate_info["area"]["y"]
                },
                "attrs": {
                    "ev": result.is_ev
                },
                "conf": {
                    "ocr": plate_info["conf"]["ocr"],
                    "plate": plate_info["conf"]["plate"],
                    "ev": result.confidence
                },
                "elapsed": result.processing_time,
                "ev": result.is_ev,
                "text": result.plate_number,
                "timestamp": result.timestamp.isoformat()
            }
            
            # 처리 시간 체크
            if (result.processing_time > config['realtime']['performance']['max_processing_time'] and 
                config['realtime']['performance']['skip_if_exceeded']):
                error_msg = f"처리 시간 초과: {result.processing_time:.4f}초"
                save_error_case(config, frame, plate_info, error_msg)
                return None
            
            # 불확실한 판정 결과 저장
            save_uncertain_case(config, frame, plate_info, detection_result)
            
            return detection_result
            
        except Exception as e:
            error_msg = f"실시간 처리 중 오류 발생 (시도 {attempt + 1}/{retry_count}): {str(e)}"
            logging.error(error_msg)
            
            if attempt < retry_count - 1:
                import time
                time.sleep(retry_delay)
                continue
                
            save_error_case(config, frame, plate_info, error_msg)
            return None

def process_image_in_memory(frame: np.ndarray, plate_info: dict) -> dict:
    """In-memory 방식의 이미지 처리
    
    Args:
        frame: np.ndarray - (1080, 1920, 3) 크기의 BGR 이미지
        plate_info: dict - 번호판 정보 {
            "area": {"angle": float, "height": int, "width": int, "x": int, "y": int},
            "attrs": {"ev": bool},
            "conf": {"ocr": float, "plate": float},
            "text": str
        }
    
    Returns:
        dict - 처리 결과 {
            "area": {"angle": float, "height": int, "width": int, "x": int, "y": int},
            "attrs": {"ev": bool},
            "conf": {"ocr": float, "plate": float, "ev": float},
            "elapsed": float,
            "ev": bool,
            "text": str,
            "timestamp": str
        }
    
    Raises:
        RuntimeError: detector가 초기화되지 않은 경우
        ValueError: 입력 데이터가 올바르지 않은 경우
    """
    # 싱글톤 인스턴스 가져오기
    detector = EVDetectorSingleton()
    
    # 초기화 확인
    if not detector.is_initialized:
        config = load_config('config/config.yaml')
        detector.initialize(config)
    
    # 이미지 처리
    return detector.process_image(frame, plate_info)

def main():
    """테스트용 메인 함수"""
    # 설정 로드
    config = load_config('config/config.yaml')
    
    # 로깅 설정
    logger = setup_logging(config['paths']['logs_dir'])
    logger.info("전기차 감지 시스템 시작")
    
    # EVDetector 싱글톤 초기화
    detector = EVDetectorSingleton()
    detector.initialize(config)
    
    logger.info("실시간 처리 모드로 시작")
    try:
        # 테스트용 더미 데이터
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_plate_info = {
            "area": {
                "angle": 8.0732,
                "height": 60,
                "width": 111,
                "x": 612,
                "y": 447
            },
            "attrs": {
                "ev": True
            },
            "conf": {
                "ocr": 0.926,
                "plate": 0.9273
            },
            "text": "01너3346"
        }
        
        # In-memory 방식으로 처리
        result = process_image_in_memory(test_frame, test_plate_info)
        if result:
            logger.info("실시간 처리 결과:")
            logger.info(f"  - 번호판: {result['text']}")
            logger.info(f"  - 전기차 여부: {'전기차' if result['ev'] else '일반차'}")
            logger.info(f"  - EV 신뢰도: {result['conf']['ev']:.2f}")
            logger.info(f"  - 처리 시간: {result['elapsed']:.4f}초")
        
    except Exception as e:
        logger.error(f"실시간 처리 중 오류 발생: {str(e)}")

    logger.info("전기차 감지 시스템 종료")

if __name__ == "__main__":
    main() 