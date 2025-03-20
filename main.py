import os
import yaml
import cv2
import json
import logging
from datetime import datetime
from src.detector.ev_detector import EVDetector
from src.utils.logging_config import setup_logging

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 설정 로드
    config = load_config('config/config.yaml')
    
    # 로깅 설정
    logger = setup_logging(config['paths']['logs_dir'])
    logger.info("전기차 감지 시스템 시작")
    
    # 결과 저장 디렉토리 생성
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_dir = config['paths']['image_dir']
    json_dir = config['paths']['json_dir']
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    logger.info(f"발견된 이미지 파일: {len(image_files)}개")
    logger.info(f"발견된 JSON 파일: {len(json_files)}개")
    
    # 날짜별 결과를 저장할 딕셔너리
    daily_results = {}
    
    # EVDetector 초기화
    detector = EVDetector(
        config['model']['xgb_path'],
        config['model']['lgbm_path']
    )
    
    # 각 이미지 파일 처리
    for image_file in image_files:
        try:
            # 이미지 파일 경로
            image_path = os.path.join(image_dir, image_file)
            
            # JSON 파일 찾기
            json_file = image_file.rsplit('.', 1)[0] + '.json'
            json_path = os.path.join(json_dir, json_file)
            
            if not os.path.exists(json_path):
                logger.warning(f"JSON 파일을 찾을 수 없음: {json_file}")
                continue
            
            # 이미지 로드
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"이미지를 로드할 수 없음: {image_path}")
                continue
            
            # JSON 파일 읽기
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data or len(data) == 0:
                logger.warning(f"JSON 데이터가 비어있음: {json_file}")
                continue
            
            # 첫 번째 번호판 정보 사용
            plate_info = data[0]
            
            # 이미지 처리
            result = detector.process_frame(frame, plate_info)
            
            # 날짜 추출 (파일명에서)
            date_str = image_file.split('_')[0]  # 예: 20250218_000000_01가1234.jpg -> 20250218
            
            # 날짜별 결과 저장
            if date_str not in daily_results:
                daily_results[date_str] = []
            
            daily_results[date_str].append({
                'image_file': image_file,
                'plate_number': result.plate_number,
                'is_ev': result.is_ev,
                'confidence': result.confidence,
                'timestamp': result.timestamp.isoformat(),
                'processing_time': result.processing_time
            })
            
            logger.info(f"처리 완료: {image_file}")
            logger.info(f"  - 번호판: {result.plate_number}")
            logger.info(f"  - 전기차 여부: {'전기차' if result.is_ev else '일반차'}")
            logger.info(f"  - 신뢰도: {result.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {image_file}")
            logger.error(f"오류 내용: {str(e)}")
            continue
    
    # 날짜별로 결과 저장
    for date_str, results in daily_results.items():
        output_file = os.path.join(
            config['paths']['results_dir'],
            f"ev_detection_results_{date_str}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"결과 저장 완료: {output_file} ({len(results)}개 항목)")
    
    logger.info(f"총 처리된 파일 수: {len(image_files)}")
    logger.info("전기차 감지 시스템 종료")

if __name__ == "__main__":
    main() 