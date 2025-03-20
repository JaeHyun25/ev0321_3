from ev_detector_v2 import EVDetector
import cv2
import json
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_camera_stream(camera_url: str):
    """실시간 카메라 스트림 처리"""
    try:
        # 검출기 초기화
        detector = EVDetector("xgb_model.pkl", "lgbm_model.pkl")
        
        # 카메라 연결
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            raise ValueError("카메라 연결 실패")
            
        logger.info("카메라 스트림 처리 시작")
        
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                logger.error("프레임 읽기 실패")
                break             
            
            # 테스트용 번호판 정보
            plate_info = {
                "area": {"x": 100, "y": 100, "width": 200, "height": 50, "angle": 0},
                "text": "01가1234"
            }
            
            # 전기차 검출
            result = detector.process_frame(frame, plate_info)
            
            # 결과 처리
            logger.info(f"차량번호: {result.plate_number}")
            logger.info(f"전기차 여부: {'전기차' if result.is_ev else '일반차'}")
            
            # 결과 저장 (필요한 경우)
            detector.save_results([result], "ev_detection_results.json")
            
            # 화면에 표시 (디버깅용)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"카메라 스트림 처리 중 오류 발생: {str(e)}")
        raise

def process_single_image(image_path: str, json_path: str):
    """단일 이미지 처리 (이미지 + JSON 파일)"""
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return None
            
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # JSON 데이터 구조 로깅
        logging.info(f"JSON 데이터 구조: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
        # JSON 데이터에서 첫 번째 항목 가져오기
        if not data or len(data) == 0:
            logging.error(f"JSON 파일에 데이터가 없습니다: {json_path}")
            return None
            
        plate_info = data[0]
        
        # 번호판 정보 추출
        plate_area = plate_info.get('area', {})
        plate_text = plate_info.get('text', '')
        
        if not plate_area or not plate_text:
            logging.error(f"번호판 정보가 불완전합니다: {json_path}")
            return None
            
        # EVDetector 초기화
        detector = EVDetector()
        logging.info("EVDetector 초기화 완료")
        
        # 프레임 처리
        result = detector.process_frame(image, plate_info)
        if result is None:
            logging.error("프레임 처리 중 오류 발생")
            return None
            
        # 결과 반환
        return {
            'plate_text': plate_text,
            'is_ev': result.is_ev,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        
    except Exception as e:
        logging.error(f"단일 이미지 처리 중 오류 발생: {str(e)}")
        return None

def process_image_batch(image_paths: list, plate_info_path: str):
    """배치 이미지 처리"""
    try:
        # 검출기 초기화
        detector = EVDetector("xgb_model.pkl", "lgbm_model.pkl")
        
        # 번호판 정보 로드
        with open(plate_info_path, 'r') as f:
            plate_infos = json.load(f)
        
        all_results = []
        
        # 각 이미지 처리
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"이미지를 로드할 수 없습니다: {image_path}")
                continue
                
            # 해당 이미지의 번호판 정보 찾기
            image_plate_infos = [p for p in plate_infos if p['image_path'] == image_path]
            
            if not image_plate_infos:
                logger.warning(f"번호판 정보를 찾을 수 없습니다: {image_path}")
                continue
            
            # 전기차 검출
            results = detector.process_batch(frame, image_plate_infos)
            all_results.extend(results)
        
        # 모든 결과 저장
        detector.save_results(all_results, "batch_results.json")
        
        return all_results
        
    except Exception as e:
        logger.error(f"배치 이미지 처리 중 오류 발생: {str(e)}")
        raise

def main():
    # 이미지와 JSON 파일이 있는 디렉토리
    image_dir = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG/"
    json_dir = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG_JSON/"
    
    # 결과 저장 디렉토리 생성
    results_dir = "/home/ijh/auto_llm/LLMs/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    logging.info(f"발견된 이미지 파일: {len(image_files)}개")
    logging.info(f"발견된 JSON 파일: {len(json_files)}개")
    
    # 날짜별 결과를 저장할 딕셔너리
    daily_results = {}
    
    # EVDetector 초기화
    detector = EVDetector()
    
    # 각 이미지 파일 처리
    for image_file in image_files:
        try:
            # 이미지 파일 경로
            image_path = os.path.join(image_dir, image_file)
            
            # JSON 파일 찾기
            json_file = image_file.rsplit('.', 1)[0] + '.json'
            json_path = os.path.join(json_dir, json_file)
            
            if not os.path.exists(json_path):
                logging.warning(f"JSON 파일을 찾을 수 없음: {json_file}")
                continue
            
            # JSON 파일 읽기
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data or len(data) == 0:
                logging.warning(f"JSON 데이터가 비어있음: {json_file}")
                continue
            
            # 첫 번째 번호판 정보 사용
            plate_info = data[0]
            
            # 이미지 처리
            result = detector.process_frame(image_path, plate_info)
            
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
                'timestamp': result.timestamp,
                'processing_time': result.processing_time
            })
            
            logging.info(f"처리 완료: {image_file}")
            logging.info(f"  - 번호판: {result.plate_number}")
            logging.info(f"  - 전기차 여부: {'전기차' if result.is_ev else '일반차'}")
            logging.info(f"  - 신뢰도: {result.confidence:.2f}")
            
        except Exception as e:
            logging.error(f"파일 처리 중 오류 발생: {image_file}")
            logging.error(f"오류 내용: {str(e)}")
            continue
    
    # 날짜별로 결과 저장
    for date_str, results in daily_results.items():
        output_file = os.path.join(results_dir, f"ev_detection_results_{date_str}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"결과 저장 완료: {output_file} ({len(results)}개 항목)")
    
    logging.info(f"총 처리된 파일 수: {len(image_files)}")

if __name__ == "__main__":
    main() 