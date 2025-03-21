import logging
from datetime import datetime
from .ev_detector import EVDetector
from ..utils.logging_config import setup_logging
from typing import Dict, Optional

class EVDetectorSingleton:
    _instance = None
    _detector: Optional[EVDetector] = None
    _config: Optional[Dict] = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EVDetectorSingleton, cls).__new__(cls)
        return cls._instance

    def initialize(self, config: Dict):
        """detector 초기화"""
        if self._detector is None:
            self._config = config
            self._logger = setup_logging(config['paths']['logs_dir'])
            self._detector = EVDetector(
                config['model']['xgb_path'],
                config['model']['lgbm_path']
            )
            self._logger.info("EVDetector 초기화 완료")

    def process_image(self, frame, plate_info: Dict) -> Dict:
        """이미지 처리
        
        Args:
            frame: np.ndarray - 입력 이미지
            plate_info: dict - 번호판 정보
            
        Returns:
            dict - 처리 결과
            
        Raises:
            RuntimeError: detector가 초기화되지 않은 경우
            ValueError: 입력 데이터가 올바르지 않은 경우
        """
        if self._detector is None:
            raise RuntimeError("EVDetector가 초기화되지 않았습니다.")
            
        # area 정보 변환
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
        result = self._detector.process_frame(frame, area_info)
        
        # 결과 생성
        return {
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

    @property
    def is_initialized(self) -> bool:
        """초기화 여부 확인"""
        return self._detector is not None 