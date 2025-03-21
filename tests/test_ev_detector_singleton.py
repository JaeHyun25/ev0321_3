import unittest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.detector.ev_detector_singleton import EVDetectorSingleton

class TestEVDetectorSingleton(unittest.TestCase):
    def setUp(self):
        # 테스트용 설정
        self.config = {
            'paths': {
                'logs_dir': '/tmp/test_logs'
            },
            'model': {
                'xgb_path': '/tmp/test_models/xgb_model.json',
                'lgbm_path': '/tmp/test_models/lgbm_model.txt'
            }
        }
        
        # 테스트용 이미지와 번호판 정보
        self.test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.test_plate_info = {
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
        
        # 싱글톤 인스턴스 초기화
        self.detector = EVDetectorSingleton()
    
    def tearDown(self):
        # 테스트 후 싱글톤 인스턴스 초기화
        EVDetectorSingleton._instance = None
        EVDetectorSingleton._detector = None
        EVDetectorSingleton._config = None
        EVDetectorSingleton._logger = None
    
    def test_singleton_pattern(self):
        """싱글톤 패턴 테스트"""
        detector1 = EVDetectorSingleton()
        detector2 = EVDetectorSingleton()
        self.assertIs(detector1, detector2)
    
    def test_initialization(self):
        """초기화 테스트"""
        with patch('src.detector.ev_detector_singleton.EVDetector') as mock_detector:
            self.detector.initialize(self.config)
            mock_detector.assert_called_once_with(
                self.config['model']['xgb_path'],
                self.config['model']['lgbm_path']
            )
            self.assertTrue(self.detector.is_initialized)
    
    def test_process_image_without_initialization(self):
        """초기화되지 않은 상태에서 이미지 처리 테스트"""
        with self.assertRaises(RuntimeError):
            self.detector.process_image(self.test_frame, self.test_plate_info)
    
    def test_process_image(self):
        """이미지 처리 테스트"""
        # EVDetector 목 객체 생성
        mock_result = MagicMock()
        mock_result.is_ev = True
        mock_result.confidence = 0.95
        mock_result.processing_time = 0.1
        mock_result.plate_number = "01너3346"
        mock_result.timestamp = datetime.now()
        
        with patch('src.detector.ev_detector_singleton.EVDetector') as mock_detector_class:
            # 목 객체 설정
            mock_detector = mock_detector_class.return_value
            mock_detector.process_frame.return_value = mock_result
            
            # 초기화
            self.detector.initialize(self.config)
            
            # 이미지 처리
            result = self.detector.process_image(self.test_frame, self.test_plate_info)
            
            # 결과 검증
            self.assertIsInstance(result, dict)
            self.assertEqual(result['text'], "01너3346")
            self.assertTrue(result['ev'])
            self.assertEqual(result['conf']['ev'], 0.95)
            self.assertEqual(result['elapsed'], 0.1)
            
            # area 정보 검증
            self.assertEqual(result['area']['angle'], self.test_plate_info['area']['angle'])
            self.assertEqual(result['area']['height'], self.test_plate_info['area']['height'])
            self.assertEqual(result['area']['width'], self.test_plate_info['area']['width'])
            self.assertEqual(result['area']['x'], self.test_plate_info['area']['x'])
            self.assertEqual(result['area']['y'], self.test_plate_info['area']['y'])

if __name__ == '__main__':
    unittest.main() 