import cv2
import numpy as np
from PIL import Image
import json
import joblib
import xgboost as xgb
import lightgbm as lgb
import os
import logging
from typing import List, Tuple, Dict, Union
import time

class EVClassifier:
    def __init__(self, xgb_model_path: str, lgb_model_path: str):
        """ 전기차 판별 모델 로드 """
        self.xgb_model = joblib.load(xgb_model_path)
        self.lgb_model = joblib.load(lgb_model_path)
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: np.ndarray, crop_box: Tuple[int, int, int, int], angle: float) -> np.ndarray:
        """ 이미지 전처리 (크롭, 리사이즈, 회전) """
        try:
            x, y, w, h = crop_box
            cropped = image[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (320, 180))
            
            if angle != 0:
                center = (resized.shape[1]//2, resized.shape[0]//2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                resized = cv2.warpAffine(resized, matrix, (resized.shape[1], resized.shape[0]))
            
            return cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            raise

    def extract_features(self, hsv_image: np.ndarray) -> np.ndarray:
        """ HSV 히스토그램 특징 추출 """
        try:
            h, s, v = cv2.split(hsv_image)
            hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
            hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
            hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
            
            return np.r_[hist_h, hist_s, hist_v].squeeze()
        except Exception as e:
            self.logger.error(f"특징 추출 중 오류 발생: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray, plate_info: Dict) -> Tuple[bool, float]:
        """ 단일 프레임 처리 및 예측 """
        try:
            start_time = time.time()
            
            # 번호판 정보 추출
            area = plate_info['area']
            crop_box = (area['x'], area['y'], area['width'], area['height'])
            
            # 이미지 전처리
            hsv_image = self.preprocess_image(frame, crop_box, area['angle'])
            
            # 특징 추출
            features = self.extract_features(hsv_image)
            
            # 예측
            xgb_pred = self.xgb_model.predict([features])[0]
            xgb_prob = self.xgb_model.predict_proba([features])[0][1]
            
            # 신뢰도 기반 앙상블
            if xgb_prob < 0.45:
                prediction = self.lgb_model.predict([features])[0]
            else:
                prediction = xgb_pred
            
            elapsed_time = time.time() - start_time
            return bool(prediction), elapsed_time
            
        except Exception as e:
            self.logger.error(f"프레임 처리 중 오류 발생: {str(e)}")
            raise

    def process_batch(self, frames: List[np.ndarray], plate_infos: List[Dict]) -> List[Tuple[bool, float]]:
        """ 여러 프레임 일괄 처리 """
        results = []
        for frame, plate_info in zip(frames, plate_infos):
            result = self.process_frame(frame, plate_info)
            results.append(result)
        return results
