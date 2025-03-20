import cv2
import numpy as np
from PIL import Image
import json
import joblib
import xgboost as xgb
import lightgbm as lgb
import os

class EVClassifier:
    def __init__(self, xgb_model_path, lgb_model_path):
        """ 전기차 판별 모델 로드 """
        self.xgb_model = joblib.load(xgb_model_path)
        self.lgb_model = joblib.load(lgb_model_path)

    def feature_extraction_image(self, image_path, crop_list):
        """ 이미지에서 특징 추출 (HSV 히스토그램) """
        image = Image.open(image_path)
        x, y, width, height, angle = crop_list
        crop_box = (x, y, x + width, y + height)

        cropped_image = image.crop(crop_box).resize((320, 180)).rotate(angle)
        cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2HSV)

        h_channel, s_channel, v_channel = cv2.split(cropped_image_cv)
        hist_h = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v_channel], [0], None, [256], [0, 256])

        fv = np.r_[hist_h, hist_s, hist_v].squeeze()
        return fv

    def predict(self, image_path, json_path):
        """ 이미지 + JSON을 입력받아 전기차 여부 예측 """
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        if isinstance(json_data, list) and len(json_data) > 0:
            json_data = json_data[0]

        try:
            x, y, width, height, angle = (
                json_data['area']['x'],
                json_data['area']['y'],
                json_data['area']['width'],
                json_data['area']['height'],
                json_data['area']['angle'],
            )
        except KeyError:
            raise ValueError("🚨 JSON 파일 형식 오류!")

        crop_list = [x, y, width, height, angle]

        # HSV 특징 추출
        fv = self.feature_extraction_image(image_path, crop_list)

        # XGBoost & LightGBM 예측
        xgb_pred = self.xgb_model.predict([fv])[0]
        lgb_pred = self.lgb_model.predict([fv])[0]

        # 평균 앙상블 (가중치는 필요하면 조절 가능)
        final_pred = (xgb_pred + lgb_pred) / 2
        is_ev = int(final_pred > 0.5)  # 0.5 이상이면 전기차(1), 아니면 일반차(0)

        return is_ev
