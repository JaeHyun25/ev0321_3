import numpy as np
import pandas as pd
import joblib

# class EVPredictor:
#     def __init__(self, xgb_model_path, lgbm_model_path):
#         """ 저장된 모델 불러오기 """
#         self.xgb_model = joblib.load(xgb_model_path)
#         self.lgbm_model = joblib.load(lgbm_model_path)

#     def predict(self, features):
#         """ 새로운 데이터(features)로 전기차 여부 예측 """
#         xgb_pred = self.xgb_model.predict([features])[0]
#         lgb_pred = self.lgbm_model.predict([features])[0]

#         # XGBoost 신뢰도가 낮은 경우 LightGBM 보완 적용
#         xgb_prob = self.xgb_model.predict_proba([features])[0][1]
#         if xgb_prob < 0.45:  # 최적 임계값 사용
#             return lgb_pred
#         return xgb_pred

# if __name__ == "__main__":
#     predictor = EVPredictor("xgb_model.pkl", "lgbm_model.pkl")

#     # 🚗 테스트 데이터 예측
#     new_data = np.random.rand(768)  # 예제 (실제 데이터로 변경해야 함)
#     prediction = predictor.predict(new_data)

#     print(f"🚗 예측 결과: {'EV' if prediction == 1 else '일반차'}")

class EVPredictor:
    def __init__(self, xgb_model_path, lgbm_model_path):
        """ 저장된 모델 불러오기 """
        self.xgb_model = joblib.load(xgb_model_path)
        self.lgbm_model = joblib.load(lgbm_model_path)

    def predict(self, features):
        """ 새로운 데이터(features)로 전기차 여부 예측 """
        xgb_pred = self.xgb_model.predict([features])[0]
        xgb_prob = self.xgb_model.predict_proba([features])[0][1]

        # XGBoost 신뢰도가 낮은 경우 LightGBM 보완 적용
        if xgb_prob < 0.45:  
            return self.lgbm_model.predict([features])[0]
        return xgb_pred