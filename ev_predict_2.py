import numpy as np
import pandas as pd
import joblib
import logging
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass
from datetime import datetime

<<<<<<< HEAD
@dataclass
class PredictionResult:
    is_ev: bool
    confidence: float
    elapsed_time: float
    timestamp: datetime
    plate_number: str

=======
>>>>>>> 38b40d6c1846650cc251e70923d2ccd1510e0ed7
class EVPredictor:
    def __init__(self, xgb_model_path: str, lgbm_model_path: str):
        """ 저장된 모델 불러오기 """
        self.xgb_model = joblib.load(xgb_model_path)
        self.lgbm_model = joblib.load(lgbm_model_path)
        self.logger = logging.getLogger(__name__)

    def predict_single(self, features: np.ndarray, plate_number: str) -> PredictionResult:
        """ 단일 예측 수행 """
        try:
            start_time = time.time()
            
            # XGBoost 예측
            xgb_pred = self.xgb_model.predict([features])[0]
            xgb_prob = self.xgb_model.predict_proba([features])[0][1]
            
            # 신뢰도 기반 앙상블
            if xgb_prob < 0.45:
                prediction = self.lgbm_model.predict([features])[0]
                confidence = self.lgbm_model.predict_proba([features])[0][1]
            else:
                prediction = xgb_pred
                confidence = xgb_prob
            
            elapsed_time = time.time() - start_time
            
            return PredictionResult(
                is_ev=bool(prediction),
                confidence=float(confidence),
                elapsed_time=elapsed_time,
                timestamp=datetime.now(),
                plate_number=plate_number
            )
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            raise

    def predict_batch(self, features_list: List[np.ndarray], plate_numbers: List[str]) -> List[PredictionResult]:
        """ 여러 예측 일괄 수행 """
        results = []
        for features, plate_number in zip(features_list, plate_numbers):
            result = self.predict_single(features, plate_number)
            results.append(result)
        return results

<<<<<<< HEAD
    def save_results(self, results: List[PredictionResult], output_path: str):
        """ 예측 결과 저장 """
        try:
            # 결과를 딕셔너리 리스트로 변환
            results_dict = [
                {
                    'plate_number': r.plate_number,
                    'is_ev': r.is_ev,
                    'confidence': r.confidence,
                    'elapsed_time': r.elapsed_time,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in results
            ]
            
            # JSON 파일로 저장
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"결과가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {str(e)}")
            raise
=======
        # XGBoost 신뢰도가 낮은 경우 LightGBM 보완 적용
        if xgb_prob < 0.45:  
            return self.lgbm_model.predict([features])[0]
        return xgb_pred
>>>>>>> 38b40d6c1846650cc251e70923d2ccd1510e0ed7
