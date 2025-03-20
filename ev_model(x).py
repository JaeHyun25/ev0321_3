import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

def train_ev_model(data_path, xgb_model_path, lgbm_model_path):
    """ 전기차 판별 모델 학습 및 저장 """
    df = pd.read_csv(data_path)
    X = df.drop(columns=["carname", "folder", "ev"])
    y = df["ev"]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost 모델 학습
    xgb_model = XGBClassifier(
        scale_pos_weight=653.94,  # 실험 결과 가장 좋은 값
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # LightGBM 모델 학습
    lgbm_model = LGBMClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31
    )
    lgbm_model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(xgb_model, xgb_model_path)
    joblib.dump(lgbm_model, lgbm_model_path)

    # 성능 평가
    y_pred = xgb_model.predict(X_test)
    print("\n🚀 XGBoost 모델 성능:")
    print(classification_report(y_test, y_pred))
    print("혼동 행렬:", confusion_matrix(y_test, y_pred))

    y_pred_final = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    low_confidence_cases = y_prob_xgb < 0.45  # 최적 임계값
    y_pred_final[low_confidence_cases] = lgbm_model.predict(X_test[low_confidence_cases])

    print("\n🚀 최종 모델 성능:")
    print(classification_report(y_test, y_pred_final))
    print("혼동 행렬:", confusion_matrix(y_test, y_pred_final))

if __name__ == "__main__":
    train_ev_model("ev_data.csv", "xgb_model.pkl", "lgbm_model.pkl")
