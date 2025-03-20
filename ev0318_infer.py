import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# 🚀 데이터 로드 및 전처리 (사용자가 데이터 로드 필요)
def load_data(df):
    X = df.drop(columns=["carname", "folder", "ev"])
    y = df["ev"]
    return X, y

# 🚀 모델 학습 함수
def train_model(df):
    X, y = load_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost 초기 학습 및 FN 분석
    initial_xgb = XGBClassifier(random_state=42)
    initial_xgb.fit(X_train, y_train)
    initial_pred = initial_xgb.predict(X_test)
    fn_ratio = sum((y_test == 1) & (initial_pred == 0)) / len(X_train)
    scale_pos_weight = 1 / fn_ratio

    # 🚀 최적화된 XGBoost 모델
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 🚀 최적화된 LightGBM 모델
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.11,
        num_leaves=30,
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(lgbm_model, "lgbm_model.pkl")
    print("✅ 모델이 저장되었습니다: xgb_model.pkl, lgbm_model.pkl")

    return xgb_model, lgbm_model, X_test, y_test

# 🚀 Inference 함수
def inference(new_data):
    xgb_model = joblib.load("xgb_model.pkl")
    lgbm_model = joblib.load("lgbm_model.pkl")
    
    y_prob_xgb = xgb_model.predict_proba(new_data)[:, 1]
    best_threshold = 0.45
    low_confidence_cases = y_prob_xgb < best_threshold
    
    y_pred_final = xgb_model.predict(new_data)
    y_pred_final[low_confidence_cases] = lgbm_model.predict(new_data[low_confidence_cases])
    
    return y_pred_final

# 🚀 테스트 실행 (사용자 데이터 필요)
if __name__ == "__main__":
    # 사용자 데이터 로드 필요 (df를 파일에서 로드하도록 수정 가능)
    df = pd.read_csv("data.csv")  # 사용자의 데이터 파일로 변경
    xgb_model, lgbm_model, X_test, y_test = train_model(df)
    
    # 모델 성능 평가
    y_pred_final = inference(X_test)
    print("\n🚀 최적 모델 성능:")
    print(classification_report(y_test, y_pred_final))
    print("\n혼동 행렬:")
    print(confusion_matrix(y_test, y_pred_final))
    print(f"\nPrecision: {precision_score(y_test, y_pred_final):.6f}")
    print(f"Recall: {recall_score(y_test, y_pred_final):.6f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.6f}")