import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    높은 결측치 비율을 가진 컬럼 제거
    
    Args:
        df: 입력 데이터프레임
        threshold: NaN 비율 임계값 (기본값 0.5 = 50%)
    
    Returns:
        결측치가 많은 컬럼이 제거된 데이터프레임
    """
    missing_ratio = df.isnull().sum() / len(df)
    return df.loc[:, missing_ratio < threshold]

def impute_missing_values(df: pd.DataFrame, method: str = 'linear', n_neighbors: int = 5) -> pd.DataFrame:
    """
    시계열 데이터의 결측치 처리
    
    Args:
        df: 입력 데이터프레임
        method: 보간 방법 ('linear', 'knn', 'locf', 'nocb')
        n_neighbors: KNN 방법 사용시 이웃 개수
    
    Returns:
        결측치가 처리된 데이터프레임
    """
    df_imputed = df.copy()
    
    if method == 'linear':
        # 선형 보간법
        df_imputed = df_imputed.interpolate(method='linear')
    
    elif method == 'knn':
        # KNN 기반 결측치 대체
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed.loc[:, :] = imputer.fit_transform(df_imputed)
    
    elif method == 'locf':
        # 이전 값으로 채우기
        df_imputed = df_imputed.fillna(method='ffill')
    
    elif method == 'nocb':
        # 다음 값으로 채우기
        df_imputed = df_imputed.fillna(method='bfill')
    
    # 남은 결측치 처리 (시작과 끝 부분)
    df_imputed = df_imputed.fillna(method='ffill').fillna(method='bfill')
    
    return df_imputed

def process_sensor_data(df: pd.DataFrame,
                       missing_threshold: float = 0.5,
                       impute_method: str = 'knn',
                       n_neighbors: int = 5) -> pd.DataFrame:
    """
    센서 데이터 전처리 함수
    
    Args:
        df: 입력 데이터프레임
        missing_threshold: 결측치 비율이 이 값을 넘는 컬럼은 제거
        impute_method: 결측치 처리 방법 ('knn', 'linear', 'ffill')
        n_neighbors: KNN 방법 사용시 이웃 개수
    
    Returns:
        전처리된 데이터프레임
    """
    # 결측치가 많은 컬럼 제거
    missing_ratio = df.isnull().sum() / len(df)
    df_cleaned = df.loc[:, missing_ratio < missing_threshold]
    
    # 남은 결측치 처리
    if impute_method == 'knn':
        # KNN 기반 결측치 대체
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_cleaned),
            columns=df_cleaned.columns,
            index=df_cleaned.index
        )
    
    elif impute_method == 'linear':
        # 선형 보간
        df_imputed = df_cleaned.interpolate(method='linear', axis=0)
    
    elif impute_method == 'ffill':
        # 이전 값으로 채우기
        df_imputed = df_cleaned.fillna(method='ffill')
        # 첫 부분의 NaN은 다음 값으로 채우기
        df_imputed = df_imputed.fillna(method='bfill')
    
    else:
        raise ValueError(f"지원하지 않는 결측치 처리 방법입니다: {impute_method}")
    
    return df_imputed

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드 예시
    # df = pd.read_csv("sensor_data.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    # 전처리 파이프라인 실행
    # df_processed = process_sensor_data(df, 
    #                                   missing_threshold=0.5,
    #                                   impute_method='knn',
    #                                   n_neighbors=5)
    pass 