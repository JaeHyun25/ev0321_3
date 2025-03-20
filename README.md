# 전기차 감지 시스템

## 소개
이미지에서 전기차를 감지하는 시스템입니다. HSV 히스토그램 특징을 사용하여 전기차와 일반차를 구분합니다.

## 설치 방법
1. 저장소 클론
```bash
git clone [repository-url]
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법
1. 설정 파일 수정 (`config/config.yaml`)
2. 실행
```bash
python main.py
```

## 프로젝트 구조
```
/home/ijh/auto_llm/LLMs/
├── src/
│   ├── detector/      # 감지 관련 코드
│   │   ├── ev_detector.py
│   │   └── ev_classifier.py
│   └── utils/         # 유틸리티 함수
│       ├── image_processing.py
│       └── logging_config.py
├── config/            # 설정 파일
│   └── config.yaml
├── models/            # 모델 파일
│   ├── xgb_model.pkl
│   └── lgbm_model.pkl
├── results/           # 결과 파일
├── logs/              # 로그 파일
└── tests/             # 테스트 코드
```

## 주요 기능
- HSV 히스토그램 기반 특징 추출
- XGBoost와 LightGBM 앙상블 모델
- 날짜별 결과 저장
- 상세한 로깅 시스템

## 성능
- 정확도: 98%
- 재현율: 100%
- 정밀도: 66.67%
- F1 점수: 80%

## 라이선스
MIT License 