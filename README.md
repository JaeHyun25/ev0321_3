# 전기차 번호판 감지 시스템

이 프로젝트는 실시간으로 전기차 번호판을 감지하고 분류하는 시스템입니다.

## 주요 기능

- 실시간 번호판 이미지 처리
- 전기차/일반차 분류
- 에러 케이스 및 불확실한 판정 결과 저장
- 로깅 시스템

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/JaeHyun25/ev0321_3.git
cd ev0321_3
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 설정 파일 확인
- `config/config.yaml` 파일에서 필요한 설정을 확인하고 수정합니다.

2. 실행
```bash
python main.py
```

## 프로젝트 구조

```
ev_detector_core/
├── main.py              # 메인 실행 파일
├── config/
│   └── config.yaml     # 설정 파일
├── src/
│   ├── detector/       # 전기차 감지 모듈
│   │   ├── ev_detector.py           # 기본 detector 클래스
│   │   ├── ev_detector_singleton.py  # 싱글톤 패턴 구현
│   │   └── ev_classifier.py         # 전기차 분류기
│   ├── models/        # 학습된 모델 파일
│   │   ├── xgb_model.pkl
│   │   └── lgbm_model.pkl
│   └── utils/         # 유틸리티 함수
│       ├── logging_config.py        # 로깅 설정
│       └── image_processing.py      # 이미지 처리 유틸리티
├── logs/              # 로그 파일
├── error_cases/       # 에러 케이스
├── uncertain_cases/   # 불확실한 판정 결과
└── requirements.txt   # 의존성 관리
```

## 설정

`config.yaml` 파일에서 다음 설정을 관리할 수 있습니다:

- 모델 파일 경로
- 로그 디렉토리
- 에러 케이스 저장 디렉토리
- 불확실한 판정 결과 저장 디렉토리
- 실시간 처리 관련 설정
- 이미지 처리 설정

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 