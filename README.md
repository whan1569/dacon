# DAC 2024 대회 프로젝트

## 프로젝트 개요
이 프로젝트는 신용카드 고객의 다양한 금융 활동 데이터를 분석하여 고객 세그먼트를 예측하는 AI 모델을 개발합니다.

## 평가 지표
- Macro F1-Score
- Public Score: 전체 테스트 데이터 중 50%
- Private Score: 전체 테스트 데이터 중 100%

## 프로젝트 구조
```
dac/
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   └── train.py
├── train/
│   ├── 1.회원정보/
│   ├── 2.신용정보/
│   ├── 3.승인매출정보/
│   ├── 4.청구입금정보/
│   ├── 5.잔액정보/
│   ├── 6.채널정보/
│   ├── 7.마케팅정보/
│   └── 8.성과정보/
├── test/
│   └── (train과 동일한 구조)
├── main.py
├── requirements.txt
└── README.md
```

## 데이터 구조
- 회원정보: 고객의 기본 정보
- 신용정보: 신용 관련 정보
- 승인매출정보: 승인된 매출 정보
- 청구입금정보: 청구 및 입금 정보
- 잔액정보: 계좌 잔액 정보
- 채널정보: 거래 채널 정보
- 마케팅정보: 마케팅 관련 정보
- 성과정보: 성과 관련 정보

## 주요 기능

### 1. 데이터 전처리 (`feature_engineering.py`)
- 결측치 처리
- 데이터 타입 변환
- 특성 생성
- 특성 선택
- 데이터 스케일링

### 2. 모델 학습 (`train.py`)
- LightGBM 모델 사용
- 교차 검증 (5-fold)
- 하이퍼파라미터 최적화 (Optuna)
- 모델 평가 및 저장

### 3. 하이퍼파라미터 최적화
- Optuna를 사용한 자동 최적화
- 최적화 대상 파라미터:
  - num_leaves
  - learning_rate
  - feature_fraction
  - bagging_fraction
  - min_child_samples
  - min_child_weight
  - min_split_gain
  - reg_alpha
  - reg_lambda

### 4. 특성 엔지니어링
- 각 데이터 폴더별 특성 생성
- 결측치 처리 (평균값으로 대체)
- 수치형 특성 스케일링
- 특성 선택 및 필터링

## 실행 방법
1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 모델 학습 및 예측:
```bash
python main.py
```

## 주요 결과
- 최고 성능: 0.85911 (Optuna 최적화 결과)
- 사용 모델: LightGBM (다중 분류)
- 평가 지표: Accuracy

## 주의사항
- 데이터는 parquet 형식으로 제공됨
- 각 폴더별로 여러 개의 parquet 파일이 존재할 수 있음
- 특성 생성 시 모든 데이터 폴더의 정보를 활용
- Optuna study는 SQLite 데이터베이스에 자동 저장됨 