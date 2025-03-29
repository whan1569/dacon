import os
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import logging
from src.feature_engineering import load_data, preprocess_data, create_features, train_model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, X, y):
    """
    Optuna 최적화를 위한 목적 함수
    """
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(np.unique(y)),
        'n_estimators': 1000,
        'verbose': -1,
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e-1),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 1e-1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-1)
    }
    
    # 5-fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_model(X_train, y_train, param)
        val_pred = model.predict(X_val)
        score = accuracy_score(y_val, val_pred)
        scores.append(score)
    
    return np.mean(scores)

def print_trial_progress(study, trial):
    """
    Optuna trial 진행 상황을 출력하는 콜백 함수
    """
    logger.info(f"Trial {trial.number + 1} 완료")
    logger.info(f"현재 최고 성능: {study.best_value:.5f}")
    logger.info(f"현재 최적 파라미터: {study.best_params}")

def main():
    try:
        # 학습 데이터 로드
        logger.info("학습 데이터 로드 중...")
        train_data = load_data('train')
        logger.info("테스트 데이터 로드 중...")
        test_data = load_data('test')
        
        # 데이터 전처리
        logger.info("데이터 전처리 시작...")
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)
        
        # 특성 생성
        logger.info("특성 생성 시작...")
        X_train = create_features(train_data)
        logger.info(f"생성된 특성 수: {X_train.shape[1]}")
        
        # 학습 데이터의 특성 목록 저장
        feature_list = X_train.columns.tolist()
        logger.info(f"학습 데이터 특성 수: {len(feature_list)}")
        logger.info(f"학습 데이터 특성 목록: {feature_list}")
        
        # 테스트 데이터 특성 생성 (학습 데이터와 동일한 특성만 사용)
        logger.info("테스트 데이터 특성 생성 중...")
        X_test = create_features(test_data)
        X_test = X_test[feature_list]  # 학습 데이터와 동일한 특성만 선택
        logger.info(f"테스트 데이터 특성 수: {X_test.shape[1]}")
        
        # 타겟 변수 준비
        y_train = train_data['1.회원정보']['Segment']
        logger.info(f"타겟 변수 클래스 분포: {y_train.value_counts().to_dict()}")
        
        # 이전 study가 있는지 확인
        study_path = 'optuna_study.db'
        if os.path.exists(study_path):
            logger.info("이전 study를 로드합니다...")
            study = optuna.load_study(study_name="dac_study", storage=f"sqlite:///{study_path}")
            best_params = study.best_params
            logger.info(f"이전 study의 최고 성능: {study.best_value:.5f}")
            logger.info(f"이전 study의 최적 파라미터: {best_params}")
        else:
            logger.info("새로운 study를 생성하고 하이퍼파라미터 최적화를 시작합니다...")
            study = optuna.create_study(
                study_name="dac_study",
                storage=f"sqlite:///{study_path}",
                direction='maximize'
            )
            
            # 하이퍼파라미터 최적화
            study.optimize(
                lambda trial: objective(trial, X_train, y_train),
                n_trials=50,
                n_jobs=1,  # 병렬 처리 비활성화
                callbacks=[print_trial_progress],  # 진행 상황 출력 콜백 추가
                catch=(Exception,)  # 예외 발생 시에도 study 저장
            )
            
            best_params = study.best_params
            logger.info(f"최적화 완료. 최고 성능: {study.best_value:.5f}")
            logger.info(f"최적 파라미터: {best_params}")
        
        # 최적의 하이퍼파라미터로 최종 모델 학습
        logger.info("최종 모델 학습 시작...")
        best_params.update({
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'n_estimators': 1000,
            'verbose': -1
        })
        
        logger.info(f"최종 모델 파라미터: {best_params}")
        
        final_model = train_model(X_train, y_train, best_params)
        logger.info("최종 모델 학습 완료")
        
        # 테스트 데이터 예측
        logger.info("테스트 데이터 예측 시작...")
        test_pred = final_model.predict(X_test)
        logger.info(f"예측 완료. 예측값 분포: {pd.Series(test_pred).value_counts().to_dict()}")
        
        # 제출 파일 생성
        logger.info("제출 파일 생성 중...")
        submission = pd.DataFrame({
            'ID': test_data['1.회원정보']['ID'],
            'Segment': test_pred
        })
        
        # 제출 파일 저장
        submission.to_csv('submission.csv', index=False)
        logger.info("제출 파일 생성 완료")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 