import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import optuna
from src.feature_engineering import preprocess_data, create_features
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_type='train'):
    """
    데이터를 로드하는 함수
    """
    data = {}
    base_path = os.path.join(os.getcwd(), data_type)
    
    # 각 폴더별로 데이터 로드
    folders = ['1.회원정보', '2.신용정보', '3.승인매출정보', '4.청구입금정보', 
              '5.잔액정보', '6.채널정보', '7.마케팅정보', '8.성과정보']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            # 폴더 내의 모든 parquet 파일 로드
            dfs = []
            for file in os.listdir(folder_path):
                if file.endswith('.parquet'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_parquet(file_path)
                        print(f"Loaded {file} with columns: {df.columns.tolist()}")
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
            
            if dfs:
                # 같은 폴더의 데이터프레임들을 하나로 합치기
                data[folder] = pd.concat(dfs, axis=0).reset_index(drop=True)
                print(f"Successfully loaded {folder} with shape: {data[folder].shape}")
            else:
                print(f"No parquet files found in {folder}")
    
    return data

def train_model(X, y, params):
    """
    모델을 학습하는 함수
    """
    model = LGBMClassifier(**params)
    model.fit(X, y)
    return model

def objective(trial, X_train, y_train):
    """
    Optuna를 위한 목적 함수
    """
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e-1, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 1e-1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e-1, log=True),
        'n_estimators': 500,
        'verbose': -1,
        'colsample_bytree': None,  # feature_fraction과 중복되므로 제거
        'subsample': None,  # bagging_fraction과 중복되므로 제거
        'subsample_freq': None  # bagging_freq와 중복되므로 제거
    }
    
    # 교차 검증
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]
        y_val = y_train.iloc[valid_idx]
        
        model = train_model(X_tr, y_tr, param)
        pred = model.predict_proba(X_val)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores)

def main():
    try:
        # 학습 데이터 로드
        train_data = load_data('train')
        test_data = load_data('test')
        
        # 데이터 전처리
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)
        
        # 특성 생성
        X_train = create_features(train_data)
        X_test = create_features(test_data)
        
        # 타겟 변수 준비
        y_train = train_data['1.회원정보']['Segment']
        
        # 하이퍼파라미터 최적화 (병렬 처리 활성화)
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=50,
            n_jobs=-1  # 모든 CPU 코어 사용
        )
        
        # 최적의 하이퍼파라미터로 최종 모델 학습
        best_params = study.best_params
        best_params.update({
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'n_estimators': 1000,
            'verbose': -1
        })
        
        final_model = train_model(X_train, y_train, best_params)
        
        # 테스트 데이터 예측
        test_pred = final_model.predict(X_test)
        
        # 제출 파일 생성
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