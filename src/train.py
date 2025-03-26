import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from feature_engineering import preprocess_features, create_features, prepare_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    모델을 학습시키는 함수
    
    Args:
        X_train (pd.DataFrame): 학습 데이터 특성
        y_train (pd.Series): 학습 데이터 타겟
        
    Returns:
        RandomForestClassifier: 학습된 모델
    """
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("모델 학습 완료")
        return model
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {str(e)}")
        raise

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    모델을 평가하는 함수
    
    Args:
        model (RandomForestClassifier): 학습된 모델
        X_test (pd.DataFrame): 테스트 데이터 특성
        y_test (pd.Series): 테스트 데이터 타겟
        
    Returns:
        Dict[str, float]: 평가 지표
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"모델 정확도: {accuracy:.4f}")
        return report
        
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {str(e)}")
        raise

def main():
    try:
        # 데이터 로드
        data = pd.read_csv('data/raw/data.csv')
        
        # 데이터 전처리
        data = preprocess_features(data)
        
        # 특성 엔지니어링
        data = create_features(data)
        
        # 데이터 준비
        X, y = prepare_data(data, target_column='target')
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        model = train_model(X_train, y_train)
        
        # 모델 평가
        metrics = evaluate_model(model, X_test, y_test)
        
        logger.info("학습 완료")
        
    except Exception as e:
        logger.error(f"학습 과정 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 