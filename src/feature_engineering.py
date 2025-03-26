import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리를 수행하는 함수
    
    Args:
        df (pd.DataFrame): 전처리할 데이터프레임
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    try:
        # 결측치 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        # 스케일링
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        logger.info("데이터 전처리 완료")
        return df
        
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        raise

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    특성 엔지니어링을 수행하는 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    try:
        # 결측치 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        # 스케일링
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # 특성 생성
        df['feature1'] = df['feature1'] * df['feature2']
        df['feature2'] = df['feature2'] ** 2
        
        logger.info("특성 엔지니어링 완료")
        return df
        
    except Exception as e:
        logger.error(f"특성 엔지니어링 중 오류 발생: {str(e)}")
        raise

def select_features(df: pd.DataFrame, target_column: str) -> List[str]:
    """
    특성 선택을 수행하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        target_column (str): 타겟 컬럼명
        
    Returns:
        List[str]: 선택된 특성 리스트
    """
    try:
        # 특성 선택 로직 구현
        selected_features = [col for col in df.columns if col != target_column]
        logger.info(f"선택된 특성: {selected_features}")
        return selected_features
        
    except Exception as e:
        logger.error(f"특성 선택 중 오류 발생: {str(e)}")
        raise

def prepare_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    데이터 준비를 수행하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        target_column (str): 타겟 컬럼명
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: 특성 데이터프레임과 타겟 시리즈
    """
    try:
        # 특성 선택
        selected_features = select_features(df, target_column)
        
        # 특성과 타겟 분리
        X = df[selected_features]
        y = df[target_column]
        
        logger.info("데이터 준비 완료")
        return X, y
        
    except Exception as e:
        logger.error(f"데이터 준비 중 오류 발생: {str(e)}")
        raise

def preprocess_data(data):
    """
    데이터 전처리 함수
    
    Args:
        data (dict): 각 데이터 폴더의 DataFrame을 포함하는 딕셔너리
    
    Returns:
        dict: 전처리된 데이터 딕셔너리
    """
    try:
        processed_data = {}
        
        # 1. 회원정보
        if '1.회원정보' in data:
            member_info = data['1.회원정보'].copy()
            member_info['ID'] = member_info['ID'].astype('str')
            member_info['연령'] = member_info['연령'].astype('str')
            processed_data['1.회원정보'] = member_info

        # 2. 신용정보
        if '2.신용정보' in data:
            credit_info = data['2.신용정보'].copy()
            credit_info['ID'] = credit_info['ID'].astype('str')
            amount_columns = ['최초한도금액', '카드이용한도금액', 'CA한도금액']
            for col in amount_columns:
                if col in credit_info.columns:
                    credit_info[col] = credit_info[col].astype('float64')
            processed_data['2.신용정보'] = credit_info

        # 3. 승인매출정보
        if '3.승인매출정보' in data:
            sales_info = data['3.승인매출정보'].copy()
            sales_info['ID'] = sales_info['ID'].astype('str')
            count_columns = ['이용건수_신용_B0M', '이용건수_신판_B0M', '이용건수_일시불_B0M',
                           '이용건수_할부_B0M', '이용건수_CA_B0M', '이용건수_체크_B0M',
                           '이용건수_카드론_B0M']
            for col in count_columns:
                if col in sales_info.columns:
                    sales_info[col] = sales_info[col].astype('int64')
            processed_data['3.승인매출정보'] = sales_info

        # 4. 청구정보
        if '4.청구정보' in data:
            billing_info = data['4.청구정보'].copy()
            billing_info['ID'] = billing_info['ID'].astype('str')
            amount_columns = ['청구금액_B0M', '입금금액_B0M', '미납금액_B0M', '연체금액_B0M']
            for col in amount_columns:
                if col in billing_info.columns:
                    billing_info[col] = billing_info[col].astype('float64')
            processed_data['4.청구정보'] = billing_info

        # 5. 잔액정보
        if '5.잔액정보' in data:
            balance_info = data['5.잔액정보'].copy()
            balance_info['ID'] = balance_info['ID'].astype('str')
            amount_columns = ['잔액_일시불_B0M', '잔액_할부_B0M', '잔액_현금서비스_B0M',
                            '잔액_카드론_B0M', '연체잔액_B0M']
            for col in amount_columns:
                if col in balance_info.columns:
                    balance_info[col] = balance_info[col].astype('float64')
            processed_data['5.잔액정보'] = balance_info

        # 6. 채널정보
        if '6.채널정보' in data:
            channel_info = data['6.채널정보'].copy()
            channel_info['ID'] = channel_info['ID'].astype('str')
            visit_columns = ['인입횟수_ARS_R6M', '방문횟수_PC_R6M', '방문횟수_앱_R6M', '방문횟수_모바일웹_R6M']
            for col in visit_columns:
                if col in channel_info.columns:
                    channel_info[col] = channel_info[col].astype('str')
            if '상담건수_B0M' in channel_info.columns:
                channel_info['상담건수_B0M'] = channel_info['상담건수_B0M'].astype('int64')
            processed_data['6.채널정보'] = channel_info

        # 7. 마케팅정보
        if '7.마케팅정보' in data:
            marketing_info = data['7.마케팅정보'].copy()
            marketing_info['ID'] = marketing_info['ID'].astype('str')
            count_columns = ['컨택건수_카드론_TM_B0M', '컨택건수_리볼빙_TM_B0M',
                           '컨택건수_CA_TM_B0M', '컨택건수_이용유도_TM_B0M']
            for col in count_columns:
                if col in marketing_info.columns:
                    marketing_info[col] = marketing_info[col].astype('int64')
            campaign_columns = ['캠페인접촉건수_R12M', '캠페인접촉일수_R12M']
            for col in campaign_columns:
                if col in marketing_info.columns:
                    marketing_info[col] = marketing_info[col].astype('str')
            processed_data['7.마케팅정보'] = marketing_info

        # 8. 성과정보
        if '8.성과정보' in data:
            performance_info = data['8.성과정보'].copy()
            performance_info['ID'] = performance_info['ID'].astype('str')
            rate_columns = ['증감율_이용건수_신용_전월', '증감율_이용건수_신판_전월',
                          '증감율_이용건수_일시불_전월', '증감율_이용건수_할부_전월',
                          '증감율_이용건수_CA_전월', '혜택수혜율_R3M', '혜택수혜율_B0M']
            for col in rate_columns:
                if col in performance_info.columns:
                    performance_info[col] = performance_info[col].astype('float64')
            processed_data['8.성과정보'] = performance_info

        logger.info("데이터 전처리 완료")
        return processed_data
        
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        raise

def create_features(data):
    """
    각 데이터 폴더에서 특성을 생성하고 결합합니다.
    
    Args:
        data (dict): 각 데이터 폴더의 DataFrame을 포함하는 딕셔너리
    
    Returns:
        pd.DataFrame: 생성된 특성들이 포함된 DataFrame
    """
    try:
        features = pd.DataFrame()
        
        # 회원정보 특성
        if '1.회원정보' in data:
            member_info = data['1.회원정보']
            member_features = pd.DataFrame()
            member_features['ID'] = member_info['ID']
            member_features['남녀구분코드'] = member_info['남녀구분코드'].astype('int64')
            member_features['회원여부_이용가능'] = member_info['회원여부_이용가능'].astype('int64')
            member_features['소지카드수_유효_신용'] = member_info['소지카드수_유효_신용'].astype('int64')
            features = pd.concat([features, member_features], axis=1)
        
        # 신용정보 특성
        if '2.신용정보' in data:
            credit_info = data['2.신용정보']
            credit_features = pd.DataFrame()
            credit_features['ID'] = credit_info['ID']
            credit_features['최초한도금액'] = credit_info['최초한도금액'].astype('float64')
            credit_features['카드이용한도금액'] = credit_info['카드이용한도금액'].astype('float64')
            credit_features['CA한도금액'] = credit_info['CA한도금액'].astype('float64')
            credit_features['시장단기연체여부_R6M'] = credit_info['시장단기연체여부_R6M'].astype('int64')
            credit_features['시장단기연체여부_R3M'] = credit_info['시장단기연체여부_R3M'].astype('int64')
            credit_features['시장연체상환여부_R6M'] = credit_info['시장연체상환여부_R6M'].astype('int64')
            credit_features['시장연체상환여부_R3M'] = credit_info['시장연체상환여부_R3M'].astype('int64')
            features = pd.concat([features, credit_features], axis=1)
        
        # 승인매출정보 특성
        if '3.승인매출정보' in data:
            sales_info = data['3.승인매출정보']
            sales_features = pd.DataFrame()
            sales_features['ID'] = sales_info['ID']
            sales_features['이용건수_신용_B0M'] = sales_info['이용건수_신용_B0M'].astype('int64')
            sales_features['이용건수_신판_B0M'] = sales_info['이용건수_신판_B0M'].astype('int64')
            sales_features['이용건수_일시불_B0M'] = sales_info['이용건수_일시불_B0M'].astype('int64')
            sales_features['이용건수_할부_B0M'] = sales_info['이용건수_할부_B0M'].astype('int64')
            sales_features['이용건수_CA_B0M'] = sales_info['이용건수_CA_B0M'].astype('int64')
            sales_features['이용건수_체크_B0M'] = sales_info['이용건수_체크_B0M'].astype('int64')
            sales_features['이용건수_카드론_B0M'] = sales_info['이용건수_카드론_B0M'].astype('int64')
            features = pd.concat([features, sales_features], axis=1)
        
        # 잔액정보 특성
        if '5.잔액정보' in data:
            balance_info = data['5.잔액정보']
            balance_features = pd.DataFrame()
            balance_features['ID'] = balance_info['ID']
            balance_features['잔액_일시불_B0M'] = balance_info['잔액_일시불_B0M'].astype('float64')
            balance_features['잔액_할부_B0M'] = balance_info['잔액_할부_B0M'].astype('float64')
            balance_features['잔액_현금서비스_B0M'] = balance_info['잔액_현금서비스_B0M'].astype('float64')
            balance_features['잔액_카드론_B0M'] = balance_info['잔액_카드론_B0M'].astype('float64')
            balance_features['연체잔액_B0M'] = balance_info['연체잔액_B0M'].astype('float64')
            features = pd.concat([features, balance_features], axis=1)
        
        # 채널정보 특성
        if '6.채널정보' in data:
            channel_info = data['6.채널정보']
            channel_features = pd.DataFrame()
            channel_features['ID'] = channel_info['ID']
            channel_features['상담건수_B0M'] = channel_info['상담건수_B0M'].astype('int64')
            features = pd.concat([features, channel_features], axis=1)
        
        # 마케팅정보 특성
        if '7.마케팅정보' in data:
            marketing_info = data['7.마케팅정보']
            marketing_features = pd.DataFrame()
            marketing_features['ID'] = marketing_info['ID']
            marketing_features['컨택건수_카드론_TM_B0M'] = marketing_info['컨택건수_카드론_TM_B0M'].astype('int64')
            marketing_features['컨택건수_리볼빙_TM_B0M'] = marketing_info['컨택건수_리볼빙_TM_B0M'].astype('int64')
            marketing_features['컨택건수_CA_TM_B0M'] = marketing_info['컨택건수_CA_TM_B0M'].astype('int64')
            marketing_features['컨택건수_이용유도_TM_B0M'] = marketing_info['컨택건수_이용유도_TM_B0M'].astype('int64')
            features = pd.concat([features, marketing_features], axis=1)
        
        # 성과정보 특성
        if '8.성과정보' in data:
            performance_info = data['8.성과정보']
            performance_features = pd.DataFrame()
            performance_features['ID'] = performance_info['ID']
            performance_features['증감율_이용건수_신용_전월'] = performance_info['증감율_이용건수_신용_전월'].astype('float64')
            performance_features['증감율_이용건수_신판_전월'] = performance_info['증감율_이용건수_신판_전월'].astype('float64')
            performance_features['증감율_이용건수_일시불_전월'] = performance_info['증감율_이용건수_일시불_전월'].astype('float64')
            performance_features['증감율_이용건수_할부_전월'] = performance_info['증감율_이용건수_할부_전월'].astype('float64')
            performance_features['증감율_이용건수_CA_전월'] = performance_info['증감율_이용건수_CA_전월'].astype('float64')
            performance_features['혜택수혜율_R3M'] = performance_info['혜택수혜율_R3M'].astype('float64')
            performance_features['혜택수혜율_B0M'] = performance_info['혜택수혜율_B0M'].astype('float64')
            features = pd.concat([features, performance_features], axis=1)
        
        # 결측치 처리
        features = features.fillna(0)
        
        # ID 컬럼 제거 (모델 학습에 불필요)
        if 'ID' in features.columns:
            features = features.drop('ID', axis=1)
        
        logger.info("특성 생성 완료")
        return features
        
    except Exception as e:
        logger.error(f"특성 생성 중 오류 발생: {str(e)}")
        raise 