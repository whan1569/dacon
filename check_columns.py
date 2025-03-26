import pandas as pd
import os

def check_columns(data_type='train'):
    """각 데이터 폴더의 컬럼명을 확인합니다."""
    folders = [
        '1.회원정보', '2.신용정보', '3.승인매출정보', '4.청구입금정보',
        '5.잔액정보', '6.채널정보', '7.마케팅정보', '8.성과정보'
    ]
    
    for folder in folders:
        try:
            file_path = f'{data_type}/{folder}/201812_{data_type}_{folder.split(".")[1]}.parquet'
            print(f"\n{folder} 컬럼명:")
            df = pd.read_parquet(file_path)
            print(df.columns.tolist())
            print("\n처음 5행:")
            print(df.head())
        except Exception as e:
            print(f"\n{folder} 로드 실패: {str(e)}")

if __name__ == "__main__":
    print("학습 데이터 컬럼 확인:")
    check_columns('train')
    print("\n테스트 데이터 컬럼 확인:")
    check_columns('test') 