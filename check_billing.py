import pandas as pd
import os

# 청구입금정보 파일 읽기
file_path = os.path.join('train', '4.청구입금정보', '201812_train_청구입금정보.parquet')
df = pd.read_parquet(file_path)

print("청구입금정보 컬럼:")
print(df.columns.tolist())
print("\n처음 5개 행:")
print(df.head()) 