import pandas as pd

# 신용정보 파일 읽기
df = pd.read_parquet('train/2.신용정보/201812_train_신용정보.parquet')

print("컬럼명:")
print(df.columns.tolist())
print("\n처음 5행:")
print(df.head()) 