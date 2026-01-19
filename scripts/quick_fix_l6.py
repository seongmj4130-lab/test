import shutil

import pandas as pd

# 데이터 로드
df = pd.read_parquet("data/interim/rebalance_scores.parquet")
print(f"데이터 로드: {len(df)}행 x {len(df.columns)}열")

# 결측치 확인
missing_before = df.isnull().sum().sum()
print(f"결측치 (처리 전): {missing_before}")

# 결측치 채우기
df_filled = df.fillna(0.0)

# 결측치 확인
missing_after = df_filled.isnull().sum().sum()
print(f"결측치 (처리 후): {missing_after}")

# 백업
shutil.copy2(
    "data/interim/rebalance_scores.parquet",
    "data/interim/rebalance_scores_original.parquet",
)
print("백업 완료")

# 저장
df_filled.to_parquet("data/interim/rebalance_scores.parquet", index=False)
print("결측치 처리 완료!")
