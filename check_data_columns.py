#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

# 데이터 파일 경로
data_path = (
    Path(__file__).parent
    / "baseline_20260112_145649"
    / "data"
    / "interim"
    / "rebalance_scores.parquet"
)

# 데이터 로드
df = pd.read_parquet(data_path)

print("📊 L6 데이터 컬럼 목록:")
print("=" * 50)
for i, col in enumerate(df.columns, 1):
    print("2d")

print(f"\n📈 총 {len(df.columns)}개 컬럼")
print(f"📅 데이터 행 수: {len(df)}")

# return 관련 컬럼 찾기
return_cols = [
    col
    for col in df.columns
    if "ret" in col.lower() or "true" in col.lower() or "fwd" in col.lower()
]
print(f"\n🎯 Return 관련 컬럼들: {return_cols}")

# 샘플 데이터 확인
print("\n🔍 샘플 데이터 (첫 3행):")
print(df.head(3))
