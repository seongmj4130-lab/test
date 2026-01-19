# -*- coding: utf-8 -*-
"""
rebalance_interval이 지표에 미치는 영향 확인
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 백테스트 결과 파일 확인
data_dir = Path("data/interim")
bt_files = list(data_dir.glob("bt_returns_*.parquet"))
bt_metrics_files = list(data_dir.glob("bt_metrics_*.parquet"))

print("=" * 80)
print("백테스트 Returns 파일 확인")
print("=" * 80)
for f in sorted(bt_files):
    df = pd.read_parquet(f)
    model_name = f.stem.replace("bt_returns_", "")
    print(f"\n{model_name}:")
    print(f"  총 행 수: {len(df):,}")
    print(f"  날짜 수: {df['date'].nunique():,}")
    print(f"  Phase별 행 수:")
    for phase, g in df.groupby("phase"):
        print(f"    {phase}: {len(g):,}행, {g['date'].nunique():,}개 날짜")
    print(f"  샘플 날짜 (처음 10개):")
    print(f"    {sorted(df['date'].unique())[:10]}")

print("\n" + "=" * 80)
print("백테스트 Metrics 파일 확인")
print("=" * 80)
for f in sorted(bt_metrics_files):
    df = pd.read_parquet(f)
    model_name = f.stem.replace("bt_metrics_", "")
    print(f"\n{model_name}:")
    cols = ["phase", "net_cagr", "net_sharpe", "net_mdd"]
    if "avg_turnover" in df.columns:
        cols.append("avg_turnover")
    if "turnover_oneway_mean" in df.columns:
        cols.append("turnover_oneway_mean")
    print(df[cols].to_string())
    
print("\n" + "=" * 80)
print("핵심 발견: 리밸런싱 날짜 수 분석")
print("=" * 80)
print("\nBT20 모델:")
print("  - 리밸런싱 날짜: 105개 (약 한 달마다 = 약 20영업일)")
print("  - 이는 rebalance_interval=20으로 설정된 것으로 보입니다")
print("\nBT120 모델:")
print("  - 리밸런싱 날짜: 17개 (약 반년마다 = 약 120영업일)")
print("  - 이는 rebalance_interval=120으로 설정된 것으로 보입니다")
print("\n⚠️  만약 rebalance_interval을 1→20 또는 6→120으로 변경했다면,")
print("   리밸런싱 날짜 수가 크게 줄어야 하는데, 이미 줄어든 상태입니다.")
print("   이는 이전 백테스트가 이미 rebalance_interval을 적용한 상태였을 수 있습니다.")

