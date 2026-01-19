# -*- coding: utf-8 -*-
"""
입력 데이터(rebalance_scores)의 날짜 분포 확인
"""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("data/interim")

# rebalance_scores 파일 찾기
score_files = list(data_dir.glob("rebalance_scores_*.parquet"))
print("=" * 80)
print("rebalance_scores 파일 확인")
print("=" * 80)

for f in sorted(score_files):
    df = pd.read_parquet(f)
    model_name = f.stem.replace("rebalance_scores_", "")
    print(f"\n{model_name}:")
    print(f"  총 행 수: {len(df):,}")
    print(f"  날짜 수: {df['date'].nunique():,}")
    
    # 날짜 간격 분석
    dates = sorted(df['date'].unique())
    if len(dates) > 1:
        date_diffs = [(pd.to_datetime(dates[i+1]) - pd.to_datetime(dates[i])).days 
                      for i in range(len(dates)-1)]
        print(f"  날짜 간격 통계:")
        print(f"    평균: {np.mean(date_diffs):.1f}일")
        print(f"    중앙값: {np.median(date_diffs):.1f}일")
        print(f"    최소: {min(date_diffs)}일")
        print(f"    최대: {max(date_diffs)}일")
        print(f"  샘플 날짜 (처음 10개):")
        print(f"    {dates[:10]}")
        
        # rebalance_interval=1일 때와 20일 때 비교
        print(f"\n  rebalance_interval=1일 때 예상 리밸런싱 날짜: {len(dates)}개")
        print(f"  rebalance_interval=20일 때 예상 리밸런싱 날짜: {len([dates[i] for i in range(0, len(dates), 20)])}개")
        print(f"  rebalance_interval=120일 때 예상 리밸런싱 날짜: {len([dates[i] for i in range(0, len(dates), 120)])}개")

