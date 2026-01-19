# -*- coding: utf-8 -*-
"""IC 파일 확인"""

from pathlib import Path
import pandas as pd

base_dir = Path(__file__).parent.parent
ic_file = base_dir / 'artifacts' / 'reports' / 'feature_ic_dev.csv'

print(f"IC 파일 경로: {ic_file}")
print(f"존재 여부: {ic_file.exists()}")

if ic_file.exists():
    df = pd.read_csv(ic_file)
    print(f"\n피쳐 수: {len(df)}")
    print(f"컬럼: {list(df.columns)}")
    
    if 'rank_ic' in df.columns:
        print(f"\nRank IC 통계:")
        print(f"  평균: {df['rank_ic'].mean():.4f}")
        print(f"  최소: {df['rank_ic'].min():.4f}")
        print(f"  최대: {df['rank_ic'].max():.4f}")
        print(f"  IC < 0.01인 피쳐 수: {(df['rank_ic'].abs() < 0.01).sum()}")
        print(f"  IC < 0.02인 피쳐 수: {(df['rank_ic'].abs() < 0.02).sum()}")
        
        print(f"\n상위 10개 피쳐 (Rank IC):")
        top10 = df.nlargest(10, 'rank_ic', key=abs)
        for _, row in top10.iterrows():
            print(f"  {row['feature']}: {row['rank_ic']:.4f}")

