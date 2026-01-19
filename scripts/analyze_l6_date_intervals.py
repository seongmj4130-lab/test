from pathlib import Path

import pandas as pd

df = pd.read_parquet('data/interim/rebalance_scores.parquet')
print('L6 rebalance_scores 날짜 간격 분석:')
print(f'총 행 수: {len(df):,}')
print(f'유니크 날짜 수: {df["date"].nunique()}')

dates = sorted(pd.to_datetime(df['date'].unique()))
print(f'날짜 범위: {dates[0].date()} ~ {dates[-1].date()}')

intervals = []
for i in range(1, len(dates)):
    interval = (dates[i] - dates[i-1]).days
    intervals.append(interval)

if intervals:
    avg_interval = sum(intervals) / len(intervals)
    print('.1f')
    print(f'최소 간격: {min(intervals)}일')
    print(f'최대 간격: {max(intervals)}일')

    interval_20_count = sum(1 for i in intervals if i == 20)
    print('.1f')
else:
    print('날짜 간격 계산 불가')
