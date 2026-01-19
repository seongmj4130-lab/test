# -*- coding: utf-8 -*-
"""메트릭 구조 확인"""
import pandas as pd

df = pd.read_parquet('data/interim/bt_metrics_bt20_ens.parquet')
print('=== bt_metrics 구조 ===')
print(f'컬럼: {df.columns.tolist()}')
print(f'\n행 수: {len(df)}')
print('\n=== 전체 데이터 ===')
print(df.to_string())
print('\n=== holdout 필터링 ===')
h = df[df['phase']=='holdout']
print(h.to_string())
print(f'\nholdout 행 수: {len(h)}')
if len(h) > 0:
    print(f'\nholdout 첫 번째 행의 net_sharpe: {h["net_sharpe"].iloc[0]}')

