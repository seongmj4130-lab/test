# -*- coding: utf-8 -*-
"""최신 메트릭 확인"""
import pandas as pd

strategies = ['bt20_ens', 'bt20_short', 'bt120_ens', 'bt120_long']
print('=== 최신 저장된 메트릭 (Holdout) ===\n')

for s in strategies:
    df = pd.read_parquet(f'data/interim/bt_metrics_{s}.parquet')
    holdout = df[df['phase'] == 'holdout']
    if len(holdout) > 0:
        print(f'{s}:')
        print(f'  Sharpe: {holdout["net_sharpe"].iloc[0]:.4f}')
        print(f'  CAGR: {holdout["net_cagr"].iloc[0]:.4%}')
        print(f'  MDD: {holdout["net_mdd"].iloc[0]:.4%}')
        print(f'  Calmar: {holdout["net_calmar_ratio"].iloc[0]:.4f}')
        print()

