# -*- coding: utf-8 -*-
"""백테스트 설정값 확인"""
import pandas as pd

strategies = ['bt20_ens', 'bt20_short', 'bt120_ens', 'bt120_long']
print('=' * 100)
print('백테스트 결과값에 해당하는 설정값')
print('=' * 100)

for s in strategies:
    df = pd.read_parquet(f'data/interim/bt_metrics_{s}.parquet')
    holdout = df[df['phase'] == 'holdout']
    if len(holdout) > 0:
        h = holdout.iloc[0]
        print(f'\n[{s}]')
        print(f'  결과: Sharpe={h["net_sharpe"]:.4f}, CAGR={h["net_cagr"]:.4%}, MDD={h["net_mdd"]:.4%}, Calmar={h["net_calmar_ratio"]:.4f}')
        print(f'\n  설정값:')
        print(f'    holding_days: {h["holding_days"]}')
        print(f'    top_k: {h["top_k"]}')
        print(f'    cost_bps: {h["cost_bps"]}')
        print(f'    buffer_k: {h["buffer_k"]}')
        print(f'    weighting: {h["weighting"]}')
        if 'softmax_temperature' in h and pd.notna(h['softmax_temperature']):
            print(f'    softmax_temperature: {h["softmax_temperature"]}')
        print(f'    n_rebalances (holdout): {h["n_rebalances"]}')

