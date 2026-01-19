# -*- coding: utf-8 -*-
"""백테스트 결과와 문서 기대값 비교"""
from pathlib import Path

import pandas as pd
import yaml

base_dir = Path('C:/Users/seong/OneDrive/Desktop/bootcamp/03_code')

# 문서 기대값
expected = {
    'bt20_ens': {'net_sharpe': 0.6826, 'net_cagr': 0.1498, 'net_mdd': -0.1098, 'net_calmar_ratio': 1.3641},
    'bt20_short': {'net_sharpe': 0.6464, 'net_cagr': 0.1384, 'net_mdd': -0.0909, 'net_calmar_ratio': 1.5223},
    'bt120_ens': {'net_sharpe': 0.6263, 'net_cagr': 0.1166, 'net_mdd': -0.0769, 'net_calmar_ratio': 1.5156},
    'bt120_long': {'net_sharpe': 0.6839, 'net_cagr': 0.1360, 'net_mdd': -0.0866, 'net_calmar_ratio': 1.5700},
}

strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']

print("=" * 100)
print("백테스트 결과 vs 문서 기대값 비교 (Holdout 구간)")
print("=" * 100)
print()

results = []
for strategy in strategies:
    metrics_path = base_dir / 'data' / 'interim' / f'bt_metrics_{strategy}.parquet'
    if not metrics_path.exists():
        print(f"⚠️  {strategy}: 결과 파일 없음")
        continue

    df = pd.read_parquet(metrics_path)
    holdout = df[df['phase'] == 'holdout'].iloc[0]

    exp = expected[strategy]
    actual = {
        'net_sharpe': holdout['net_sharpe'],
        'net_cagr': holdout['net_cagr'],
        'net_mdd': holdout['net_mdd'],
        'net_calmar_ratio': holdout['net_calmar_ratio'],
    }

    results.append({
        'strategy': strategy,
        'metric': 'Net Sharpe',
        'expected': exp['net_sharpe'],
        'actual': actual['net_sharpe'],
        'diff': actual['net_sharpe'] - exp['net_sharpe'],
        'diff_pct': (actual['net_sharpe'] - exp['net_sharpe']) / abs(exp['net_sharpe']) * 100 if exp['net_sharpe'] != 0 else 0,
    })
    results.append({
        'strategy': strategy,
        'metric': 'Net CAGR',
        'expected': exp['net_cagr'],
        'actual': actual['net_cagr'],
        'diff': actual['net_cagr'] - exp['net_cagr'],
        'diff_pct': (actual['net_cagr'] - exp['net_cagr']) / abs(exp['net_cagr']) * 100 if exp['net_cagr'] != 0 else 0,
    })
    results.append({
        'strategy': strategy,
        'metric': 'Net MDD',
        'expected': exp['net_mdd'],
        'actual': actual['net_mdd'],
        'diff': actual['net_mdd'] - exp['net_mdd'],
        'diff_pct': (actual['net_mdd'] - exp['net_mdd']) / abs(exp['net_mdd']) * 100 if exp['net_mdd'] != 0 else 0,
    })
    results.append({
        'strategy': strategy,
        'metric': 'Net Calmar',
        'expected': exp['net_calmar_ratio'],
        'actual': actual['net_calmar_ratio'],
        'diff': actual['net_calmar_ratio'] - exp['net_calmar_ratio'],
        'diff_pct': (actual['net_calmar_ratio'] - exp['net_calmar_ratio']) / abs(exp['net_calmar_ratio']) * 100 if exp['net_calmar_ratio'] != 0 else 0,
    })

df_results = pd.DataFrame(results)

# 전략별로 출력
for strategy in strategies:
    print(f"\n[{strategy.upper()}]")
    print("-" * 100)
    df_strat = df_results[df_results['strategy'] == strategy]
    for _, row in df_strat.iterrows():
        match = "✓" if abs(row['diff']) < 0.001 else "✗"
        print(f"  {match} {row['metric']:15s} | 기대: {row['expected']:8.4f} | 실제: {row['actual']:8.4f} | 차이: {row['diff']:8.4f} ({row['diff_pct']:6.2f}%)")

print("\n" + "=" * 100)
print("요약")
print("=" * 100)

# 전체 일치도
total_metrics = len(results)
matched = sum(1 for r in results if abs(r['diff']) < 0.001)
print(f"일치한 지표: {matched}/{total_metrics} ({matched/total_metrics*100:.1f}%)")

# 전략별 일치도
for strategy in strategies:
    df_strat = df_results[df_results['strategy'] == strategy]
    matched = sum(1 for _, r in df_strat.iterrows() if abs(r['diff']) < 0.001)
    print(f"  {strategy}: {matched}/4 지표 일치")
