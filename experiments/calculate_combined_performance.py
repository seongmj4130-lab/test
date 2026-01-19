import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_bt_metrics(strategy_name):
    """전략별 백테스트 메트릭스 로드"""
    file_path = f'data/interim/bt_metrics_{strategy_name}.csv'
    df = pd.read_csv(file_path)
    return df

def calculate_combined_metrics(dev_metrics, holdout_metrics):
    """Dev와 Holdout 메트릭스를 결합하여 전체 기간 성과 계산"""

    # 기간 정보
    dev_start = pd.to_datetime(dev_metrics['date_start'].iloc[0])
    dev_end = pd.to_datetime(dev_metrics['date_end'].iloc[0])
    holdout_start = pd.to_datetime(holdout_metrics['date_start'].iloc[0])
    holdout_end = pd.to_datetime(holdout_metrics['date_end'].iloc[0])

    # 기간 길이 (년 단위)
    dev_years = (dev_end - dev_start).days / 365.25
    holdout_years = (holdout_end - holdout_start).days / 365.25
    total_years = dev_years + holdout_years

    # 수익률 결합 (기하 평균)
    dev_cagr = dev_metrics['net_cagr'].iloc[0]
    holdout_cagr = holdout_metrics['net_cagr'].iloc[0]

    # 전체 CAGR 계산: (1 + r1)^t1 * (1 + r2)^t2 = (1 + r_total)^(t1+t2)
    combined_cagr = ((1 + dev_cagr)**dev_years * (1 + holdout_cagr)**holdout_years)**(1/total_years) - 1

    # Sharpe Ratio 결합 (가정: 상관관계 고려하지 않음 - 단순 평균)
    dev_sharpe = dev_metrics['net_sharpe'].iloc[0]
    holdout_sharpe = holdout_metrics['net_sharpe'].iloc[0]
    combined_sharpe = (dev_sharpe * dev_years + holdout_sharpe * holdout_years) / total_years

    # MDD 결합 (최악값 선택)
    dev_mdd = dev_metrics['net_mdd'].iloc[0]
    holdout_mdd = holdout_metrics['net_mdd'].iloc[0]
    combined_mdd = min(dev_mdd, holdout_mdd)  # 더 나쁜 MDD 선택

    # Calmar Ratio
    combined_calmar = combined_cagr / abs(combined_mdd) if combined_mdd != 0 else 0

    # Total Return 결합
    dev_total_return = dev_metrics['net_total_return'].iloc[0]
    holdout_total_return = holdout_metrics['net_total_return'].iloc[0]
    combined_total_return = ((1 + dev_total_return) * (1 + holdout_total_return) - 1)

    # Hit Ratio (가중 평균)
    dev_hit_ratio = dev_metrics['net_hit_ratio'].iloc[0]
    holdout_hit_ratio = holdout_metrics['net_hit_ratio'].iloc[0]
    dev_trades = dev_metrics['n_rebalances'].iloc[0]
    holdout_trades = holdout_metrics['n_rebalances'].iloc[0]
    combined_hit_ratio = (dev_hit_ratio * dev_trades + holdout_hit_ratio * holdout_trades) / (dev_trades + holdout_trades)

    return {
        'net_sharpe': combined_sharpe,
        'net_cagr': combined_cagr,
        'net_mdd': combined_mdd,
        'net_calmar_ratio': combined_calmar,
        'net_total_return': combined_total_return,
        'net_hit_ratio': combined_hit_ratio,
        'dev_years': dev_years,
        'holdout_years': holdout_years,
        'total_years': total_years,
        'date_start': dev_start,
        'date_end': holdout_end
    }

# 전략 리스트
strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']
strategy_names = {
    'bt20_short': 'BT20 단기 (20일)',
    'bt20_ens': 'BT20 앙상블 (20일)',
    'bt120_long': 'BT120 장기 (120일)',
    'bt120_ens': 'BT120 앙상블 (120일)'
}

# 결과 저장용 데이터프레임
holdout_results = []
combined_results = []

print("=== Holdout 단독 결과 ===")
for strategy in strategies:
    df = load_bt_metrics(strategy)
    holdout_data = df[df['phase'] == 'holdout'].iloc[0]

    result = {
        'strategy': strategy_names[strategy],
        'phase': 'Holdout',
        'sharpe_ratio': holdout_data['net_sharpe'],
        'cagr': holdout_data['net_cagr'],
        'mdd': holdout_data['net_mdd'],
        'calmar_ratio': holdout_data['net_calmar_ratio'],
        'total_return': holdout_data['net_total_return'],
        'hit_ratio': holdout_data['net_hit_ratio'],
        'period': f"{holdout_data['date_start']} ~ {holdout_data['date_end']}"
    }
    holdout_results.append(result)

    print(f"{strategy_names[strategy]}:")
    print(".3f")
    print(f"  CAGR: {result['cagr']:.1%}")
    print(f"  MDD: {result['mdd']:.1%}")
    print(f"  Calmar: {result['calmar_ratio']:.3f}")
    print()

print("=== Dev+Holdout 합친 결과 ===")
for strategy in strategies:
    df = load_bt_metrics(strategy)
    dev_data = df[df['phase'] == 'dev']
    holdout_data = df[df['phase'] == 'holdout']

    if len(dev_data) > 0 and len(holdout_data) > 0:
        combined = calculate_combined_metrics(dev_data, holdout_data)

        result = {
            'strategy': strategy_names[strategy],
            'phase': 'Combined',
            'sharpe_ratio': combined['net_sharpe'],
            'cagr': combined['net_cagr'],
            'mdd': combined['net_mdd'],
            'calmar_ratio': combined['net_calmar_ratio'],
            'total_return': combined['net_total_return'],
            'hit_ratio': combined['net_hit_ratio'],
            'period': f"{combined['date_start'].strftime('%Y-%m-%d')} ~ {combined['date_end'].strftime('%Y-%m-%d')}"
        }
        combined_results.append(result)

        print(f"{strategy_names[strategy]}:")
        print(".3f")
        print(f"  CAGR: {result['cagr']:.1%}")
        print(f"  MDD: {result['mdd']:.1%}")
        print(f"  Calmar: {result['calmar_ratio']:.3f}")
        print()

# DataFrame 생성 및 저장
df_holdout = pd.DataFrame(holdout_results)
df_combined = pd.DataFrame(combined_results)

# CSV 및 Parquet 저장
df_holdout.to_csv('data/holdout_performance_metrics.csv', index=False)
df_holdout.to_parquet('data/holdout_performance_metrics.parquet', index=False)

df_combined.to_csv('data/combined_performance_metrics.csv', index=False)
df_combined.to_parquet('data/combined_performance_metrics.parquet', index=False)

print("저장 완료:")
print("- data/holdout_performance_metrics.csv/parquet")
print("- data/combined_performance_metrics.csv/parquet")

# Track A vs Track B 비교를 위한 데이터 준비
track_a_data = pd.read_csv('data/track_a_performance_metrics.csv')
track_b_holdout = df_holdout.copy()
track_b_combined = df_combined.copy()

print("\n=== Track A vs Track B 성과 비교 ===")
print("Track A (랭킹 엔진):")
track_a_data['달성률'] = track_a_data.apply(lambda x: f"{x['achievement']}", axis=1)
print(track_a_data[['metric', 'value', '달성률']].to_string(index=False))

print("\nTrack B (백테스트 전략) - Holdout:")
track_b_holdout['달성률'] = '실제 성과'
print(track_b_holdout[['strategy', 'sharpe_ratio', 'cagr', 'mdd']].to_string(index=False))

print("\nTrack B (백테스트 전략) - Combined:")
track_b_combined['달성률'] = '실제 성과'
print(track_b_combined[['strategy', 'sharpe_ratio', 'cagr', 'mdd']].to_string(index=False))
