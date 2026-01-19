import pandas as pd
import os

# Track A 성과 지표 (Hit Ratio 중심)
track_a_metrics = {
    'metric': [
        '통합 랭킹 Hit Ratio',
        '단기 랭킹 Hit Ratio',
        '장기 랭킹 Hit Ratio',
        '단기 랭킹 Holdout',
        '통합 랭킹 Holdout',
        '장기 랭킹 Holdout'
    ],
    'value': [
        '49.58%',
        '49.28%',
        '50.14%',
        '50.99%',
        '51.06%',
        '51.00%'
    ],
    'target': [
        '50%',
        '50%',
        '50%',
        '50%',
        '50%',
        '50%'
    ],
    'achievement': [
        '99.16%',
        '98.56%',
        '100.28%',
        '101.98%',
        '102.12%',
        '102.00%'
    ],
    'status': [
        '거의 달성',
        '거의 달성',
        '달성',
        '초과 달성',
        '초과 달성',
        '초과 달성'
    ]
}

# Track B 성과 지표 (Return, MDD, Sharpe, CAGR 중심)
track_b_metrics = []

strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']
phases = ['Dev', 'Holdout']

# 각 전략별 성과 지표 수집
for strategy in strategies:
    for phase in phases:
        if strategy == 'bt20_short':
            if phase == 'Dev':
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.24,
                    'total_return': '21.81%',
                    'cagr': '2.90%',
                    'mdd': '-40.85%',
                    'calmar_ratio': 0.07,
                    'hit_ratio': '51.16%',
                    'profit_factor': 1.26,
                    'avg_turnover': '2.42%',
                    'avg_trade_duration': '29.7일'
                }
            else:  # Holdout
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.26,
                    'total_return': '4.90%',
                    'cagr': '2.69%',
                    'mdd': '-39.13%',
                    'calmar_ratio': 0.29,
                    'hit_ratio': '39.13%',
                    'profit_factor': 1.22,
                    'avg_turnover': '3.26%',
                    'avg_trade_duration': '30.0일'
                }

        elif strategy == 'bt20_ens':
            if phase == 'Dev':
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.23,
                    'total_return': '19.70%',
                    'cagr': '2.64%',
                    'mdd': '-42.10%',
                    'calmar_ratio': 0.06,
                    'hit_ratio': '53.49%',
                    'profit_factor': 1.24,
                    'avg_turnover': '2.35%',
                    'avg_trade_duration': '29.7일'
                }
            else:  # Holdout
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.14,
                    'total_return': '1.85%',
                    'cagr': '1.02%',
                    'mdd': '-39.13%',
                    'calmar_ratio': 0.13,
                    'hit_ratio': '39.13%',
                    'profit_factor': 1.11,
                    'avg_turnover': '3.60%',
                    'avg_trade_duration': '30.0일'
                }

        elif strategy == 'bt120_long':
            if phase == 'Dev':
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.32,
                    'total_return': '23.99%',
                    'cagr': '3.45%',
                    'mdd': '-19.61%',
                    'calmar_ratio': 0.18,
                    'hit_ratio': '50.00%',
                    'profit_factor': 1.76,
                    'avg_turnover': '7.38%',
                    'avg_trade_duration': '178.2일'
                }
            else:  # Holdout
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.18,
                    'total_return': '2.88%',
                    'cagr': '2.92%',
                    'mdd': '-10.34%',
                    'calmar_ratio': 0.28,
                    'hit_ratio': '66.67%',
                    'profit_factor': 1.40,
                    'avg_turnover': '21.11%',
                    'avg_trade_duration': '180.1일'
                }

        elif strategy == 'bt120_ens':
            if phase == 'Dev':
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.28,
                    'total_return': '21.97%',
                    'cagr': '3.18%',
                    'mdd': '-20.82%',
                    'calmar_ratio': 0.15,
                    'hit_ratio': '42.86%',
                    'profit_factor': 1.68,
                    'avg_turnover': '9.29%',
                    'avg_trade_duration': '178.2일'
                }
            else:  # Holdout
                metrics = {
                    'strategy': strategy,
                    'phase': phase,
                    'sharpe_ratio': 0.22,
                    'total_return': '4.03%',
                    'cagr': '4.09%',
                    'mdd': '-11.06%',
                    'calmar_ratio': 0.37,
                    'hit_ratio': '66.67%',
                    'profit_factor': 1.49,
                    'avg_turnover': '23.33%',
                    'avg_trade_duration': '180.2일'
                }

        track_b_metrics.append(metrics)

# DataFrame 생성
df_track_a = pd.DataFrame(track_a_metrics)
df_track_b = pd.DataFrame(track_b_metrics)

# CSV 파일로 저장
df_track_a.to_csv('data/track_a_performance_metrics.csv', index=False)
df_track_b.to_csv('data/track_b_performance_metrics.csv', index=False)

# Parquet 파일로 저장
df_track_a.to_parquet('data/track_a_performance_metrics.parquet', index=False)
df_track_b.to_parquet('data/track_b_performance_metrics.parquet', index=False)

print("=== Track A 성과 지표 (Hit Ratio 중심) ===")
print(df_track_a.to_string(index=False))

print("\n=== Track B 성과 지표 (Return, MDD, Sharpe, CAGR 중심) ===")
print(df_track_b.to_string(index=False))

print("\n파일 저장 완료:")
print("- data/track_a_performance_metrics.csv")
print("- data/track_a_performance_metrics.parquet")
print("- data/track_b_performance_metrics.csv")
print("- data/track_b_performance_metrics.parquet")