import os

import numpy as np
import pandas as pd


def load_bt_metrics(strategy_name):
    """전략별 백테스트 메트릭스 로드"""
    file_path = f'data/interim/bt_metrics_{strategy_name}.csv'
    df = pd.read_csv(file_path)
    return df

# 전략 리스트
strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']
strategy_names = {
    'bt20_short': 'BT20 단기 (20일)',
    'bt20_ens': 'BT20 앙상블 (20일)',
    'bt120_long': 'BT120 장기 (120일)',
    'bt120_ens': 'BT120 앙상블 (120일)'
}

print("=== Holdout 기간 IC 성과 지표 (정확한 값) ===")

# 각 전략별 Holdout IC 메트릭스 추출
ic_results = []
for strategy in strategies:
    df = load_bt_metrics(strategy)
    holdout_data = df[df['phase'] == 'holdout']

    if len(holdout_data) > 0:
        ic = holdout_data['ic'].iloc[0]
        rank_ic = holdout_data['rank_ic'].iloc[0]
        icir = holdout_data['icir'].iloc[0]
        rank_icir = holdout_data['rank_icir'].iloc[0]

        result = {
            'strategy': strategy_names[strategy],
            'ic': ic,
            'rank_ic': rank_ic,
            'icir': icir,
            'rank_icir': rank_icir
        }
        ic_results.append(result)

        print(f"{strategy_names[strategy]}:")
        print(".4f")
        print(".4f")
        print(".3f")
        print(".3f")
        print()

# DataFrame 생성
df_ic_results = pd.DataFrame(ic_results)

# CSV 및 Parquet 저장
df_ic_results.to_csv('data/holdout_ic_metrics_correct.csv', index=False)
df_ic_results.to_parquet('data/holdout_ic_metrics_correct.parquet', index=False)

print("정확한 IC 메트릭스 저장 완료:")
print("- data/holdout_ic_metrics_correct.csv")
print("- data/holdout_ic_metrics_correct.parquet")
print()

# Track A와 Track B 비교를 위한 종합 분석
print("=== Track A vs Track B 최종 성과 비교 ===")

# Track A 현재 성과 (Hit Ratio)
track_a_hit_ratio = pd.read_csv('data/track_a_performance_metrics.csv')

print("Track A (랭킹 엔진) - Hit Ratio:")
for idx, row in track_a_hit_ratio.iterrows():
    if 'Holdout' in row['metric']:
        print(f"  {row['metric']}: {row['value']} ({row['achievement']})")

print("\nTrack B (백테스트 전략) - IC 메트릭스 (Holdout):")
ic_summary = df_ic_results[['strategy', 'rank_ic', 'rank_icir']].copy()
ic_summary['rank_ic'] = ic_summary['rank_ic'].round(4)
ic_summary['rank_icir'] = ic_summary['rank_icir'].round(3)
print(ic_summary.to_string(index=False))

print("\nTrack B (백테스트 전략) - Holdout 종합 성과:")
holdout_data = pd.read_csv('data/holdout_performance_metrics.csv')
holdout_summary = holdout_data[['strategy', 'sharpe_ratio', 'cagr', 'hit_ratio']].copy()
holdout_summary['sharpe_ratio'] = holdout_summary['sharpe_ratio'].round(3)
holdout_summary['cagr'] = (holdout_summary['cagr'] * 100).round(1).astype(str) + '%'
holdout_summary['hit_ratio'] = (holdout_summary['hit_ratio'] * 100).round(1).astype(str) + '%'
print(holdout_summary.to_string(index=False))

# 최종 종합 보고
print("\n=== 최종 성과 비교표 ===")
print("| 전략 | Hit Ratio (Track A) | Rank IC (Track B) | Rank ICIR (Track B) | Sharpe (Track B) | CAGR (Track B) |")
print("|------|-------------------|------------------|-------------------|-----------------|----------------|")

# 전략별로 매핑
strategy_mapping = {
    'BT20 단기 (20일)': '단기 랭킹 Holdout',
    'BT20 앙상블 (20일)': '통합 랭킹 Holdout',
    'BT120 장기 (120일)': '장기 랭킹 Holdout',
    'BT120 앙상블 (120일)': '통합 랭킹 Holdout'
}

for idx, ic_row in df_ic_results.iterrows():
    strategy_name = ic_row['strategy']
    track_a_metric = strategy_mapping.get(strategy_name, '통합 랭킹 Holdout')

    # Track A Hit Ratio 찾기
    track_a_row = track_a_hit_ratio[track_a_hit_ratio['metric'] == track_a_metric]
    if len(track_a_row) > 0:
        hit_ratio = track_a_row['value'].iloc[0]
    else:
        hit_ratio = "N/A"

    # Track B 데이터
    rank_ic = ".4f"
    rank_icir = ".3f"

    # Sharpe와 CAGR
    bt_row = holdout_data[holdout_data['strategy'] == strategy_name]
    if len(bt_row) > 0:
        sharpe = ".3f"
        cagr = ".1f"
    else:
        sharpe = "N/A"
        cagr = "N/A"

    print(f"| {strategy_name} | {hit_ratio} | {rank_ic} | {rank_icir} | {sharpe} | {cagr}% |")

print("\n📊 메트릭스 설명:")
print("- Hit Ratio: Track A 랭킹 엔진의 예측 정확도")
print("- Rank IC: Track B에서 랭킹 스코어와 실제 수익률의 상관관계")
print("- Rank ICIR: Rank IC의 정보 비율 (1.0 이상 = 우수)")
print("- Sharpe: Track B 리스크 조정 수익률")
print("- CAGR: Track B 연평균 복리 수익률")
print("\n💡 인사이트:")
print("- Track A의 Hit Ratio와 Track B의 Rank IC가 서로 검증")
print("- IC 값이 높을수록 랭킹 예측력이 우수함을 의미")
print("- ICIR > 1.0이면 우연이 아닌 의미있는 예측력")
