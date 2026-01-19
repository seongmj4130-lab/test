#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track A 성과 결과 정리
"""

import pandas as pd

def summarize_track_a_results():
    """Track A 성과 결과를 PPT 형식으로 정리"""

    # 기존 Track A 성과 데이터 로드
    df = pd.read_csv('results/final_track_a_performance_results.csv')
    print('=== 기존 Track A 성과 데이터 ===')
    print(df.to_string())
    print()

    # PPT 형식으로 변환
    print('=== PPT 슬라이드 10용 성과 지표 ===')
    print('### 랭킹 엔진 성과 지표')
    print()
    print('| 전략 | Hit Ratio | IC | ICIR | 과적합 위험 |')
    print('|------|-----------|----|------|------------|')

    for idx, row in df.iterrows():
        strategy_name = row['strategy']
        if 'BT20 단기' in strategy_name:
            display_name = '단기랭킹 전략'
        elif 'BT120 장기' in strategy_name:
            display_name = '장기랭킹 전략'
        elif 'BT20 앙상블' in strategy_name:
            display_name = '통합랭킹 전략'
        else:
            display_name = strategy_name

        hit_ratio = f"Dev {row['hit_ratio_dev']}% → Holdout {row['hit_ratio_holdout']}%"
        ic = f"Dev {row['ic_dev']:.3f} → Holdout {row['ic_holdout']:.3f}"
        icir = f"Dev {row['icir_dev']:.3f} → Holdout {row['icir_holdout']:.3f}"
        risk = row['overfitting_risk']

        print(f'| **{display_name}** | {hit_ratio} | {ic} | {icir} | {risk} |')

    print()
    print('**평가**: 예측력(IC/ICIR)은 장기랭킹 전략이 더 안정적이지만, 특정 국면(holdout)에서는 단기랭킹 전략이 더 큰 트레이딩 알파를 만들었다.')

    # 상세 분석
    print()
    print('=== 상세 분석 결과 ===')

    # 최고 성과 전략 찾기
    best_icir = df.loc[df['icir_holdout'].idxmax()]
    print(f"최고 ICIR 전략: {best_icir['strategy']} (ICIR: {best_icir['icir_holdout']:.3f})")

    # 안정성 평가
    stable_strategies = df[df['overfitting_risk'] == 'LOW']
    if len(stable_strategies) > 0:
        print(f"안정적 전략: {', '.join(stable_strategies['strategy'].tolist())}")

    # 평균 성과
    print(f"평균 Hit Ratio (Dev): {df['hit_ratio_dev'].mean():.1f}%")
    print(f"평균 Hit Ratio (Holdout): {df['hit_ratio_holdout'].mean():.1f}%")
    print(f"평균 ICIR (Holdout): {df['icir_holdout'].mean():.3f}")

if __name__ == '__main__':
    summarize_track_a_results()