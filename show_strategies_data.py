#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3전략 6구간 월별 누적 수익률 데이터 표시
"""

import pandas as pd


def main():
    # 3전략 6구간 월별 누적 수익률 데이터 로드
    df = pd.read_csv('data/ui_strategies_cumulative_comparison_updated.csv')
    print('=== 3전략 6구간 월별 누적 수익률 데이터 ===')
    print(f'총 {len(df)}개월 데이터 (2023-01 ~ 2024-12)')
    print()

    # 전략별 컬럼 그룹화
    strategies = {
        'BT20 단기 전략': [col for col in df.columns if col.startswith('bt20_short_')],
        'BT120 장기 전략': [col for col in df.columns if col.startswith('bt120_long_')],
        'BT20 앙상블 전략': [col for col in df.columns if col.startswith('bt20_ens_')]
    }

    # 각 전략별로 6구간 표시
    for strategy_name, cols in strategies.items():
        print(f'## {strategy_name}')
        print('month,20d,40d,60d,80d,100d,120d')

        for _, row in df.iterrows():
            values = [f'{row[col]:.2f}' for col in cols]
            print(f'{row["month"]},' + ','.join(values))
        print()

    print('## KOSPI200 지수 (벤치마크)')
    print('month,kospi200')
    for _, row in df.iterrows():
        print(f'{row["month"]},{row["kospi200"]:.2f}')

if __name__ == "__main__":
    main()
