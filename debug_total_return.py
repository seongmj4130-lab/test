#!/usr/bin/env python3
"""
총수익률 과대 문제 디버깅
"""

from pathlib import Path

import numpy as np
import pandas as pd


def debug_total_return():
    """총수익률 과대 문제 원인 분석"""

    print("🔍 총수익률 과대 문제 디버깅")
    print("=" * 50)

    # L6 랭킹 데이터 로드
    baseline_dir = Path('baseline_20260112_145649')
    l6_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores.parquet'

    if not l6_path.exists():
        print("❌ L6 데이터 파일이 없습니다.")
        return

    df = pd.read_parquet(l6_path)
    print(f"📊 L6 데이터 로드: {len(df)} 행")

    # return 컬럼 확인
    return_cols = [col for col in df.columns if 'true' in col.lower() or 'ret' in col.lower()]
    print(f"🎯 Return 관련 컬럼: {return_cols}")

    # 데이터 샘플 확인
    print("\n🔍 L6 데이터 샘플:")
    sample_cols = ['date', 'ticker', 'phase', 'true_short', 'true_long']
    if all(col in df.columns for col in sample_cols):
        print(df[sample_cols].head(10))

        # true_short, true_long 값의 범위 확인
        print("\n📊 true_short 통계:")
        print(f"  평균: {df['true_short'].mean():.6f}")
        print(f"  최소: {df['true_short'].min():.6f}")
        print(f"  최대: {df['true_short'].max():.6f}")
        print(f"  표준편차: {df['true_short'].std():.6f}")

        print("\n📊 true_long 통계:")
        print(f"  평균: {df['true_long'].mean():.6f}")
        print(f"  최소: {df['true_long'].min():.6f}")
        print(f"  최대: {df['true_long'].max():.6f}")
        print(f"  표준편차: {df['true_long'].std():.6f}")

    # 백테스트 결과와 비교
    print("\n📈 백테스트 결과에서 비정상적 수익률:")
    results_file = Path('results/dynamic_period_backtest_clean_20260113_212022.csv')

    if results_file.exists():
        results_df = pd.read_csv(results_file)

        # Total Return이 100% 이상인 경우 필터링
        high_returns = results_df[results_df['Total Return (%)'] > 100]
        if len(high_returns) > 0:
            print("비정상적 총수익률 (>100%):")
            for _, row in high_returns.iterrows():
                print(".2f")
        else:
            print("비정상적 수익률이 발견되지 않았습니다.")
    else:
        print("백테스트 결과 파일을 찾을 수 없습니다.")

    # 잠재적 원인 분석
    print("\n🔍 잠재적 원인 분석:")

    # 1. 데이터 scale 문제
    if 'true_short' in df.columns:
        max_short = df['true_short'].max()
        if max_short > 10:  # 1000% 이상
            print("⚠️  true_short 최대값이 비정상적으로 높음 (scale 문제 가능성)")
        elif max_short > 1:  # 100% 이상
            print("⚠️  true_short 최대값이 높음 (백분율로 표현된 것 같음)")

    # 2. 계산 문제
    print("💡 가능한 원인:")
    print("  1. L6 데이터의 true_short/true_long이 이미 백분율로 표현됨")
    print("  2. 백테스트에서 복리 계산이 중복 적용됨")
    print("  3. 데이터 정규화 누락")
    print("  4. 데모/가상 데이터 사용")

    print("\n📋 해결 방안:")
    print("  1. true_short/true_long 값을 소수점으로 변환 (÷100)")
    print("  2. 백테스트 수익률 계산 로직 검토")
    print("  3. 실제 OHLCV 데이터로 검증")

if __name__ == "__main__":
    debug_total_return()