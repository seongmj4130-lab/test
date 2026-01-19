#!/usr/bin/env python3
"""
L6 단계에서 Hit Ratio가 어떻게 계산되는지 확인
"""

from pathlib import Path

import pandas as pd


def check_l6_hit_ratio():
    """L6 데이터에서 Hit Ratio 관련 컬럼 확인"""

    print("🔍 L6 단계 Hit Ratio 확인")
    print("=" * 50)

    # L6 데이터 로드 (수정된 버전)
    baseline_dir = Path('baseline_20260112_145649')
    l6_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores_corrected.parquet'

    if not l6_path.exists():
        print("❌ 수정된 L6 데이터 파일이 없습니다.")
        return

    df = pd.read_parquet(l6_path)
    print(f"📊 L6 데이터 로드: {len(df)} 행")

    # Hit Ratio 관련 컬럼 찾기
    hit_related_cols = [col for col in df.columns if 'hit' in col.lower() or 'ratio' in col.lower()]
    print(f"🎯 Hit Ratio 관련 컬럼: {hit_related_cols}")

    # IC 관련 컬럼 찾기 (Hit Ratio와 관련)
    ic_cols = [col for col in df.columns if 'ic' in col.lower()]
    print(f"📊 IC 관련 컬럼: {ic_cols}")

    # true 값들의 분포 확인 (Hit Ratio 계산의 기초)
    if 'true_short' in df.columns and 'true_long' in df.columns:
        print("\n📈 true 값 분포 (Hit Ratio 계산 기초):")
        print(f"true_short > 0: {(df['true_short'] > 0).sum()} / {len(df)} ({(df['true_short'] > 0).mean():.1%})")
        print(f"true_long > 0: {(df['true_long'] > 0).sum()} / {len(df)} ({(df['true_long'] > 0).mean():.1%})")

        # phase별 Hit Ratio 계산
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            short_hit = (phase_data['true_short'] > 0).mean()
            long_hit = (phase_data['true_long'] > 0).mean()
            print(f"{phase.upper()} 구간: short_hit={short_hit:.1%}, long_hit={long_hit:.1%}")

    print("\n💡 Hit Ratio 계산 방법:")
    print("1. L6 단계: 각 피처의 예측력 평가 (IC, Hit Ratio)")
    print("2. L7 단계: 실제 백테스트에서 거래별 Hit Ratio 계산")
    print("3. 현재 문제: L7의 Hit Ratio를 사용하고 있지만 L6의 값을 사용해야 함")

    print("\n📋 수정 방안:")
    print("1. L6에서 계산된 Hit Ratio를 L7 백테스트에 전달")
    print("2. 피처별 Hit Ratio를 종합하여 최종 Hit Ratio 산출")
    print("3. 랭킹 단계의 예측력을 정확히 반영")

if __name__ == "__main__":
    check_l6_hit_ratio()
