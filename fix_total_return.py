#!/usr/bin/env python3
"""
총수익률 과대 문제 해결 - L6 데이터 정규화
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def fix_total_return_data():
    """L6 데이터의 true_short/true_long을 백분율에서 소수점으로 변환"""

    print("🔧 총수익률 과대 문제 해결")
    print("=" * 50)

    # L6 데이터 경로
    baseline_dir = Path('baseline_20260112_145649')
    original_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores.parquet'
    backup_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores_original.parquet'

    if not original_path.exists():
        print("❌ L6 데이터 파일이 없습니다.")
        return

    # 원본 데이터 백업
    import shutil
    if not backup_path.exists():
        shutil.copy2(original_path, backup_path)
        print("📦 원본 데이터 백업 완료")

    # 데이터 로드
    df = pd.read_parquet(original_path)
    print(f"📊 원본 데이터 로드: {len(df)} 행")

    # 수정 전 통계
    print("\n📊 수정 전 true_short 통계:")
    print(f"  평균: {df['true_short'].mean():.6f}")
    print(f"  최소: {df['true_short'].min():.6f}")
    print(f"  최대: {df['true_short'].max():.6f}")

    print("\n📊 수정 전 true_long 통계:")
    print(f"  평균: {df['true_long'].mean():.6f}")
    print(f"  최소: {df['true_long'].min():.6f}")
    print(f"  최대: {df['true_long'].max():.6f}")

    # 백분율에서 소수점으로 변환 (÷100)
    df['true_short'] = df['true_short'] / 100
    df['true_long'] = df['true_long'] / 100

    # 수정 후 통계
    print("\n✅ 수정 후 true_short 통계:")
    print(f"  평균: {df['true_short'].mean():.6f}")
    print(f"  최소: {df['true_short'].min():.6f}")
    print(f"  최대: {df['true_short'].max():.6f}")

    print("\n✅ 수정 후 true_long 통계:")
    print(f"  평균: {df['true_long'].mean():.6f}")
    print(f"  최소: {df['true_long'].min():.6f}")
    print(f"  최대: {df['true_long'].max():.6f}")

    # 수정된 데이터 샘플
    print("\n🔍 수정된 데이터 샘플:")
    sample_cols = ['date', 'ticker', 'phase', 'true_short', 'true_long']
    print(df[sample_cols].head(10))

    # 수정된 데이터 저장
    corrected_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores_corrected.parquet'
    df.to_parquet(corrected_path, index=False)
    print(f"\n💾 수정된 데이터 저장: {corrected_path}")

    # 검증: 백테스트에서 사용할 데이터 경로 업데이트
    print("\n🔄 백테스트 재실행 준비:")
    print("  1. run_dynamic_period_backtest.py에서 데이터 경로 변경")
    print("  2. rebalance_scores_corrected.parquet 사용")
    print("  3. 백테스트 재실행으로 수익률 검증")

    return corrected_path

def update_backtest_data_path():
    """백테스트 스크립트에서 수정된 데이터 경로 사용하도록 업데이트"""

    script_path = Path('run_dynamic_period_backtest.py')

    if not script_path.exists():
        print("❌ 백테스트 스크립트를 찾을 수 없습니다.")
        return

    # 스크립트 내용 읽기
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 데이터 경로 변경
    old_path = "rebalance_scores.parquet"
    new_path = "rebalance_scores_corrected.parquet"

    if old_path in content:
        updated_content = content.replace(old_path, new_path)

        # 업데이트된 내용 저장
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print(f"✅ 백테스트 스크립트 업데이트: {old_path} → {new_path}")
    else:
        print("ℹ️  백테스트 스크립트에 변경사항 없음")

if __name__ == "__main__":
    corrected_path = fix_total_return_data()
    if corrected_path:
        update_backtest_data_path()
        print("\n🎯 다음 단계: python run_dynamic_period_backtest.py 실행")