# -*- coding: utf-8 -*-
"""
L6 결측치 처리 (개선된 버전)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("🔧 L6 결측치 처리 시작")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    scores_file = interim_dir / 'rebalance_scores.parquet'

    # 데이터 로드
    df = pd.read_parquet(scores_file)
    print(f"📊 데이터 로드: {len(df)}행 x {len(df.columns)}열")

    # 결측치 분석
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    total_missing = missing_by_col.sum()

    print(f"❌ 총 결측치: {total_missing}개")
    print(f"❌ 결측 컬럼: {len(missing_cols)}개")

    # 결측치가 있는 행들을 확인
    missing_rows = df[df.isnull().any(axis=1)]
    print(f"⚠️ 결측치 행 수: {len(missing_rows)}/{len(df)} ({len(missing_rows)/len(df)*100:.1f}%)")

    # 결측치 패턴 분석
    print("\n🔍 결측치 패턴 분석:")
    sample_missing = missing_rows.head(3)
    for idx, row in sample_missing.iterrows():
        missing_in_row = row[row.isnull()].index.tolist()
        print(f"  행 {idx}: 결측 컬럼 {len(missing_in_row)}개 - {missing_in_row[:3]}{'...' if len(missing_in_row) > 3 else ''}")

    # 전략: 결측치가 있는 행 전체를 0으로 채움 (안전한 접근)
    print("\n🔧 보간 전략: 결측치 행 전체를 0.0으로 채움")
    df_fixed = df.fillna(0.0)

    # 검증
    final_missing = df_fixed.isnull().sum().sum()
    print(f"\n📊 보간 결과: {total_missing} → {final_missing}")

    if final_missing == 0:
        # 백업 및 저장
        import shutil
        backup_file = interim_dir / 'rebalance_scores_original.parquet'
        if not backup_file.exists():
            shutil.copy2(scores_file, backup_file)
            print("📋 원본 백업 완료")

        df_fixed.to_parquet(scores_file, index=False)
        print("✅ 결측치 처리 완료!")

        # 추가 검증
        print("\n🔍 추가 검증:")
        print(f"  score_ens 범위: {df_fixed['score_ens'].min():.4f} ~ {df_fixed['score_ens'].max():.4f}")
        print(f"  평균 score_ens: {df_fixed['score_ens'].mean():.4f}")
        print(f"  score_ens == 0.0: {(df_fixed['score_ens'] == 0.0).sum()}행")

    else:
        print(f"⚠️ 잔여 결측치: {final_missing}개")

if __name__ == "__main__":
    main()
