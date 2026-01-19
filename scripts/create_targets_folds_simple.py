# -*- coding: utf-8 -*-
"""
targets_and_folds.parquet 생성 (간단 버전)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("🎯 targets_and_folds.parquet 생성 시작")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    try:
        # 기존 데이터 로드
        cv_short = pd.read_parquet(interim_dir / 'cv_folds_short.parquet')
        cv_long = pd.read_parquet(interim_dir / 'cv_folds_long.parquet')
        dataset = pd.read_parquet(interim_dir / 'dataset_daily.parquet')

        print("✅ 데이터 로드 완료")

        # 타겟 변수 식별
        target_cols = [col for col in dataset.columns if 'ret_fwd' in col and 'd' in col]
        print(f"📈 타겟 변수: {target_cols}")

        # 데이터 생성
        data = []

        # 단기 타겟
        short_targets = [col for col in target_cols if '20d' in col]
        for _, row in cv_short.iterrows():
            for target in short_targets:
                data.append({
                    'date': row['date'],
                    'fold': row['fold'],
                    'set': row['set'],
                    'target': target,
                    'horizon': 'short'
                })

        # 장기 타겟
        long_targets = [col for col in target_cols if '120d' in col]
        for _, row in cv_long.iterrows():
            for target in long_targets:
                data.append({
                    'date': row['date'],
                    'fold': row['fold'],
                    'set': row['set'],
                    'target': target,
                    'horizon': 'long'
                })

        # 저장
        df = pd.DataFrame(data)
        output_file = interim_dir / 'targets_and_folds.parquet'
        df.to_parquet(output_file, index=False)

        print(f"✅ 생성 완료: {len(df)}행")
        print(f"💾 저장: {output_file}")

    except Exception as e:
        print(f"❌ 실패: {str(e)}")

if __name__ == "__main__":
    main()
