# -*- coding: utf-8 -*-
"""
targets_and_folds.parquet 생성 (올바른 버전)
"""

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def generate_date_range(start_date, end_date):
    """두 날짜 사이의 모든 날짜 생성"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates

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

        # 단기 CV 처리
        print("\n🔄 단기 CV 처리 중...")
        short_targets = [col for col in target_cols if '20d' in col]
        fold_count = 0

        for _, row in cv_short.iterrows():
            test_start = pd.to_datetime(row['test_start'])
            test_end = pd.to_datetime(row['test_end'])
            fold_id = row['fold_id']
            segment = row['segment']

            # 테스트 기간의 모든 날짜 생성
            test_dates = generate_date_range(test_start, test_end)

            for date in test_dates:
                for target in short_targets:
                    data.append({
                        'date': date,
                        'fold': fold_count,
                        'set': segment,
                        'target': target,
                        'horizon': 'short'
                    })
            fold_count += 1

        # 장기 CV 처리
        print("🔄 장기 CV 처리 중...")
        long_targets = [col for col in target_cols if '120d' in col]

        for _, row in cv_long.iterrows():
            test_start = pd.to_datetime(row['test_start'])
            test_end = pd.to_datetime(row['test_end'])
            fold_id = row['fold_id']
            segment = row['segment']

            # 테스트 기간의 모든 날짜 생성
            test_dates = generate_date_range(test_start, test_end)

            for date in test_dates:
                for target in long_targets:
                    data.append({
                        'date': date,
                        'fold': fold_count,
                        'set': segment,
                        'target': target,
                        'horizon': 'long'
                    })
            fold_count += 1

        # 저장
        df = pd.DataFrame(data)
        output_file = interim_dir / 'targets_and_folds.parquet'
        df.to_parquet(output_file, index=False)

        print(f"\n✅ 생성 완료!")
        print(f"📊 총 행 수: {len(df):,}")
        print(f"📅 유니크 날짜: {df['date'].nunique()}")
        print(f"🎯 타겟 변수: {df['target'].unique()}")
        print(f"📂 세트 분포: {df['set'].value_counts().to_dict()}")
        print(f"💾 저장: {output_file}")

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()