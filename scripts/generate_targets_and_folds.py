# -*- coding: utf-8 -*-
"""
targets_and_folds.parquet 생성 스크립트

L4 CV 분할 단계에서 필요한 targets_and_folds.parquet 파일을 생성합니다.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def generate_targets_and_folds():
    """targets_and_folds.parquet 생성"""
    print("🎯 targets_and_folds.parquet 생성 시작")
    print("="*60)

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    try:
        # 기존 CV 폴드 데이터 로드
        cv_short = pd.read_parquet(interim_dir / 'cv_folds_short.parquet')
        cv_long = pd.read_parquet(interim_dir / 'cv_folds_long.parquet')
        dataset = pd.read_parquet(interim_dir / 'dataset_daily.parquet')

        print("✅ 데이터 로드 완료")
        print(f"  단기 CV: {len(cv_short)}개 날짜")
        print(f"  장기 CV: {len(cv_long)}개 날짜")
        print(f"  데이터셋: {len(dataset)}행 x {len(dataset.columns)}열")

        # 타겟 변수 식별 (수익률 컬럼)
        target_cols = [col for col in dataset.columns if 'ret_fwd' in col and 'd' in col]
        print(f"\n📈 식별된 타겟 변수: {target_cols}")

        # targets_and_folds 데이터 생성
        targets_folds_data = []

        # 단기 타겟 (20d)
        short_targets = [col for col in target_cols if '20d' in col]
        if short_targets:
            for _, row in cv_short.iterrows():
                date = row['date']
                fold = row['fold']
                set_type = row['set']

                for target_col in short_targets:
                    targets_folds_data.append({
                        'date': date,
                        'fold': fold,
                        'set': set_type,
                        'target': target_col,
                        'horizon': 'short'
                    })

        # 장기 타겟 (120d)
        long_targets = [col for col in target_cols if '120d' in col]
        if long_targets:
            for _, row in cv_long.iterrows():
                date = row['date']
                fold = row['fold']
                set_type = row['set']

                for target_col in long_targets:
                    targets_folds_data.append({
                        'date': date,
                        'fold': fold,
                        'set': set_type,
                        'target': target_col,
                        'horizon': 'long'
                    })

        # DataFrame 생성 및 저장
        if targets_folds_data:
            targets_folds_df = pd.DataFrame(targets_folds_data)

            # 결과 검증
            print("\n📊 생성된 데이터 구조:")            print(f"  총 행 수: {len(targets_folds_df):,}")
            print(f"  유니크 날짜: {targets_folds_df['date'].nunique()}")
            print(f"  타겟 변수: {targets_folds_df['target'].unique()}")
            print(f"  호리즌 분포: {targets_folds_df['horizon'].value_counts().to_dict()}")
            print(f"  세트 분포: {targets_folds_df['set'].value_counts().to_dict()}")

            # 파일 저장
            output_file = interim_dir / 'targets_and_folds.parquet'
            targets_folds_df.to_parquet(output_file, index=False)

            print("\n✅ targets_and_folds.parquet 생성 완료")            print(f"💾 저장 위치: {output_file}")
            print(f"📏 파일 크기: {output_file.stat().st_size / 1024:.1f} KB")

            return targets_folds_df
        else:
            print("❌ 생성할 데이터가 없습니다.")
            return None

    except Exception as e:
        print(f"❌ 생성 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_generation():
    """생성 결과 검증"""
    print("\n🔍 생성 결과 검증")
    print("="*40)

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    targets_file = interim_dir / 'targets_and_folds.parquet'

    if targets_file.exists():
        df = pd.read_parquet(targets_file)
        print("✅ 파일 존재 확인")
        print(f"📊 데이터 구조: {len(df)}행 x {len(df.columns)}열")
        print(f"📋 컬럼: {list(df.columns)}")

        # 데이터 무결성 검증
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        print(".2f")

        # 호리즌별 분포
        horizon_dist = df['horizon'].value_counts()
        print(f"🎯 호리즌 분포: {horizon_dist.to_dict()}")

        # 세트별 분포
        set_dist = df['set'].value_counts()
        print(f"📂 세트 분포: {set_dist.to_dict()}")

        if missing_rate == 0 and len(df) > 0:
            print("✅ 데이터 무결성 검증 통과")
            return True
        else:
            print("⚠️ 데이터 품질 문제 발견")
            return False
    else:
        print("❌ 파일 생성 실패")
        return False

def main():
    """메인 함수"""
    print("🚀 targets_and_folds.parquet 생성 작업 시작")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 생성 실행
    result_df = generate_targets_and_folds()

    # 검증 실행
    if result_df is not None:
        success = verify_generation()

        if success:
            print("
🎉 targets_and_folds.parquet 생성 성공!"            print("📈 다음 단계: L6 결측치 처리 및 백테스트 재실행 준비 완료")
        else:
            print("
⚠️ 생성은 되었으나 품질 검증 실패"    else:
        print("
❌ 생성 실패"    print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()