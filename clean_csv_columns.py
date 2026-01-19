#!/usr/bin/env python3
"""
CSV 파일에서 불필요한 컬럼들 제거
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def clean_csv_columns():
    """warnings, timestamp, strategy_name 컬럼 제거"""

    # 최신 백분율 변환 파일 찾기
    results_dir = Path('results')
    csv_files = list(results_dir.glob('dynamic_period_backtest_results_percentage_*.csv'))

    if not csv_files:
        print("❌ 백분율 변환 파일을 찾을 수 없습니다.")
        return

    # 최신 파일 선택
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 정리할 파일: {latest_file}")

    # 데이터 로드
    df = pd.read_csv(latest_file)
    print(f"📈 원본 데이터: {df.shape[0]}행 × {df.shape[1]}열")

    # 컬럼 목록 확인
    print(f"📋 원본 컬럼들: {list(df.columns)}")

    # 제거할 컬럼들
    columns_to_drop = ['warnings', 'timestamp', 'strategy_name']

    # 실제로 존재하는 컬럼만 제거
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"🗑️ 제거할 컬럼들: {existing_columns_to_drop}")

    # 컬럼 제거
    df_cleaned = df.drop(columns=existing_columns_to_drop)
    print(f"✅ 정리된 데이터: {df_cleaned.shape[0]}행 × {df_cleaned.shape[1]}열")
    print(f"📋 남은 컬럼들: {list(df_cleaned.columns)}")

    # 정리된 데이터 샘플 출력
    print("\n🔍 정리된 데이터 샘플:")
    print(df_cleaned.head(3).to_string(index=False, float_format='%.2f'))

    # 새 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"dynamic_period_backtest_clean_{timestamp}.csv"
    output_path = results_dir / output_filename

    # CSV 저장 (float_format 지정)
    df_cleaned.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\n💾 정리된 결과 저장: {output_path}")

    # 정리 결과 요약
    print("\n📊 정리 결과 요약:")
    print("=" * 40)
    print(f"원본 파일: {latest_file.name}")
    print(f"정리 파일: {output_filename}")
    print(f"제거된 컬럼 수: {len(existing_columns_to_drop)}")
    print(f"남은 컬럼 수: {len(df_cleaned.columns)}")
    print(f"데이터 행 수: {len(df_cleaned)}")

    print("\n✅ CSV 컬럼 정리 완료!")

if __name__ == "__main__":
    clean_csv_columns()
