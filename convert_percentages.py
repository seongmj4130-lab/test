#!/usr/bin/env python3
"""
백테스트 결과에서 백분율 변환 및 새 CSV 저장
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def convert_percentages_to_csv():
    """백분율이 필요한 지표들을 %로 변환하여 새 CSV 저장"""

    # 최신 결과 파일 찾기
    results_dir = Path('results')
    csv_files = list(results_dir.glob('dynamic_period_backtest_results_*.csv'))

    if not csv_files:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        return

    # 최신 파일 선택
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 변환할 파일: {latest_file}")

    # 데이터 로드
    df = pd.read_csv(latest_file)
    print(f"📈 {len(df)}개 결과 로드됨")

    # 백분율로 변환해야 할 컬럼들
    percentage_columns = {
        'cagr': 'CAGR (%)',           # 0.3399 -> 33.99%
        'total_return': 'Total Return (%)',  # 7.3987 -> 739.87% (또는 그대로)
        'mdd': 'MDD (%)',             # -0.1715 -> -17.15%
        'hit_ratio': 'Hit Ratio (%)'   # 0.5201 -> 52.01%
    }

    # 변환 전 데이터 샘플 출력
    print("\n🔍 변환 전 샘플 데이터:")
    sample_cols = ['strategy_name', 'holding_days', 'cagr', 'total_return', 'mdd', 'hit_ratio']
    print(df[sample_cols].head(3).to_string(index=False))

    # 백분율 변환 수행
    df_converted = df.copy()

    for col, new_name in percentage_columns.items():
        if col in df.columns:
            if col == 'total_return':
                # total_return은 이미 백분율로 표현된 것 같지만, 일관성을 위해 *100
                df_converted[col] = df[col] * 100
            else:
                # 다른 컬럼들은 소수점에서 백분율로 변환
                df_converted[col] = df[col] * 100

            # 컬럼명도 변경
            df_converted = df_converted.rename(columns={col: new_name})

    # 변환 후 데이터 샘플 출력
    print("\n✅ 변환 후 샘플 데이터:")
    converted_cols = ['strategy_name', 'holding_days', 'CAGR (%)', 'Total Return (%)', 'MDD (%)', 'Hit Ratio (%)']
    print(df_converted[converted_cols].head(3).to_string(index=False, float_format='%.2f'))

    # 새 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"dynamic_period_backtest_results_percentage_{timestamp}.csv"
    output_path = results_dir / output_filename

    # CSV 저장
    df_converted.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\n💾 백분율 변환 결과 저장: {output_path}")

    # 변환된 데이터의 통계 출력
    print("\n📊 변환된 데이터 요약:")
    print("=" * 60)

    for strategy in df_converted['strategy_name'].unique():
        strategy_data = df_converted[df_converted['strategy_name'] == strategy]
        print(f"\n{strategy} 전략:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

    print("\n✅ 백분율 변환 완료!")
    print(f"📁 결과 파일: {output_path}")

if __name__ == "__main__":
    convert_percentages_to_csv()