import pandas as pd


def create_filtered_csv():
    """전략별 누적 비교 그래프용 필터링된 CSV 생성"""

    print("📊 전략별 누적 비교 그래프용 CSV 필터링 중...")

    # 원본 데이터 로드
    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')

    # 필요한 컬럼만 선택
    filtered_columns = [
        'year_month',
        'kospi_tr_cumulative_log_return',
        'bt20_단기_cumulative_log_return',
        'bt20_앙상블_cumulative_log_return',
        'bt120_장기_cumulative_log_return',
        'bt120_앙상블_cumulative_log_return'
    ]

    # 필터링된 데이터프레임 생성
    df_filtered = df[filtered_columns]

    # 새로운 CSV 파일로 저장
    output_file = 'data/ui_strategies_cumulative_comparison.csv'
    df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✅ 필터링된 CSV 생성: {output_file}")
    print(f"   • 원본 컬럼 수: {len(df.columns)}개")
    print(f"   • 필터링 후 컬럼 수: {len(df_filtered.columns)}개")
    print(f"   • 데이터 행 수: {len(df_filtered)}개")

    # 필터링된 데이터 미리보기
    print("\n📋 필터링된 데이터 미리보기:")
    print("-" * 80)
    print(df_filtered.head())

    # 컬럼별 기본 통계
    print("\n📊 컬럼별 기본 통계:")
    print("-" * 80)
    for col in filtered_columns[1:]:  # year_month 제외
        values = df_filtered[col]
        print(f"{col}:")
        print(".2f")
        print(".2f")
        print(".3f")
        print()

    return df_filtered

if __name__ == "__main__":
    create_filtered_csv()
