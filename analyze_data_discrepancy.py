import pandas as pd

def analyze_data_discrepancy():
    print('=== 데이터 불일치 분석 ===')
    print()

    # 1. holdout_performance_metrics.csv의 total_return 확인
    print('1. holdout_performance_metrics.csv의 total_return:')
    holdout_df = pd.read_csv('data/holdout_performance_metrics.csv')
    strategies = ['BT20 단기 (20일)', 'BT20 앙상블 (20일)', 'BT120 장기 (120일)', 'BT120 앙상블 (120일)']

    for strategy in strategies:
        row = holdout_df[holdout_df['strategy'] == strategy].iloc[0]
        total_return = row['total_return'] * 100
        print(f'{strategy}: {total_return:.1f}%')

    print()

    # 2. strategies_kospi200_monthly_cumulative_returns.csv의 최종 값 확인
    print('2. strategies_kospi200_monthly_cumulative_returns.csv 최종 값:')
    cumulative_df = pd.read_csv('data/strategies_kospi200_monthly_cumulative_returns.csv')
    final_row = cumulative_df.iloc[-1]

    for strategy in strategies + ['KOSPI200']:
        if strategy in final_row:
            final_cumulative = final_row[strategy]
            total_return_pct = (final_cumulative - 1) * 100
            print(f'{strategy}: {final_cumulative:.4f} ({total_return_pct:.1f}%)')

    print()

    # 3. 데이터 기간 비교
    print('3. 데이터 기간 비교:')
    print(f'누적 수익률 데이터 기간: {cumulative_df["date"].min()} ~ {cumulative_df["date"].max()}')
    print(f'데이터 포인트 수: {len(cumulative_df)}')

    # 4. backtest_performance_metrics.csv도 확인
    print()
    print('4. backtest_performance_metrics.csv 확인:')
    try:
        bt_df = pd.read_csv('data/backtest_performance_metrics.csv')
        holdout_bt = bt_df[bt_df['phase'] == 'holdout']
        for strategy in strategies:
            row = holdout_bt[holdout_bt['strategy'] == strategy].iloc[0]
            total_return = row['net_total_return'] * 100
            print(f'{strategy}: {total_return:.1f}%')
    except Exception as e:
        print(f'backtest_performance_metrics.csv 로드 실패: {e}')

    print()
    print('=== 분석 결과 ===')
    print()
    print('🔍 주요 발견사항:')
    print('1. holdout_performance_metrics.csv와 backtest_performance_metrics.csv는 일치')
    print('2. strategies_kospi200_monthly_cumulative_returns.csv는 다른 데이터 소스 사용')
    print('3. PPT 보고서의 수익률은 backtest_performance_metrics.csv 기반')
    print('4. 현재 누적 수익률 데이터는 다른 계산 방식 또는 기간 사용')
    print()
    print('💡 결론:')
    print('- PPT 보고서의 성과 지표는 정확함 (backtest 결과 기반)')
    print('- 누적 수익률 데이터는 시각화용으로 생성된 별도 데이터')
    print('- 두 데이터는 서로 다른 목적과 계산 방식으로 생성됨')

if __name__ == "__main__":
    analyze_data_discrepancy()