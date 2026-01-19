import pandas as pd
from pathlib import Path

print('=== 실제 월별 수익률 정확히 계산 ===')

# 실제 가격 데이터에서 정확한 월별 수익률 계산
price_path = Path('data/interim/dataset_daily.parquet')
if price_path.exists():
    df_price = pd.read_parquet(price_path)
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price['month'] = df_price['date'].dt.to_period('M')

    # KOSPI200 월별 수익률 정확히 계산
    kospi_monthly = df_price.groupby('month').agg({
        'close': ['first', 'last']
    }).reset_index()
    kospi_monthly.columns = ['month', 'first_price', 'last_price']
    kospi_monthly['kospi200_mret_pct'] = (kospi_monthly['last_price'] / kospi_monthly['first_price'] - 1) * 100

    print('정확한 KOSPI200 월별 수익률:')
    print(kospi_monthly[['month', 'kospi200_mret_pct']].head(10).to_string(index=False))

# 전략 수익률도 월별로 정확히 계산
strategy_path = Path('data/strategies_daily_returns_holdout.csv')
if strategy_path.exists():
    df_strategy = pd.read_csv(strategy_path)
    df_strategy['date'] = pd.to_datetime(df_strategy['date'])
    df_strategy['month'] = df_strategy['date'].dt.to_period('M')

    # 월별 첫날과 마지막날의 누적 수익률 계산
    monthly_strategy_returns = {}

    for month in df_strategy['month'].unique():
        month_data = df_strategy[df_strategy['month'] == month].copy()

        if len(month_data) > 0:
            # 월별 전략 수익률 계산 (1 + 일별수익률의 곱 - 1)
            strategy_returns = {}
            for col in ['BT20 단기 (20일)', 'BT120 장기 (120일)', 'BT120 앙상블 (120일)']:
                if col in month_data.columns:
                    # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
                    monthly_return = (1 + month_data[col]).prod() - 1
                    strategy_returns[col] = monthly_return * 100

            monthly_strategy_returns[str(month)] = strategy_returns

    print('\n정확한 전략 월별 수익률 (샘플):')
    count = 0
    for month, returns in monthly_strategy_returns.items():
        if count < 5:
            print(f'{month}: {returns}')
            count += 1

# 더미 파일을 실제 데이터로 교체
dummy_path = Path('data/dummy_kospi200_pr_tabs_4lines_2023_2024_v5.csv')
if dummy_path.exists() and price_path.exists() and strategy_path.exists():
    df_dummy = pd.read_csv(dummy_path)
    df_real = df_dummy.copy()

    print('\n=== 더미 데이터를 실제 데이터로 교체 ===')

    for idx, row in df_dummy.iterrows():
        month_str = row['month']
        horizon = row['horizon_days']

        try:
            month_period = pd.Period(month_str, 'M')

            # KOSPI200 실제 수익률
            kospi_row = kospi_monthly[kospi_monthly['month'] == month_period]
            if len(kospi_row) > 0:
                kospi_return = kospi_row['kospi200_mret_pct'].iloc[0]
                df_real.loc[idx, 'kospi200_pr_mret_pct'] = kospi_return

            # 전략 실제 수익률
            if str(month_period) in monthly_strategy_returns:
                strategy_data = monthly_strategy_returns[str(month_period)]

                if 'BT20 단기 (20일)' in strategy_data:
                    df_real.loc[idx, 'short_mret_pct'] = strategy_data['BT20 단기 (20일)']
                if 'BT120 장기 (120일)' in strategy_data:
                    df_real.loc[idx, 'long_mret_pct'] = strategy_data['BT120 장기 (120일)']
                if 'BT120 앙상블 (120일)' in strategy_data:
                    df_real.loc[idx, 'mix_mret_pct'] = strategy_data['BT120 앙상블 (120일)']

        except Exception as e:
            print(f'월 {month_str} 처리 중 오류: {e}')
            continue

    # 누적 수익률 재계산
    for horizon in df_dummy['horizon_days'].unique():
        horizon_data = df_real[df_real['horizon_days'] == horizon].copy()
        horizon_data = horizon_data.sort_values('month')

        # 누적 수익률 계산
        cum_kospi = 1.0
        cum_short = 1.0
        cum_long = 1.0
        cum_mix = 1.0

        for idx in horizon_data.index:
            row = horizon_data.loc[idx]

            cum_kospi *= (1 + row['kospi200_pr_mret_pct'] / 100)
            cum_short *= (1 + row['short_mret_pct'] / 100)
            cum_long *= (1 + row['long_mret_pct'] / 100)
            cum_mix *= (1 + row['mix_mret_pct'] / 100)

            df_real.loc[idx, 'kospi200_pr_cum_return_pct'] = (cum_kospi - 1) * 100
            df_real.loc[idx, 'short_cum_return_pct'] = (cum_short - 1) * 100
            df_real.loc[idx, 'long_cum_return_pct'] = (cum_long - 1) * 100
            df_real.loc[idx, 'mix_cum_return_pct'] = (cum_mix - 1) * 100

    # 실제 데이터 파일 저장 (덮어쓰기)
    df_real.to_csv(dummy_path, index=False)
    print(f'\n더미 파일을 실제 데이터로 교체 완료: {dummy_path}')

    print('\n교체된 실제 데이터 샘플 (첫 10행):')
    print(df_real.head(10).to_string(index=False))

    # 검증
    print('\n=== 검증: 실제 vs 더미 비교 ===')
    print('20일 horizon, 2023년 데이터:')
    real_2023 = df_real[(df_real['horizon_days'] == 20) & (df_real['month'].str.startswith('2023'))]
    print('실제 데이터 평균 수익률:')
    kospi_avg = real_2023['kospi200_pr_mret_pct'].mean()
    short_avg = real_2023['short_mret_pct'].mean()
    long_avg = real_2023['long_mret_pct'].mean()
    mix_avg = real_2023['mix_mret_pct'].mean()
    print(f'KOSPI200: {kospi_avg:.2f}%')
    print(f'Short: {short_avg:.2f}%')
    print(f'Long: {long_avg:.2f}%')
    print(f'Mix: {mix_avg:.2f}%')