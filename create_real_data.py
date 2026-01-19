import pandas as pd
from pathlib import Path

print('=== 실제 데이터로 교체할 더미 파일 분석 ===')

# 더미 파일 구조 확인
dummy_path = Path('data/dummy_kospi200_pr_tabs_4lines_2023_2024_v5.csv')
if dummy_path.exists():
    df_dummy = pd.read_csv(dummy_path)
    print('더미 파일 shape:', df_dummy.shape)
    print('컬럼들:', list(df_dummy.columns))
    print('고유 horizon_days:', sorted(df_dummy['horizon_days'].unique()))
    print('기간 범위:', df_dummy['month'].min(), '~', df_dummy['month'].max())

    # 샘플 데이터
    print('\n샘플 데이터 (첫 5행):')
    print(df_dummy.head().to_string(index=False))

print('\n=== 실제 데이터 생성 ===')

# 실제 가격 데이터에서 KOSPI200 월별 수익률 계산
price_path = Path('data/interim/dataset_daily.parquet')
if price_path.exists():
    df_price = pd.read_parquet(price_path)
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price['month'] = df_price['date'].dt.to_period('M')

    # KOSPI200 월별 수익률 계산 (첫 번째와 마지막 가격으로)
    kospi_monthly = df_price.groupby('month').agg({
        'close': ['first', 'last']
    }).reset_index()
    kospi_monthly.columns = ['month', 'first_price', 'last_price']
    kospi_monthly['kospi200_mret_pct'] = (kospi_monthly['last_price'] / kospi_monthly['first_price'] - 1) * 100

    print('실제 KOSPI200 월별 수익률 계산 완료')

# 전략 수익률 데이터 로드
strategy_path = Path('data/strategies_daily_returns_holdout.csv')
if strategy_path.exists():
    df_strategy = pd.read_csv(strategy_path)
    df_strategy['date'] = pd.to_datetime(df_strategy['date'])
    df_strategy['month'] = df_strategy['date'].dt.to_period('M')

    print('전략 수익률 데이터 로드 완료')

    # 실제 데이터로 더미 데이터 교체
    if dummy_path.exists() and price_path.exists():
        # 더미 데이터 구조 유지하면서 실제 데이터로 교체
        df_real = df_dummy.copy()

        # 각 horizon_days별로 데이터 교체
        for horizon in df_dummy['horizon_days'].unique():
            mask = df_real['horizon_days'] == horizon
            months = df_real.loc[mask, 'month']

            for month_str in months:
                try:
                    month_period = pd.Period(month_str, 'M')

                    # KOSPI200 실제 수익률
                    kospi_mask = kospi_monthly['month'] == month_period
                    if kospi_mask.any():
                        kospi_return = kospi_monthly.loc[kospi_mask, 'kospi200_mret_pct'].iloc[0]
                        df_real.loc[(df_real['month'] == month_str) & (df_real['horizon_days'] == horizon), 'kospi200_pr_mret_pct'] = kospi_return

                    # 전략별 실제 수익률 (월별 합계)
                    month_mask = df_strategy['month'] == month_period
                    if month_mask.any():
                        month_data = df_strategy[month_mask]

                        # short 전략 (BT20 단기)
                        if 'BT20 단기 (20일)' in df_strategy.columns:
                            short_return = month_data['BT20 단기 (20일)'].sum() * 100
                            df_real.loc[(df_real['month'] == month_str) & (df_real['horizon_days'] == horizon), 'short_mret_pct'] = short_return

                        # long 전략 (BT120 장기)
                        if 'BT120 장기 (120일)' in df_strategy.columns:
                            long_return = month_data['BT120 장기 (120일)'].sum() * 100
                            df_real.loc[(df_real['month'] == month_str) & (df_real['horizon_days'] == horizon), 'long_mret_pct'] = long_return

                        # mix 전략 (BT120 앙상블)
                        if 'BT120 앙상블 (120일)' in df_strategy.columns:
                            mix_return = month_data['BT120 앙상블 (120일)'].sum() * 100
                            df_real.loc[(df_real['month'] == month_str) & (df_real['horizon_days'] == horizon), 'mix_mret_pct'] = mix_return

                except Exception as e:
                    print(f'월 {month_str} 처리 중 오류: {e}')
                    continue

        # 누적 수익률 재계산
        for horizon in df_dummy['horizon_days'].unique():
            horizon_data = df_real[df_real['horizon_days'] == horizon].copy()
            horizon_data = horizon_data.sort_values('month')

            # 누적 수익률 계산
            horizon_data['kospi200_pr_cum_return_pct'] = (1 + horizon_data['kospi200_pr_mret_pct'] / 100).cumprod() - 1
            horizon_data['short_cum_return_pct'] = (1 + horizon_data['short_mret_pct'] / 100).cumprod() - 1
            horizon_data['long_cum_return_pct'] = (1 + horizon_data['long_mret_pct'] / 100).cumprod() - 1
            horizon_data['mix_cum_return_pct'] = (1 + horizon_data['mix_mret_pct'] / 100).cumprod() - 1

            # 백분율로 변환
            for col in ['kospi200_pr_cum_return_pct', 'short_cum_return_pct', 'long_cum_return_pct', 'mix_cum_return_pct']:
                horizon_data[col] = horizon_data[col] * 100

            # 원본 데이터에 업데이트
            for idx, row in horizon_data.iterrows():
                mask = (df_real['month'] == row['month']) & (df_real['horizon_days'] == horizon)
                df_real.loc[mask, 'kospi200_pr_cum_return_pct'] = row['kospi200_pr_cum_return_pct']
                df_real.loc[mask, 'short_cum_return_pct'] = row['short_cum_return_pct']
                df_real.loc[mask, 'long_cum_return_pct'] = row['long_cum_return_pct']
                df_real.loc[mask, 'mix_cum_return_pct'] = row['mix_cum_return_pct']

        # 실제 데이터 파일 저장
        real_file_path = Path('data/real_kospi200_pr_tabs_4lines_2023_2024_v5.csv')
        df_real.to_csv(real_file_path, index=False)
        print(f'\n실제 데이터 파일 생성 완료: {real_file_path}')
        print('실제 데이터 샘플:')
        print(df_real.head(10).to_string(index=False))

        # 비교를 위한 통계
        print('\n=== 데이터 비교 ===')
        print('더미 vs 실제 데이터 비교 (첫 5개월, 20일 horizon):')
        dummy_20d = df_dummy[df_dummy['horizon_days'] == 20].head()
        real_20d = df_real[df_real['horizon_days'] == 20].head()

        comparison_cols = ['month', 'kospi200_pr_mret_pct', 'short_mret_pct', 'long_mret_pct', 'mix_mret_pct']
        print('더미 데이터:')
        print(dummy_20d[comparison_cols].to_string(index=False))
        print('\n실제 데이터:')
        print(real_20d[comparison_cols].to_string(index=False))

else:
    print('필요한 데이터 파일들이 없습니다.')