from pathlib import Path

import pandas as pd

# 벤치마크 대비 성과 계산
returns_path = Path('data/strategies_daily_returns_holdout.csv')
if returns_path.exists():
    df_returns = pd.read_csv(returns_path)

    # KOSPI200과 전략들의 누적 수익률 계산
    strategies = ['KOSPI200', 'BT120 앙상블 (120일)', 'BT120 장기 (120일)', 'BT20 앙상블 (20일)', 'BT20 단기 (20일)']

    print('홀드아웃 기간 벤치마크 대비 성과:')
    print('기간: 2023-01-01 ~ 2024-10-31')
    print()

    kospi_cum = None
    for col in strategies:
        if col in df_returns.columns:
            # 마지막 값에서 첫 번째 값 빼기 (누적 수익률)
            final_val = df_returns[col].iloc[-1]
            initial_val = df_returns[col].iloc[0] if df_returns[col].iloc[0] != 0 else 1

            if col == 'KOSPI200':
                kospi_cum = (final_val - initial_val) / initial_val
                print(f'{col}: {kospi_cum:.2%}')
            else:
                strategy_cum = (final_val - initial_val) / initial_val
                excess = strategy_cum - kospi_cum
                print(f'{col}: {strategy_cum:.2%} (KOSPI200 대비 {excess:+.2%})')

    print()
    print('그래프용 데이터 (마지막 30일 추이):')
    recent_data = df_returns.tail(30)[['date', 'KOSPI200', 'BT120 장기 (120일)']]
    for _, row in recent_data.tail(5).iterrows():
        date = row['date']
        kospi = row['KOSPI200']
        bt120 = row['BT120 장기 (120일)']
        print(f'{date}, KOSPI200: {kospi:.4f}, BT120_long: {bt120:.4f}')
else:
    print('데이터 파일을 찾을 수 없습니다')
