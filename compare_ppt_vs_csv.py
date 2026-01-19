from pathlib import Path

import pandas as pd

print('=== PPT 가이드 vs 실제 CSV 데이터 비교 ===')

# PPT 가이드에 있는 데이터
ppt_data = {
    'bt20_short': {'total_return': 4.90, 'sharpe': 0.26},
    'bt120_long': {'total_return': 2.88, 'sharpe': 0.18},
    'bt120_ens': {'total_return': 4.03, 'sharpe': 0.22}
}

print('PPT 가이드 데이터:')
for strategy, metrics in ppt_data.items():
    print(f'{strategy}: 총수익률 {metrics["total_return"]}%, Sharpe {metrics["sharpe"]}')

print()

# 실제 CSV 파일 데이터 분석
csv_path = Path('data/dummy_kospi200_pr_tabs_4lines_2023_2024_v5.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)

    print('CSV 파일 구조:')
    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Horizon_days:', sorted(df['horizon_days'].unique()))

    # 20일 horizon 데이터로 분석 (가장 기본적인 기간)
    df_20d = df[df['horizon_days'] == 20].copy()

    print(f'\n20일 horizon 데이터 (2023-2024, {len(df_20d)}개월):')

    # 실제 최종 누적 수익률 확인
    last_row = df_20d.iloc[-1]  # 2024-12 데이터
    print('최종 누적 수익률 (2024-12):')
    print(f'KOSPI200: {last_row["kospi200_pr_cum_return_pct"]:.2f}%')
    print(f'Short(bt20_short): {last_row["short_cum_return_pct"]:.2f}%')
    print(f'Long(bt120_long): {last_row["long_cum_return_pct"]:.2f}%')
    print(f'Mix(bt120_ens): {last_row["mix_cum_return_pct"]:.2f}%')

    # 월별 수익률 계산 및 통계
    print('\n월별 수익률 통계 (20일 horizon):')
    for col in ['short_mret_pct', 'long_mret_pct', 'mix_mret_pct']:
        returns = df_20d[col].values
        avg_return = returns.mean()
        std_return = returns.std()
        sharpe = avg_return / std_return if std_return > 0 else 0
        total_return = (1 + returns/100).prod() - 1
        total_return_pct = total_return * 100

        strategy_name = col.replace('mret_pct', '').replace('short', 'bt20_short').replace('long', 'bt120_long').replace('mix', 'bt120_ens')
        print(f'{strategy_name}:')
        print(f'  평균 월수익률: {avg_return:.2f}%')
        print(f'  표준편차: {std_return:.2f}%')
        print(f'  Sharpe 비율: {sharpe:.3f}')
        print(f'  총수익률: {total_return_pct:.2f}%')
        print()

    # PPT 데이터와 비교
    print('=== PPT vs 실제 데이터 비교 ===')
    csv_calculated = {
        'bt20_short': {'total_return': (1 + df_20d['short_mret_pct']/100).prod() - 1, 'sharpe': (df_20d['short_mret_pct']/100).mean() / (df_20d['short_mret_pct']/100).std()},
        'bt120_long': {'total_return': (1 + df_20d['long_mret_pct']/100).prod() - 1, 'sharpe': (df_20d['long_mret_pct']/100).mean() / (df_20d['long_mret_pct']/100).std()},
        'bt120_ens': {'total_return': (1 + df_20d['mix_mret_pct']/100).prod() - 1, 'sharpe': (df_20d['mix_mret_pct']/100).mean() / (df_20d['mix_mret_pct']/100).std()}
    }

    for strategy in ppt_data.keys():
        ppt_return = ppt_data[strategy]['total_return']
        ppt_sharpe = ppt_data[strategy]['sharpe']
        csv_return = csv_calculated[strategy]['total_return'] * 100
        csv_sharpe = csv_calculated[strategy]['sharpe']

        return_diff = abs(ppt_return - csv_return)
        sharpe_diff = abs(ppt_sharpe - csv_sharpe)

        status = '✅ 일치' if return_diff < 0.1 and sharpe_diff < 0.01 else '❌ 불일치'

        print(f'{strategy}:')
        print(f'  PPT - 총수익률: {ppt_return:.2f}%, Sharpe: {ppt_sharpe:.3f}')
        print(f'  CSV - 총수익률: {csv_return:.2f}%, Sharpe: {csv_sharpe:.3f}')
        print(f'  차이: 총수익률 {return_diff:.2f}%, Sharpe {sharpe_diff:.3f} - {status}')
        print()

else:
    print('CSV 파일을 찾을 수 없습니다.')
