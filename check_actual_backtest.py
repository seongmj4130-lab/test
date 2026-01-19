from pathlib import Path

import pandas as pd

print('=== 실제 백테스트 결과 확인 ===')

# 실제 백테스트 성과 데이터 확인
perf_path = Path('data/track_b_performance_metrics.parquet')
if perf_path.exists():
    df_perf = pd.read_parquet(perf_path)

    print('백테스트 성과 데이터 구조:')
    print('Shape:', df_perf.shape)
    print('Columns:', list(df_perf.columns))
    print('Unique phases:', df_perf['phase'].unique())
    print('Unique strategies:', sorted(df_perf['strategy'].unique()))

    # Holdout 데이터만 필터링
    holdout_data = df_perf[df_perf['phase'] == 'Holdout']

    print('\nHoldout 기간 백테스트 결과:')
    for _, row in holdout_data.iterrows():
        strategy = row['strategy']
        total_return = row['total_return']
        sharpe = row['sharpe_ratio']
        cagr = row['cagr']
        mdd = row['mdd']

        print(f'{strategy}:')
        print(f'  총수익률: {total_return}')
        print(f'  Sharpe: {sharpe}')
        print(f'  CAGR: {cagr}')
        print(f'  MDD: {mdd}')
        print()

    print('PPT 가이드와 비교:')
    ppt_data = {
        'bt20_short': {'total_return': 4.90, 'sharpe': 0.26},
        'bt120_long': {'total_return': 2.88, 'sharpe': 0.18},
        'bt120_ens': {'total_return': 4.03, 'sharpe': 0.22}
    }

    for ppt_strategy, ppt_metrics in ppt_data.items():
        # 백테스트 데이터에서 해당 전략 찾기
        matching_rows = holdout_data[holdout_data['strategy'].str.contains(ppt_strategy.replace('bt', '').replace('_', ''))]
        if len(matching_rows) > 0:
            actual = matching_rows.iloc[0]
            actual_return = float(actual['total_return'].strip('%'))
            actual_sharpe = float(actual['sharpe_ratio'])

            return_diff = abs(ppt_metrics['total_return'] - actual_return)
            sharpe_diff = abs(ppt_metrics['sharpe'] - actual_sharpe)

            status = '✅ 일치' if return_diff < 0.1 and sharpe_diff < 0.01 else '❌ 불일치'

            print(f'{ppt_strategy}:')
            print(f'  PPT: {ppt_metrics["total_return"]}%, Sharpe {ppt_metrics["sharpe"]}')
            print(f'  실제: {actual_return}%, Sharpe {actual_sharpe}')
            print(f'  차이: 총수익률 {return_diff:.2f}%, Sharpe {sharpe_diff:.3f} - {status}')
        else:
            print(f'{ppt_strategy}: 실제 데이터에서 찾을 수 없음')
        print()

else:
    print('백테스트 성과 파일을 찾을 수 없습니다.')
