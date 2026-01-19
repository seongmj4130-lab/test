from pathlib import Path

import pandas as pd

# 백테스트 성과 데이터 확인
perf_path = Path('data/track_b_performance_metrics.parquet')
if perf_path.exists():
    df_perf = pd.read_parquet(perf_path)

    print('=== Track B 백테스트 성과 분석 ===')
    print('전체 전략별 총수익률 (Holdout 기간):')
    print()

    # Holdout 데이터만 필터링
    holdout_data = df_perf[df_perf['phase'] == 'Holdout']

    # 총수익률 기준 정렬
    sorted_data = holdout_data.sort_values('total_return', ascending=False)

    results = []
    for _, row in sorted_data.iterrows():
        strategy = row['strategy']
        total_return = row['total_return']
        sharpe = row['sharpe_ratio']
        cagr = row['cagr']
        mdd = row['mdd']

        results.append({
            'strategy': strategy,
            'total_return': total_return,
            'sharpe': sharpe,
            'cagr': cagr,
            'mdd': mdd
        })

        print(f'{strategy}:')
        print(f'  총수익률: {total_return}')
        print(f'  Sharpe: {sharpe}')
        print(f'  CAGR: {cagr}')
        print(f'  MDD: {mdd}')
        print()

    # 최고 성과 전략 선정
    best_strategy = max(results, key=lambda x: x['total_return'])
    print('🎯 총수익률 기준 최고 성과 전략:')
    print(f'전략: {best_strategy["strategy"]}')
    print(f'총수익률: {best_strategy["total_return"]}')
    print(f'Sharpe 비율: {best_strategy["sharpe"]}')
    print(f'CAGR: {best_strategy["cagr"]}')
    print(f'MDD: {best_strategy["mdd"]}')

    print()
    print('📊 전략별 순위 (총수익률 기준):')
    for i, result in enumerate(results, 1):
        print(f'{i}위: {result["strategy"]} ({result["total_return"]})')

    # 전략별 특징 분석
    print()
    print('📈 전략별 특징 분석:')
    for result in results:
        strategy = result['strategy']
        total_return = result['total_return']
        sharpe = result['sharpe']
        mdd = result['mdd']

        if 'short' in strategy:
            horizon = '단기 (20일)'
        elif 'long' in strategy:
            horizon = '장기 (120일)'
        else:
            horizon = '앙상블'

        risk_level = '높음' if abs(float(mdd.strip('%'))) > 35 else '중간' if abs(float(mdd.strip('%'))) > 20 else '낮음'

        print(f'{strategy}:')
        print(f'  유형: {horizon}')
        print(f'  수익률: {total_return}')
        print(f'  리스크: {mdd} ({risk_level})')
        print(f'  효율성: Sharpe {sharpe}')
        print()

else:
    print('성과 데이터 파일을 찾을 수 없습니다.')
