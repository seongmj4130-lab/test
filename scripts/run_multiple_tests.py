#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Track A/B 다중 실행 테스트 스크립트
각 전략을 3번씩 실행해서 결과를 비교
"""

from src.tracks.track_b.backtest_service import run_backtest_strategy
import pandas as pd
import time

def run_strategy_multiple_times(strategy_name, times=3):
    """전략을 여러 번 실행하고 결과를 수집"""
    results = []

    for i in range(times):
        print(f'=== {strategy_name} 실행 {i+1}/{times} ===')
        try:
            result = run_backtest_strategy(strategy_name)
            results.append(result)
            time.sleep(1)  # 실행 간격
        except Exception as e:
            print(f'실행 {i+1} 실패: {e}')
            results.append(None)

    return results

def extract_metrics(result):
    """결과에서 메트릭 추출"""
    if result is None or 'bt_metrics' not in result or result['bt_metrics'] is None:
        return None

    metrics = result['bt_metrics']
    if not isinstance(metrics, pd.DataFrame) or len(metrics) == 0:
        return None

    # 메트릭이 Series인 경우 float로 변환
    def get_metric_value(col_name):
        value = metrics.get(col_name)
        if isinstance(value, pd.Series) and len(value) > 0:
            return float(value.iloc[0])
        elif pd.notna(value):
            return float(value)
        else:
            return None

    return {
        'net_sharpe': get_metric_value('net_sharpe'),
        'net_total_return': get_metric_value('net_total_return'),
        'net_cagr': get_metric_value('net_cagr'),
        'net_mdd': get_metric_value('net_mdd'),
        'net_hit_ratio': get_metric_value('net_hit_ratio'),
        'ic': get_metric_value('ic'),
        'rank_ic': get_metric_value('rank_ic')
    }

def compare_results(strategy_name, results):
    """결과 비교 및 출력"""
    print(f'\n=== {strategy_name} 3회 실행 결과 비교 ===')

    extracted_results = []
    for i, result in enumerate(results, 1):
        metrics = extract_metrics(result)
        if metrics:
            print(f'실행 {i}:')
            for key, value in metrics.items():
                if pd.notna(value):
                    print(f'  {key}: {value:.4f}')
                else:
                    print(f'  {key}: N/A')
            extracted_results.append(metrics)
        else:
            print(f'실행 {i}: 메트릭 데이터 없음')
            extracted_results.append(None)

    # 평균 계산
    if extracted_results and all(r is not None for r in extracted_results):
        print(f'\n{strategy_name} 평균:')
        for key in extracted_results[0].keys():
            values = [r[key] for r in extracted_results if r and pd.notna(r.get(key))]
            if values:
                avg = sum(values) / len(values)
                std = pd.Series(values).std()
                print(f'  {key}: {avg:.4f} ± {std:.4f}')

if __name__ == '__main__':
    strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']

    for strategy in strategies:
        results = run_strategy_multiple_times(strategy, 3)
        compare_results(strategy, results)
        print('\n' + '='*50 + '\n')