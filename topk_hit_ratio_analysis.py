#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-K 방향 적중률 분석 및 PPT 업데이트
"""

from pathlib import Path

import pandas as pd


def analyze_topk_hit_ratio_results():
    """Top-K 방향 적중률 분석 결과 정리"""

    print('=== Top-K 방향 적중률 분석 결과 ===')

    # 분석 결과 데이터 (실행 결과로부터)
    results = {
        '단기랭킹': {
            10: {'hit_ratio': 0.415, 'avg_return': -0.0065, 'samples': 104},
            20: {'hit_ratio': 0.427, 'avg_return': -0.0047, 'samples': 104},
            30: {'hit_ratio': 0.429, 'avg_return': -0.0045, 'samples': 104},
            50: {'hit_ratio': 0.438, 'avg_return': -0.0026, 'samples': 104}
        },
        '장기랭킹': {
            10: {'hit_ratio': 0.376, 'avg_return': 0.0275, 'samples': 104},
            20: {'hit_ratio': 0.396, 'avg_return': 0.0312, 'samples': 104},
            30: {'hit_ratio': 0.398, 'avg_return': 0.0229, 'samples': 104},
            50: {'hit_ratio': 0.397, 'avg_return': 0.0219, 'samples': 104}
        },
        '통합랭킹': {
            10: {'hit_ratio': 0.444, 'avg_return': 0.0018, 'samples': 104},
            20: {'hit_ratio': 0.442, 'avg_return': -0.0031, 'samples': 104},
            30: {'hit_ratio': 0.442, 'avg_return': -0.0016, 'samples': 104},
            50: {'hit_ratio': 0.450, 'avg_return': -0.0004, 'samples': 104}
        }
    }

    # 기존 hit_ratio와 비교
    existing_hit_ratios = {
        '단기랭킹': {'dev': 57.3, 'holdout': 43.5},
        '장기랭킹': {'dev': 50.5, 'holdout': 49.2},
        '통합랭킹': {'dev': 52.0, 'holdout': 48.0}
    }

    print('\n=== 전략별 Top-K 방향 적중률 ===')
    for strategy, topk_results in results.items():
        print(f'\n{strategy}:')
        for top_k, data in topk_results.items():
            hit_ratio_pct = data['hit_ratio'] * 100
            avg_return_pct = data['avg_return'] * 100
            print(f'  Top-{top_k}: 방향적중률 {hit_ratio_pct:.1f}%, '
                  f'평균수익률 {avg_return_pct:+.2f}% ({data["samples"]}일)')

    print('\n=== 기존 hit_ratio vs 새로운 방향 적중률 비교 ===')
    print('| 전략 | 기존 Hit Ratio (Holdout) | 새 방향적중률 (Top-20) | 차이 |')
    print('|------|-------------------------|-------------------------|------|')

    for strategy in ['단기랭킹', '장기랭킹', '통합랭킹']:
        old_hit = existing_hit_ratios[strategy]['holdout']
        new_hit = results[strategy][20]['hit_ratio'] * 100
        diff = new_hit - old_hit
        print(f'| {strategy} | {old_hit:>7.1f}% | {new_hit:>7.1f}% | {diff:+5.1f}% |')

    print('\n=== 분석 인사이트 ===')

    # 1. 방향 예측력 분석
    print('\n1. 방향 예측력 분석:')
    baseline_hit_ratio = 50.0  # 무작위 예측의 기대값
    for strategy, topk_results in results.items():
        top20_hit = topk_results[20]['hit_ratio'] * 100
        excess_hit = top20_hit - baseline_hit_ratio
        print(f'   {strategy}: {top20_hit:.1f}% (기준대비 {excess_hit:+.1f}%)')

    # 2. 전략별 특징
    print('\n2. 전략별 특징:')
    print('   - 단기랭킹: Top-K가 커질수록 예측력 향상 (41.5% → 43.8%)')
    print('   - 장기랭킹: 안정적인 예측력, 양의 평균 수익률 (2.8% 포인트)')
    print('   - 통합랭킹: 가장 높은 방향 예측력 (44.4% at Top-10)')

    # 3. 실무적 의미
    print('\n3. 실무적 의미:')
    print('   - 방향 예측력이 37-45% 사이로, 무작위(50%)보다 낮음')
    print('   - 하지만 장기랭킹은 양의 평균 수익률을 기록')
    print('   - Top-K 선택이 전략 성과에 미치는 영향 확인')

    # PPT 업데이트 제안
    print('\n=== PPT 업데이트 제안 ===')
    print('슬라이드 10 (Track A 성과):')
    print('- IC 기반 기존 지표 유지')
    print('- 방향 적중률을 보조 지표로 추가')
    print('- "방향 예측력: Top-20 기준 39.6-44.2%" 표기')

    return results

def generate_ppt_update_content():
    """PPT 업데이트용 내용 생성"""

    content = """
=== PPT 슬라이드 10 업데이트 제안 ===

## 기존 내용 (IC 기반)
| 전략 | Hit Ratio | IC | ICIR | 과적합 위험 |
|------|-----------|----|------|------------|
| **장기랭킹 전략** | Dev 50.5% → Holdout **49.2%** | Dev -0.040 → Holdout **+0.026** | Dev -0.375 → Holdout **+0.178** | **VERY_LOW** |
| **단기랭킹 전략** | Dev **57.3%** → Holdout 43.5% | Dev -0.031 → Holdout -0.001 | Dev -0.214 → Holdout -0.006 | LOW |
| **통합랭킹 전략** | Dev 52.0% → Holdout 48.0% | Dev -0.025 → Holdout -0.010 | Dev -0.180 → Holdout -0.070 | MEDIUM |

## 추가할 방향 예측력 지표
| 전략 | 방향적중률 (Top-20) | 평균수익률 | 분석기간 |
|------|---------------------|------------|----------|
| **장기랭킹 전략** | 39.6% | +3.1% | 104일 |
| **단기랭킹 전략** | 42.7% | -0.5% | 104일 |
| **통합랭킹 전략** | 44.2% | -0.3% | 104일 |

**참고**: 방향적중률 = Top-K 종목 중 미래 수익률이 양수(+)인 비율
"""

    print(content)

if __name__ == '__main__':
    analyze_topk_hit_ratio_results()
    generate_ppt_update_content()