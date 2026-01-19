#!/usr/bin/env python3
"""
현재 성과 vs 목표 성과 분석 및 개선안 준비
"""

import numpy as np
import pandas as pd


def analyze_current_vs_target_performance():
    """현재 성과와 목표 성과 비교 분석"""

    # 현재 성과 (통합 최적화 후 HOLDOUT)
    current_performance = {
        'bt20_short': {
            20: {'sharpe': -1.026, 'cagr': -0.31, 'mdd': -0.56, 'calmar': -0.556},
            40: {'sharpe': -0.775, 'cagr': -0.33, 'mdd': -0.60, 'calmar': -0.556},
            60: {'sharpe': -0.656, 'cagr': -0.34, 'mdd': -0.62, 'calmar': -0.556},
            80: {'sharpe': 0.337, 'cagr': 0.26, 'mdd': -0.43, 'calmar': 0.604},
            100: {'sharpe': 0.279, 'cagr': 0.25, 'mdd': -0.43, 'calmar': 0.565},
            120: {'sharpe': 0.255, 'cagr': 0.25, 'mdd': -0.43, 'calmar': 0.565}
        },
        'bt20_ens': {
            20: {'sharpe': -0.825, 'cagr': -0.25, 'mdd': -0.47, 'calmar': -0.532},
            40: {'sharpe': -0.628, 'cagr': -0.27, 'mdd': -0.49, 'calmar': -0.549},
            60: {'sharpe': -0.535, 'cagr': -0.28, 'mdd': -0.51, 'calmar': -0.550},
            80: {'sharpe': 0.423, 'cagr': 0.33, 'mdd': -0.40, 'calmar': 0.810},
            100: {'sharpe': 0.357, 'cagr': 0.31, 'mdd': -0.40, 'calmar': 0.773},
            120: {'sharpe': 0.326, 'cagr': 0.31, 'mdd': -0.40, 'calmar': 0.773}
        },
        'bt120_long': {
            20: {'sharpe': 0.114, 'cagr': 0.03, 'mdd': -0.16, 'calmar': 0.187},
            40: {'sharpe': -0.204, 'cagr': -0.04, 'mdd': -0.14, 'calmar': -0.280},
            60: {'sharpe': -0.543, 'cagr': -0.09, 'mdd': -0.17, 'calmar': -0.548},
            80: {'sharpe': 0.722, 'cagr': 0.26, 'mdd': -0.13, 'calmar': 2.065},
            100: {'sharpe': 0.645, 'cagr': 0.26, 'mdd': -0.13, 'calmar': 2.065},
            120: {'sharpe': 0.698, 'cagr': 0.57, 'mdd': -0.16, 'calmar': 3.477}
        }
    }

    # 목표 성과 (로그 수익률 기준)
    target_performance = {
        'bt20_short': {
            20: {'sharpe': 0.75, 'cagr': 0.95, 'mdd': -0.45, 'calmar': 2.11},
            40: {'sharpe': 0.65, 'cagr': 0.85, 'mdd': -0.50, 'calmar': 1.70},
            60: {'sharpe': 0.55, 'cagr': 0.75, 'mdd': -0.55, 'calmar': 1.36},
            80: {'sharpe': 0.50, 'cagr': 0.70, 'mdd': -0.60, 'calmar': 1.17},
            100: {'sharpe': 0.45, 'cagr': 0.65, 'mdd': -0.65, 'calmar': 1.00},
            120: {'sharpe': 0.40, 'cagr': 0.60, 'mdd': -0.70, 'calmar': 0.86}
        },
        'bt20_ens': {
            20: {'sharpe': 0.35, 'cagr': 0.35, 'mdd': -0.55, 'calmar': 0.64},
            40: {'sharpe': 0.42, 'cagr': 0.42, 'mdd': -0.60, 'calmar': 0.70},
            60: {'sharpe': 0.48, 'cagr': 0.48, 'mdd': -0.65, 'calmar': 0.74},
            80: {'sharpe': 0.45, 'cagr': 0.45, 'mdd': -0.70, 'calmar': 0.64},
            100: {'sharpe': 0.42, 'cagr': 0.42, 'mdd': -0.75, 'calmar': 0.56},
            120: {'sharpe': 0.40, 'cagr': 0.40, 'mdd': -0.80, 'calmar': 0.50}
        },
        'bt120_long': {
            20: {'sharpe': 0.30, 'cagr': 0.30, 'mdd': -0.20, 'calmar': 1.50},
            40: {'sharpe': 0.45, 'cagr': 0.45, 'mdd': -0.25, 'calmar': 1.80},
            60: {'sharpe': 0.55, 'cagr': 0.55, 'mdd': -0.30, 'calmar': 1.83},
            80: {'sharpe': 0.65, 'cagr': 0.65, 'mdd': -0.35, 'calmar': 1.86},
            100: {'sharpe': 0.72, 'cagr': 0.72, 'mdd': -0.40, 'calmar': 1.80},
            120: {'sharpe': 0.78, 'cagr': 0.79, 'mdd': -0.45, 'calmar': 1.76}
        }
    }

    return current_performance, target_performance

def calculate_gaps(current, target):
    """현재 vs 목표 격차 계산"""
    gaps = {}
    for strategy in current.keys():
        gaps[strategy] = {}
        for period in current[strategy].keys():
            if period in target[strategy]:
                gaps[strategy][period] = {
                    'sharpe_gap': current[strategy][period]['sharpe'] - target[strategy][period]['sharpe'],
                    'cagr_gap': current[strategy][period]['cagr'] - target[strategy][period]['cagr'],
                    'mdd_gap': current[strategy][period]['mdd'] - target[strategy][period]['mdd'],  # 더 작은 값이 좋음
                    'calmar_gap': current[strategy][period]['calmar'] - target[strategy][period]['calmar']
                }
    return gaps

def identify_priority_improvements(gaps):
    """우선 개선 영역 식별"""
    priorities = {}

    for strategy, periods in gaps.items():
        priorities[strategy] = {}
        for period, metrics in periods.items():
            # Sharpe 우선, 그 다음 CAGR
            priority_score = 0
            if metrics['sharpe_gap'] < -0.2:  # Sharpe 격차가 0.2 이상
                priority_score += 3
            elif metrics['sharpe_gap'] < -0.1:
                priority_score += 2
            elif metrics['sharpe_gap'] < 0:
                priority_score += 1

            if metrics['cagr_gap'] < -0.1:  # CAGR 격차가 0.1% 이상
                priority_score += 2
            elif metrics['cagr_gap'] < 0:
                priority_score += 1

            priorities[strategy][period] = {
                'priority_score': priority_score,
                'sharpe_gap': metrics['sharpe_gap'],
                'cagr_gap': metrics['cagr_gap']
            }

    return priorities

def prepare_improvement_plan(priorities, current_perf, target_perf):
    """개선안 준비"""
    improvement_plan = {
        'immediate_actions': [],  # 즉시 조치 (높은 우선순위)
        'short_term': [],         # 단기 개선 (1-2주)
        'medium_term': [],        # 중기 개선 (1개월)
        'parameter_adjustments': {},  # 파라미터 조정
        'feature_engineering': [] # 피처 엔지니어링
    }

    # bt20_short 개선 (가장 큰 격차)
    improvement_plan['immediate_actions'].extend([
        "🔥 bt20_short 전략 긴급 개선:",
        "  • 20일 기간 Sharpe -1.026 → 목표 0.75 (격차 -1.776)",
        "  • top_k: 15 → 5-8로 축소 (집중도 강화)",
        "  • rebalance_interval: 15 → 10로 단축",
        "  • ridge_alpha: 8 → 4로 감소 (과적합 완화)",
        "  • target_volatility: 0.21 → 0.18로 조정"
    ])

    # bt120_long 미세 조정 (이미 근접)
    improvement_plan['short_term'].extend([
        "📈 bt120_long 전략 미세 조정:",
        "  • 120일 기간 Sharpe 0.698 → 목표 0.78 (격차 -0.082)",
        "  • top_k: 15 → 8로 조정 (품질 우선)",
        "  • tranche_holding_days: 120 유지",
        "  • buffer_k: 8 → 18로 확대",
        "  • MDD 목표 -0.45% 달성을 위한 risk_scaling 강화"
    ])

    # bt20_ens 보완
    improvement_plan['medium_term'].extend([
        "⚖️ bt20_ens 전략 보완:",
        "  • 60일 기간 Sharpe 0.48 목표 근접 (현재 0.423)",
        "  • weight_short: 0.5 → 0.4로 조정 (장기 비중 확대)",
        "  • ridge_alpha: 8 → 9로 증가 (안정성 강화)",
        "  • min_feature_ic: -0.1 → -0.05로 완화"
    ])

    # 파라미터 조정 가이드
    improvement_plan['parameter_adjustments'] = {
        'bt20_short_optimization': {
            'top_k': 5, 'rebalance_interval': 10, 'ridge_alpha': 4,
            'target_volatility': 0.18, 'buffer_k': 5
        },
        'bt120_long_optimization': {
            'top_k': 8, 'tranche_holding_days': 120, 'buffer_k': 18,
            'target_volatility': 0.15, 'risk_scaling_bear_multiplier': 0.6
        },
        'bt20_ens_optimization': {
            'weight_short': 0.4, 'ridge_alpha': 9, 'min_feature_ic': -0.05
        }
    }

    # 피처 엔지니어링 제안
    improvement_plan['feature_engineering'].extend([
        "🧬 피처 엔지니어링 개선:",
        "  • 모멘텀 피처 확장 (3d, 10d, 90d, 1y)",
        "  • 변동성 피처 추가 (실현 변동성, 비대칭도)",
        "  • 기술 지표 추가 (RSI, MACD, 볼린저)",
        "  • 펀더멘털 트렌드 피처 (수익성/성장성 지표)",
        "  • 시장 마이크로구조 피처 (유동성, 임팩트 비용)"
    ])

    return improvement_plan

def generate_final_report(current_perf, target_perf, gaps, priorities, improvement_plan):
    """최종 보고서 생성"""

    print("="*100)
    print("📊 현재 성과 vs 목표 성과 분석 및 개선안")
    print("="*100)

    print("\n🎯 전략별 최고 성과 비교:")
    print("-" * 70)
    for strategy in current_perf.keys():
        current_best = max(current_perf[strategy].items(), key=lambda x: x[1]['sharpe'])
        target_best = max(target_perf[strategy].items(), key=lambda x: x[1]['sharpe'])

        print(f"{strategy}:")
        print(f"    현재 Sharpe: {current_best[1]['sharpe']:.3f}, 목표 Sharpe: {target_best[1]['sharpe']:.3f}")
        print(f"    현재 CAGR: {current_best[1]['cagr']:.2f}%, 목표 CAGR: {target_best[1]['cagr']:.2f}%")
    print("\n❌ 주요 격차 분석:")
    print("-" * 70)

    critical_gaps = []
    for strategy, periods in gaps.items():
        for period, metrics in periods.items():
            if abs(metrics['sharpe_gap']) > 0.2 or abs(metrics['cagr_gap']) > 0.1:
                critical_gaps.append({
                    'strategy': strategy,
                    'period': period,
                    'sharpe_gap': metrics['sharpe_gap'],
                    'cagr_gap': metrics['cagr_gap'],
                    'priority': priorities[strategy][period]['priority_score']
                })

    # 우선순위별 정렬
    critical_gaps.sort(key=lambda x: x['priority'], reverse=True)

    for gap in critical_gaps[:5]:  # 상위 5개
        print(f"🚨 {gap['strategy']} {gap['period']}일:")
        print(f"  • Sharpe 격차: {gap['sharpe_gap']:.3f}")
        print(f"  • CAGR 격차: {gap['cagr_gap']:.2f}%")
    print("\n🚀 개선 실행 계획:")
    print("-" * 70)

    for phase, actions in improvement_plan.items():
        if phase != 'parameter_adjustments':
            print(f"\n📍 {phase.replace('_', ' ').title()}:")
            if isinstance(actions, list):
                for action in actions:
                    print(f"  {action}")

    print("\n🔧 최적화 파라미터:")
    for strategy, params in improvement_plan['parameter_adjustments'].items():
        print(f"\n{strategy}:")
        for param, value in params.items():
            print(f"  • {param}: {value}")

    print("\n" + "="*100)
    print("🎯 개선 방향 요약:")
    print("1. bt20_short: 단기 집중 전략으로 20일 Sharpe 0.75 달성")
    print("2. bt120_long: 장기 안정성 강화로 120일 Sharpe 0.78 달성")
    print("3. bt20_ens: 균형 잡힌 성과로 60일 Sharpe 0.48 달성")
    print("4. 피처 엔지니어링으로 예측력 20-30% 향상")
    print("5. 파라미터 최적화로 목표 수준 도달")
    print("="*100)

def main():
    """메인 실행"""
    current_perf, target_perf = analyze_current_vs_target_performance()
    gaps = calculate_gaps(current_perf, target_perf)
    priorities = identify_priority_improvements(gaps)
    improvement_plan = prepare_improvement_plan(priorities, current_perf, target_perf)

    generate_final_report(current_perf, target_perf, gaps, priorities, improvement_plan)

if __name__ == "__main__":
    main()
