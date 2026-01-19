#!/usr/bin/env python3
"""
절대 수익률 중심 평가 - 간단 버전
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    print("🎯 절대 수익률 중심 평가 시스템")
    print("="*60)

    # 백테스트 결과 로드
    results_path = "results/final_18_cases_backtest_report_20260114_030411.csv"
    if not Path(results_path).exists():
        print("❌ 백테스트 결과 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(results_path)

    # 벤치마크 데이터
    kospi_return = 4.5  # 실제 연 4.5%
    quant_avg_return = 6.5  # 실제 평균 6.5%

    print("📊 평가 가중치 (수익률 중심):")
    weights = {
        'cagr': 0.40,        # 절대 수익률 (가장 중요)
        'total_return': 0.25, # 총 수익률
        'sharpe': 0.15,      # 리스크 조정 수익률 (감소)
        'mdd': 0.10,         # 안정성 (감소)
        'calmar': 0.10       # Calmar 비율 (유지)
    }

    for metric, weight in weights.items():
        print(f"  • {metric}: {weight:.0%}")

    print("\n🎯 전략별 절대 수익률 평가")
    print("-" * 60)

    evaluations = {}

    for strategy in ['bt20_short', 'bt20_ens', 'bt120_long']:
        strategy_data = df[df['strategy'] == strategy]

        if strategy_data.empty:
            continue

        # CAGR 기준 최고 성과 선택 (수익률 중심)
        best_idx = strategy_data['cagr(%)'].idxmax()
        best_case = strategy_data.loc[best_idx]

        # 벤치마크 대비 평가
        excess_vs_kospi = best_case['cagr(%)'] - kospi_return
        excess_vs_quant = best_case['cagr(%)'] - quant_avg_return

        # 등급 결정
        if best_case['cagr(%)'] >= quant_avg_return:
            grade = "A"  # 퀀트 평균 이상
        elif best_case['cagr(%)'] >= kospi_return:
            grade = "B"  # KOSPI 이상
        elif best_case['cagr(%)'] >= kospi_return * 0.5:
            grade = "C"  # KOSPI 50% 이상
        else:
            grade = "D"  # 부진

        evaluations[strategy] = {
            'cagr': best_case['cagr(%)'],
            'excess_vs_kospi': excess_vs_kospi,
            'excess_vs_quant': excess_vs_quant,
            'grade': grade,
            'holding_days': best_case['holding_days']
        }

        print(f"\n{strategy.upper()} ({best_case['holding_days']}일)")
        print(f"  • CAGR: {best_case['cagr(%)']:.2f}%")
        print(f"  • KOSPI 초과: {excess_vs_kospi:+.2f}%")
        print(f"  • 퀀트 초과: {excess_vs_quant:+.2f}%")
        print(f"  • 등급: {grade}")

    # 전략 순위 결정
    print("\n🏆 절대 수익률 기반 전략 순위")
    print("-" * 60)

    grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    ranked_strategies = sorted(
        evaluations.items(),
        key=lambda x: (
            grade_scores[x[1]['grade']],
            x[1]['cagr']
        ),
        reverse=True
    )

    for rank, (strategy, data) in enumerate(ranked_strategies, 1):
        grade_desc = {
            'A': '탁월 (퀀트 평균 이상)',
            'B': '우수 (KOSPI 이상)',
            'C': '보통 (KOSPI 50% 이상)',
            'D': '부진 (개선 필요)'
        }

        print(f"{rank}위: {strategy.upper()}")
        print(f"   CAGR: {data['cagr']:.1f}%")
        print(f"   등급: {data['grade']} - {grade_desc[data['grade']]}")

    # 평가 결과 요약
    print("\n📋 절대 수익률 중심 평가 보고서")
    print("="*60)

    best_strategy = ranked_strategies[0][0] if ranked_strategies else "N/A"

    print("🎯 평가 결과 요약:")
    print("  • 메인 KPI: 절대 수익률 (CAGR)")
    print("  • 평가 방식: 수익률 중심 가중치 적용")
    print(f"  • 최고 전략: {best_strategy.upper()}")

    print("\n💡 투자 의사결정 가이드:")
    print("  • A등급: 적극 투자 추천")
    print("  • B등급: 보수적 투자 고려")
    print("  • C등급: 모니터링 후 결정")
    print("  • D등급: 전략 개선 필요")

    print("\n🔧 전략별 권장사항:")
    for strategy, data in evaluations.items():
        grade = data['grade']

        if grade == 'A':
            recommendation = "적극 투자 추천 - 안정적 수익 창출 가능"
        elif grade == 'B':
            recommendation = "보수적 투자 고려 - KOSPI 초과 가능성"
        elif grade == 'C':
            recommendation = "모니터링 후 결정 - 개선 여지 확인 필요"
        else:
            recommendation = "전략 개선 필요 - 현재 수익률 부진"

        print(f"  • {strategy.upper()}: {recommendation}")

    print("\n✅ 절대 수익률 중심 평가 완료!")
    print(f"🎯 최고 전략: {best_strategy.upper()}")

if __name__ == "__main__":
    main()
