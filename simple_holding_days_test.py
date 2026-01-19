import pandas as pd
import numpy as np
from pathlib import Path

def simple_holding_days_test():
    """간단한 방식으로 holding_days 영향 분석"""

    print("🔬 Holding Days 영향 분석 (간단 버전)")
    print("=" * 50)

    # 기존 백테스트 결과 활용
    base_results = {
        'bt20_ens': {'holding_days': 20, 'sharpe': 0.656, 'cagr': 0.092, 'mdd': -0.058},
        'bt120_ens': {'holding_days': 20, 'sharpe': 0.695, 'cagr': 0.087, 'mdd': -0.052}
    }

    # holding_days 변화에 따른 예상 성과 (이론적 추정)
    # 실제로는 turnover, transaction cost 등이 영향을 미침

    print("📊 현재 기준 성과 (holding_days=20)")
    print("-" * 40)
    for strategy, metrics in base_results.items():
        strategy_name = 'BT20 앙상블' if 'bt20' in strategy else 'BT120 앙상블'
        print(f"{strategy_name}: Sharpe {metrics['sharpe']:.3f}, CAGR {metrics['cagr']:.1%}, MDD {metrics['mdd']:.1%}")

    print("\n🎯 Holding Days 변화 영향 분석")
    print("-" * 50)

    # holding_days가 길어질수록:
    # - Turnover 감소 → 거래비용 감소 → 성과 향상
    # - Market timing 기회 감소 → 변동성 증가 가능
    # - 실제 효과는 데이터와 전략에 따라 다름

    holding_days_options = [40, 60, 80, 100]
    impact_analysis = {}

    for hd in holding_days_options:
        print(f"\nholding_days = {hd} 분석:")
        print("-" * 30)

        # Turnover 영향 (holding_days 증가 → turnover 감소)
        turnover_reduction = (20 / hd)  # 20일 기준 대비
        print(".1f")

        # 예상 성과 변화 (단순 추정)
        # 실제로는 더 복잡한 요인들이 작용
        sharpe_change = min(0.05, (hd - 20) * 0.001)  # 보수적 추정
        cagr_change = min(0.01, (hd - 20) * 0.0003)   # 보수적 추정

        print(".3f")
        print(".2%")

        # 전략별 영향
        for strategy, base_metrics in base_results.items():
            strategy_name = 'BT20 앙상블' if 'bt20' in strategy else 'BT120 앙상블'
            new_sharpe = base_metrics['sharpe'] + sharpe_change
            new_cagr = base_metrics['cagr'] + cagr_change

            print(f"  • {strategy_name}:")
            print(".3f")
            print(".1%")

            impact_analysis[f"{strategy}_{hd}"] = {
                'strategy': strategy_name,
                'holding_days': hd,
                'base_sharpe': base_metrics['sharpe'],
                'new_sharpe': new_sharpe,
                'sharpe_change': sharpe_change,
                'base_cagr': base_metrics['cagr'],
                'new_cagr': new_cagr,
                'cagr_change': cagr_change
            }

    print("\n📋 종합 비교표")
    print("-" * 80)

    # DataFrame으로 정리
    analysis_df = pd.DataFrame.from_dict(impact_analysis, orient='index')
    analysis_df = analysis_df[['strategy', 'holding_days', 'base_sharpe', 'new_sharpe', 'sharpe_change', 'base_cagr', 'new_cagr', 'cagr_change']]

    print("<15")
    print("-" * 80)

    for _, row in analysis_df.iterrows():
        print("<15")

    # CSV 저장
    csv_file = 'results/holding_days_impact_analysis.csv'
    analysis_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 분석 결과 저장: {csv_file}")

    print("\n🎯 결론 및 권장사항")
    print("-" * 40)

    print("1️⃣ Turnover 영향:")
    print("   • holding_days 증가 → turnover 감소 → 거래비용 절감")
    print("   • 40일: 50% 감소, 100일: 80% 감소")

    print("\n2️⃣ 성과 영향:")
    print("   • Sharpe Ratio 약간 개선 (+0.02~0.05)")
    print("   • CAGR 소폭 향상 (+0.3~1.0%p)")
    print("   • 실제 효과는 전략과 시장 상황에 따라 다름")

    print("\n3️⃣ 전략별 차이:")
    print("   • BT20 (단기): holding_days 연장 효과 상대적으로 작음")
    print("   • BT120 (장기): holding_days 연장 효과 더 유의미할 수 있음")

    print("\n4️⃣ 권장사항:")
    print("   • 60-80일 범위에서 최적점 탐색 추천")
    print("   • 실제 백테스트로 정확한 효과 검증 필요")
    print("   • 리스크-리턴 트레이드오프 고려")

if __name__ == "__main__":
    simple_holding_days_test()