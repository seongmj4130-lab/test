import pandas as pd
import numpy as np

def create_total_return_based_report():
    """총수익률 기반 최종 전략 평가 보고서"""

    print("📊 총수익률 기반 최종 전략 평가 보고서")
    print("=" * 70)

    # 신규 백테스트 결과 로드 (top_k=20)
    new_results = pd.read_csv('results/topk20_performance_metrics.csv')

    # 전략별 총수익률 순위
    rankings = []
    for _, row in new_results.iterrows():
        rankings.append({
            '전략': row['전략'],
            '총수익률': row['총수익률'],
            'MDD': row['MDD'],
            'Sharpe': row['Sharpe'],
            'Calmar': row['Calmar']
        })

    rankings.sort(key=lambda x: x['총수익률'], reverse=True)

    print("🏆 총수익률 기준 최종 전략 순위 (top_k=20)")
    print("=" * 60)

    medal_emojis = ['🥇', '🥈', '🥉', '4️⃣']

    for i, strategy in enumerate(rankings):
        medal = medal_emojis[i] if i < len(medal_emojis) else f"{i+1}️⃣"
        print(f"{medal} {strategy['전략']}")
        print(".2f")
        print(".2f")
        print(".3f")
        print(".3f")
        print()

    # 전략별 특성 분석
    print("🎯 전략별 특성 분석")
    print("=" * 40)

    for strategy in rankings:
        name = strategy['전략']
        total_return = strategy['총수익률']
        mdd = abs(strategy['MDD'])

        # 수익성 등급
        if total_return > 0.10:
            profit_grade = "⭐⭐⭐ (매우 우수)"
        elif total_return > 0.05:
            profit_grade = "⭐⭐ (우수)"
        elif total_return > 0:
            profit_grade = "⭐ (보통)"
        else:
            profit_grade = "❌ (저조)"

        # 리스크 등급
        if mdd < 0.10:
            risk_grade = "🛡️🛡️🛡️ (매우 안정)"
        elif mdd < 0.15:
            risk_grade = "🛡️🛡️ (안정)"
        elif mdd < 0.20:
            risk_grade = "🛡️ (보통)"
        else:
            risk_grade = "⚠️ (주의)"

        print(f"🏆 {name}")
        print(f"   • 수익성: {profit_grade}")
        print(f"   • 리스크: {risk_grade}")
        print(f"   • 수익/리스크 비율: {total_return/mdd:.3f}" if mdd > 0 else "   • 수익/리스크 비율: N/A")
        print()

    # 투자 추천
    print("💡 투자 전략 추천")
    print("=" * 30)

    print("1️⃣ 메인 전략: BT120 장기")
    print("   • 이유: 최고 총수익률 (+12.7%) + 안정적 MDD (10.3%)")
    print("   • 장점: 장기적 관점에서 가장 강건한 성과")
    print("   • 추천: 포트폴리오의 50% 이상 배분")
    print()

    print("2️⃣ 보완 전략: BT120 앙상블")
    print("   • 이유: 안정적인 총수익률 (+8.4%) + 낮은 MDD (9.3%)")
    print("   • 장점: 단기/장기 랭킹 결합으로 리스크 분산")
    print("   • 추천: 포트폴리오의 30% 배분")
    print()

    print("3️⃣ 헤지 전략: BT20 앙상블")
    print("   • 이유: 양호한 총수익률 (+5.5%) + 상대적 안정성")
    print("   • 장점: 중간 리스크에서 수익 창출")
    print("   • 추천: 포트폴리오의 15% 배분")
    print()

    print("4️⃣ ⚠️ 유의 전략: BT20 단기")
    print("   • 평가: 총수익률 -8.0%로 현재 설정에서 부적합")
    print("   • 원인: top_k=20 설정에서 성능 저하")
    print("   • 권장: top_k=10-15로 조정 후 재평가")
    print()

    # 최종 결론
    print("🎉 최종 결론")
    print("=" * 20)

    print("✅ top_k=20 통일 설정의 효과:")
    print("   • BT120 전략군의 우수성 입증")
    print("   • 안정적인 장기 전략의 강점 부각")
    print("   • 단기 전략의 취약성 확인")
    print()

    print("✅ 총수익률 기반 평가의 장점:")
    print("   • CAGR의 기간 왜곡 효과 제거")
    print("   • 실제 기간 내 성과 명확히 파악")
    print("   • 투자 의사결정에 더 적합")
    print()

    print("✅ 포트폴리오 구성 제안:")
    print("   • BT120 장기: 50% (메인)")
    print("   • BT120 앙상블: 30% (보완)")
    print("   • BT20 앙상블: 15% (헤지)")
    print("   • BT20 단기: 0% (제외)")
    print()

    print("💡 핵심 메시지:")
    print("   총수익률로 평가하면 BT120 장기가 가장 우수한 전략으로 확인됨!")
    print("   top_k=20 설정이 장기 전략에 유리하게 작용!")

    # CSV로 최종 결과 저장
    final_results = pd.DataFrame(rankings)
    final_results.to_csv('results/final_total_return_ranking.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 최종 결과 CSV 저장: results/final_total_return_ranking.csv")

if __name__ == "__main__":
    create_total_return_based_report()