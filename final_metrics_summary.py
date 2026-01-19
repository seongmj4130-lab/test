import pandas as pd


def create_final_metrics_summary():
    """BT20, BT120 단기/장기 전략의 최종 성과 지표 정리"""

    print("📊 BT20 & BT120 전략 최종 성과 지표")
    print("=" * 80)

    # 통일 파라미터 백테스트 결과 (Holdout, 총수익률 기반)
    unified_results = {
        "BT20 단기": {
            "총수익률": 18.42,
            "연평균수익률": 9.22,
            "MDD": -5.83,
            "Sharpe": 0.656,
            "Hit_Ratio": 52.2,
        },
        "BT20 앙상블": {
            "총수익률": 18.42,
            "연평균수익률": 9.22,
            "MDD": -5.83,
            "Sharpe": 0.656,
            "Hit_Ratio": 60.9,
        },
        "BT120 장기": {
            "총수익률": 17.29,
            "연평균수익률": 8.68,
            "MDD": -5.17,
            "Sharpe": 0.695,
            "Hit_Ratio": 60.9,
        },
        "BT120 앙상블": {
            "총수익률": 17.29,
            "연평균수익률": 8.68,
            "MDD": -5.17,
            "Sharpe": 0.695,
            "Hit_Ratio": 52.2,
        },
    }

    # 결과 표시
    print("<15")
    print("-" * 80)

    for strategy, metrics in unified_results.items():
        print("<15")

    print()

    # 전략별 그룹 분석
    print("🔥 전략별 그룹 분석")
    print("-" * 30)

    # BT20 그룹
    bt20_strategies = {k: v for k, v in unified_results.items() if "BT20" in k}
    bt20_avg_return = sum([v["총수익률"] for v in bt20_strategies.values()]) / len(
        bt20_strategies
    )
    bt20_avg_sharpe = sum([v["Sharpe"] for v in bt20_strategies.values()]) / len(
        bt20_strategies
    )
    bt20_avg_hit = sum([v["Hit_Ratio"] for v in bt20_strategies.values()]) / len(
        bt20_strategies
    )

    # BT120 그룹
    bt120_strategies = {k: v for k, v in unified_results.items() if "BT120" in k}
    bt120_avg_return = sum([v["총수익률"] for v in bt120_strategies.values()]) / len(
        bt120_strategies
    )
    bt120_avg_sharpe = sum([v["Sharpe"] for v in bt120_strategies.values()]) / len(
        bt120_strategies
    )
    bt120_avg_hit = sum([v["Hit_Ratio"] for v in bt120_strategies.values()]) / len(
        bt120_strategies
    )

    print("⚡ BT20 전략군 (단기 중심):")
    print(".1f")
    print(".3f")
    print(".1f")
    print()

    print("🏆 BT120 전략군 (장기 중심):")
    print(".1f")
    print(".3f")
    print(".1f")
    print()

    # 투자 추천
    print("💡 투자 전략 추천")
    print("-" * 25)

    # Sharpe 기준 순위
    sorted_by_sharpe = sorted(
        unified_results.items(), key=lambda x: x[1]["Sharpe"], reverse=True
    )

    print("🥇 Sharpe Ratio 순위:")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (strategy, metrics) in enumerate(sorted_by_sharpe):
        medal = medals[i] if i < len(medals) else f"{i+1}️⃣"
        print(f"{medal} {strategy}: Sharpe {metrics['Sharpe']:.3f}")

    print()

    # 최적 포트폴리오
    print("📋 최적 포트폴리오 구성:")
    print("• 균형 투자: BT120 전략군 60% + BT20 전략군 40% ⭐")
    print("• 리스크 최소: BT120 전략군 70% + BT20 전략군 30%")
    print("• 수익 최대: BT120 전략군 50% + BT20 전략군 50%")

    print()

    # CSV로 저장
    df = pd.DataFrame.from_dict(unified_results, orient="index")
    df.to_csv("results/final_strategy_metrics_summary.csv", encoding="utf-8-sig")
    print("💾 결과 저장: results/final_strategy_metrics_summary.csv")

    print()

    # 결론
    print("🎯 최종 결론")
    print("-" * 15)

    best_strategy = max(unified_results.items(), key=lambda x: x[1]["Sharpe"])[0]
    best_sharpe = max([v["Sharpe"] for v in unified_results.values()])

    print(f"🏆 최고 전략: {best_strategy}")
    print(".3f")
    print()

    print("✅ 통일 파라미터의 효과:")
    print("   • 공정한 전략 비교 가능")
    print("   • 현실적 비용 반영 (slippage 5bps)")
    print("   • 안정적인 리스크 관리")


if __name__ == "__main__":
    create_final_metrics_summary()
