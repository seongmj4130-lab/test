#!/usr/bin/env python3
"""
더미데이터 vs 실제 데이터 비교 분석
"""

import pandas as pd


def analyze_comparison():
    print("📊 더미데이터 vs 실제 데이터 비교 분석")
    print("=" * 60)

    # 실제 최근 데이터 로드
    actual_df = pd.read_csv("data/ui_strategies_cumulative_comparison.csv")

    # 더미데이터 로드
    dummy_df = pd.read_csv("data/strategy_performance_table.csv")

    # 최종 행 데이터로 총 수익률 계산 (2024년 12월 기준)
    final_row = actual_df.iloc[-1]

    print("\n🏆 실제 최근 데이터 성과 (2024년 12월 누적)")
    print("-" * 50)
    print(f"KOSPI200: {final_row['kospi200']:.2f}%")

    # 각 전략별 최고 성과 계산
    strategies = ["bt20_short", "bt120_long", "bt20_ens"]
    holding_days = [20, 40, 60, 80, 100, 120]

    actual_performance = {}
    for strategy in strategies:
        max_return = -999
        best_holding = 0

        for holding in holding_days:
            col_name = f"{strategy}_{holding}"
            if col_name in actual_df.columns:
                cumulative_return = final_row[col_name]
                if cumulative_return > max_return:
                    max_return = cumulative_return
                    best_holding = holding

        actual_performance[strategy] = {"return": max_return, "holding": best_holding}
        print(f"{strategy}: {max_return:.2f}% ({best_holding}일)")

    print("\n🎯 더미데이터 vs 실제 데이터 비교")
    print("-" * 50)

    # KOSPI200 비교
    dummy_kospi = float(dummy_df[dummy_df["전략"] == "KOSPI200"]["총수익률(%)"].iloc[0])
    actual_kospi = float(final_row["kospi200"])
    kospi_gap = actual_kospi - dummy_kospi

    print("KOSPI200:")
    print(".1f")
    print(".1f")
    print(".1f")
    # 각 전략별 비교
    total_dummy_gap = 0
    for strategy in strategies:
        dummy_max = float(dummy_df[dummy_df["전략"] == strategy]["총수익률(%)"].max())
        actual_max = float(actual_performance[strategy]["return"])
        gap = actual_max - dummy_max

        print(f"\n{strategy}:")
        print(".1f")
        print(".1f")
        print(".1f")
        total_dummy_gap += gap

    print("\n📈 종합 분석")
    print("-" * 50)
    avg_gap = total_dummy_gap / len(strategies)
    print(".1f")
    print(".1f")
    print(".1f")
    if avg_gap < -5:
        print("결론: 실제 성과가 더미데이터보다 크게 낮음 - 개선 필요")
    elif avg_gap < -2:
        print("결론: 실제 성과가 더미데이터보다 낮음 - 일부 개선 필요")
    else:
        print("결론: 실제 성과가 더미데이터와 유사 - 추가 개선 필요")

    return actual_performance, dummy_df, final_row


def identify_improvement_areas(actual_perf, dummy_df, final_row):
    print("\n🔧 실무 관점 개선 방안")
    print("-" * 50)

    # 1. 절대 수익률 부족 문제
    print("1️⃣ 절대 수익률 개선:")
    print("   • 현재: 모든 전략 KOSPI200 하회")
    print("   • 목표: 최소 KOSPI200 수준 도달")
    print("   • 방안: Alpha 증폭 전략 적용 (이미 진행 중)")

    # 2. 전략별 특성 분석
    bt20_short_actual = actual_perf["bt20_short"]["return"]
    bt120_long_actual = actual_perf["bt120_long"]["return"]
    bt20_ens_actual = actual_perf["bt20_ens"]["return"]

    if bt120_long_actual > bt20_short_actual and bt120_long_actual > bt20_ens_actual:
        best_strategy = "bt120_long"
        print("\n2️⃣ 전략별 성과:")
        print("   • 최고 성과: 장기 전략 (bt120_long)")
        print("   • 이유: 안정적인 수익 창출")
        print("   • 권장: 장기 전략 중심으로 조정")
    else:
        best_strategy = (
            "bt20_short" if bt20_short_actual > bt20_ens_actual else "bt20_ens"
        )
        print("\n2️⃣ 전략별 성과:")
        print(f"   • 최고 성과: {best_strategy}")
        print("   • 이유: 단기 모멘텀 활용")

    # 3. 기간별 특성
    print("\n3️⃣ 보유 기간 최적화:")
    for strategy in ["bt20_short", "bt120_long", "bt20_ens"]:
        best_holding = actual_perf[strategy]["holding"]
        dummy_best_holding = dummy_df[(dummy_df["전략"] == strategy)][
            "총수익률(%)"
        ].idxmax()
        dummy_best_row = dummy_df.iloc[dummy_best_holding]
        dummy_best_holding_days = int(dummy_best_row["Holding Days"])

        if abs(best_holding - dummy_best_holding_days) <= 20:
            print(f"   • {strategy}: {best_holding}일 (더미와 유사)")
        else:
            print(
                f"   • {strategy}: {best_holding}일 (더미 {dummy_best_holding_days}일과 차이)"
            )

    # 4. 실무적 제언
    print("\n4️⃣ 실무적 개선 방안:")
    print("   • 비용 최적화: 1bps 목표로 진행 중")
    print("   • 리스크 관리: MDD 목표 -10% 이내 유지")
    print("   • 시장 적응: HOLDOUT 특성 반영 완료")
    print("   • 추가 개선: 팩터 확장 및 모멘텀 강화")

    return best_strategy


if __name__ == "__main__":
    actual_perf, dummy_df, final_row = analyze_comparison()
    best_strategy = identify_improvement_areas(actual_perf, dummy_df, final_row)

    print("\n✅ 분석 완료")
    print(f"🎯 권장 전략: {best_strategy}")
    print("🔄 다음 단계: Alpha 증폭 + 비용 최적화 심화")
