import numpy as np
import pandas as pd


def explain_cagr_calculation():
    """CAGR 계산 방식 설명"""

    print("📊 CAGR vs 총수익률 차이 설명")
    print("=" * 60)

    # 실제 데이터로 예시
    print("🔢 CAGR 계산 수식:")
    print("CAGR = (1 + 총수익률)^(252/n) - 1")
    print("  - 252: 연간 거래일 수")
    print("  - n: 실제 데이터 일 수")
    print()

    print("📈 계산 예시:")
    print("총수익률 r = 10% (0.10)")
    print("데이터 일 수 n = 23일")
    print()
    print("CAGR = (1 + 0.10)^(252/23) - 1")
    print("     = (1.10)^(11.0) - 1")
    print(".3f")
    print()

    # 실제 신규 결과 분석
    new_results = pd.read_csv('results/topk20_performance_metrics.csv')

    print("📋 신규 결과 분석 (top_k=20, 23일 데이터):")
    print("-" * 50)

    for _, row in new_results.iterrows():
        total_return = row['총수익률']
        cagr = row['CAGR']
        data_points = int(row['데이터포인트'])

        print(f"🏆 {row['전략']}")
        print(".2%")
        print(".2%")
        print(f"   • 데이터 기간: {data_points}일")
        print(".1f")
        print()

    print("⚠️  문제점:")
    print("-" * 20)
    print("1. 짧은 기간 데이터를 연간화하니 복리 효과 과대")
    print("2. 23일 → 252일 연간화: 11배 기간 확대")
    print("3. CAGR 수치가 비현실적으로 높아짐")
    print()

    print("💡 올바른 해석:")
    print("-" * 20)
    print("• 총수익률: 실제 기간 내 성과")
    print("• CAGR: 연간화된 기대 성과 (참고용)")
    print("• 짧은 기간: CAGR 신뢰성 낮음")
    print("• 긴 기간: CAGR 의미 있음")
    print()

    # 기간별 CAGR 비교
    print("📊 기간별 CAGR 비교:")
    print("-" * 30)

    periods = [23, 63, 126, 252]  # 3개월, 1년, 2년, 4년

    for period in periods:
        if period <= 252:
            annualization_factor = 252 / period
            example_return = 0.10  # 10%
            cagr = (1 + example_return) ** annualization_factor - 1

            print("<8")

    print()
    print("🎯 결론:")
    print("23일 데이터의 CAGR는 11배 기간 연장 효과로 비현실적!")
    print("총수익률로 평가하는 것이 더 정확함")

if __name__ == "__main__":
    explain_cagr_calculation()
