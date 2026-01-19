import numpy as np
import pandas as pd


def analyze_period_difference():
    """기존 vs 신규 결과의 기간 차이 분석"""

    print("🔍 기존 vs 신규 결과 기간 차이 분석")
    print("=" * 60)

    # 기존 결과 (더 긴 기간 사용 추정)
    print("📊 기존 결과 (개별 top_k):")
    print("• 기간: Dev + Holdout 전체 (약 5-7년)")
    print("• CAGR 계산: 실제 연간 성과 반영")
    print("• Sharpe/CAGR: 정상 범위 (0.5-0.7)")
    print()

    # 신규 결과 (짧은 기간)
    print("📊 신규 결과 (top_k=20):")
    print("• 기간: Holdout만 (23개월)")
    print("• CAGR 계산: 23일 → 252일 연간화")
    print("• Sharpe/CAGR: 비정상 범위 (-0.8 ~ 2.5)")
    print()

    # 기간별 CAGR 왜곡 효과 계산
    print("📈 기간별 CAGR 왜곡 효과:")
    print("-" * 40)

    total_return = 0.08  # 8% 총수익률 예시
    periods = [23, 63, 126, 252, 504, 1008]  # 일 단위

    print("<8")
    print("-" * 50)

    for period in periods:
        annualization_factor = 252 / period
        cagr = (1 + total_return) ** annualization_factor - 1
        period_years = period / 252

        print("<8")

    print()
    print("💡 기간이 짧을수록 CAGR가 과도하게 부풀려짐!")
    print()

    # 실제 사례 비교
    print("📋 실제 사례 비교:")
    print("-" * 30)

    examples = [
        {"name": "신규 BT120 장기", "total_return": 0.127, "period_days": 23},
        {"name": "가정 정상 케이스", "total_return": 0.127, "period_days": 252},
        {"name": "가정 장기 케이스", "total_return": 0.127, "period_days": 1008}
    ]

    for ex in examples:
        annualization_factor = 252 / ex["period_days"]
        cagr = (1 + ex["total_return"]) ** annualization_factor - 1
        period_years = ex["period_days"] / 252

        print("<18")

    print()
    print("🎯 핵심 인사이트:")
    print("- 23일 데이터로는 CAGR가 비현실적")
    print("- 1년 데이터로는 CAGR가 적정")
    print("- 총수익률이 더 신뢰할 수 있음")
    print()

    # 평가 방식 권장
    print("💡 평가 방식 권장:")
    print("-" * 25)
    print("1. 짧은 기간 (1년 미만): 총수익률 우선")
    print("2. 중간 기간 (1-3년): CAGR + 총수익률")
    print("3. 긴 기간 (3년 이상): CAGR 우선")
    print("4. 백테스트: Holdout 기간 성과로 검증")
    print()

if __name__ == "__main__":
    analyze_period_difference()
