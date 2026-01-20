#!/usr/bin/env python3
"""
업계표준 비용 적용 후 최종 성과 분석
"""

from pathlib import Path

import pandas as pd


def analyze_final_performance():
    """업계표준 비용 적용 후 최종 성과 분석"""

    print("📊 업계표준 비용 적용 후 최종 성과 분석")
    print("=" * 60)

    # 최신 결과 파일 로드
    results_dir = Path("results")
    csv_files = list(results_dir.glob("dynamic_period_backtest_clean_*.csv"))
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)

    df = pd.read_csv(latest_file)
    print(f"📊 분석 파일: {latest_file.name}")
    print()

    # 전략별 평균 성과 계산
    strategy_summary = (
        df.groupby("strategy")
        .agg(
            {
                "sharpe": "mean",
                "CAGR (%)": "mean",
                "Total Return (%)": "mean",
                "MDD (%)": "mean",
                "calmar": "mean",
                "Hit Ratio (%)": "mean",
                "avg_turnover": "mean",
                "profit_factor": "mean",
            }
        )
        .round(3)
    )

    print("📊 전략별 평균 성과 (업계표준 비용 적용):")
    print(strategy_summary)
    print()

    # 실무 평가 기준
    evaluation_criteria = {
        "cagr": {"excellent": 0.15, "good": 0.10, "acceptable": 0.05, "poor": 0.0},
        "sharpe": {"excellent": 1.0, "good": 0.5, "acceptable": 0.2, "poor": 0.0},
        "mdd": {"excellent": -5, "good": -10, "acceptable": -15, "poor": -20},
        "profit_factor": {
            "excellent": 1.5,
            "good": 1.3,
            "acceptable": 1.1,
            "poor": 1.0,
        },
    }

    # 전략별 평가
    print("🎯 실무 평가 결과:")
    print("=" * 40)

    for strategy in df["strategy"].unique():
        strategy_data = df[df["strategy"] == strategy]
        avg_performance = strategy_data[
            ["CAGR (%)", "sharpe", "MDD (%)", "profit_factor"]
        ].mean()

        print(f"\n{strategy} 전략 평가:")

        # CAGR 평가
        cagr = avg_performance["CAGR (%)"]
        if cagr >= evaluation_criteria["cagr"]["excellent"]:
            cagr_grade = "⭐ 우수 (15%+)"
        elif cagr >= evaluation_criteria["cagr"]["good"]:
            cagr_grade = "✅ 양호 (10%+)"
        elif cagr >= evaluation_criteria["cagr"]["acceptable"]:
            cagr_grade = "⚠️ 보통 (5%+)"
        else:
            cagr_grade = "❌ 미흡 (0% 미만)"

        # Sharpe 평가
        sharpe = avg_performance["sharpe"]
        if sharpe >= evaluation_criteria["sharpe"]["excellent"]:
            sharpe_grade = "⭐ 우수 (1.0+)"
        elif sharpe >= evaluation_criteria["sharpe"]["good"]:
            sharpe_grade = "✅ 양호 (0.5+)"
        elif sharpe >= evaluation_criteria["sharpe"]["acceptable"]:
            sharpe_grade = "⚠️ 보통 (0.2+)"
        else:
            sharpe_grade = "❌ 미흡 (0.0 미만)"

        # MDD 평가
        mdd = avg_performance["MDD (%)"]
        if abs(mdd) <= abs(evaluation_criteria["mdd"]["excellent"]):
            mdd_grade = "⭐ 우수 (5% 미만)"
        elif abs(mdd) <= abs(evaluation_criteria["mdd"]["good"]):
            mdd_grade = "✅ 양호 (10% 미만)"
        elif abs(mdd) <= abs(evaluation_criteria["mdd"]["acceptable"]):
            mdd_grade = "⚠️ 보통 (15% 미만)"
        else:
            mdd_grade = "❌ 미흡 (20% 초과)"

        # Profit Factor 평가
        pf = avg_performance["profit_factor"]
        if pf >= evaluation_criteria["profit_factor"]["excellent"]:
            pf_grade = "⭐ 우수 (1.5+)"
        elif pf >= evaluation_criteria["profit_factor"]["good"]:
            pf_grade = "✅ 양호 (1.3+)"
        elif pf >= evaluation_criteria["profit_factor"]["acceptable"]:
            pf_grade = "⚠️ 보통 (1.1+)"
        else:
            pf_grade = "❌ 미흡 (1.0 미만)"

        print(f"  CAGR: {cagr:.2f}% - {cagr_grade}")
        print(f"  Sharpe: {sharpe:.2f} - {sharpe_grade}")
        print(f"  MDD: {mdd:.2f}% - {mdd_grade}")
        print(f"  Profit Factor: {pf:.2f} - {pf_grade}")

    # 종합 평가
    print("\n🏆 종합 평가:")
    print("=" * 30)

    overall_cagr = df["CAGR (%)"].mean()
    overall_sharpe = df["sharpe"].mean()
    overall_mdd = df["MDD (%)"].mean()

    print(".2f")
    print(".2f")
    print(".2f")
    # 투자 매력도 평가
    if overall_cagr >= 0.05 and overall_sharpe >= 0.2 and abs(overall_mdd) <= 15:
        attractiveness = "🟢 투자 매력 높음 (실전 적용 가능)"
    elif overall_cagr >= 0.02 and overall_sharpe >= 0.0 and abs(overall_mdd) <= 20:
        attractiveness = "🟡 투자 매력 보통 (추가 개선 필요)"
    else:
        attractiveness = "🔴 투자 매력 낮음 (전면 재검토 필요)"

    print(f"투자 매력도: {attractiveness}")

    print("\n💡 주요 문제점:")
    print("- 수익률이 업계 평균에 크게 미달")
    print("- Sharpe 비율이 대부분 음수")
    print("- 모델 예측력이 부족한 것으로 보임")

    print("\n📋 권장사항:")
    print("1. 모델 예측력 강화 (피처 엔지니어링, 앙상블)")
    print("2. 전략 로직 재검토 (단기/장기 특성 반영)")
    print("3. 데이터 품질 검증 (L6 레이블링 정확도)")
    print("4. 백테스트 방법론 검증 (과적합 여부)")


if __name__ == "__main__":
    analyze_final_performance()
