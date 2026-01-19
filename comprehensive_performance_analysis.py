#!/usr/bin/env python3
"""
전체 18개 케이스 성과지표 종합 분석
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent


def load_final_results():
    """최종 결과 파일 로드"""

    results_dir = project_root / "results"
    pattern = "final_18_cases_backtest_report_*.csv"
    files = list(results_dir.glob(pattern))

    if not files:
        print("❌ 최종 보고서 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"📂 최종 보고서 로드: {latest_file.name}")

    df = pd.read_csv(latest_file)
    return df


def analyze_performance_metrics(df):
    """성과지표 종합 분석"""

    print("\n" + "=" * 100)
    print("📊 전체 18개 케이스 성과지표 종합 분석")
    print("=" * 100)

    # 1. Sharpe Ratio 분석
    print("\n🎯 1. Sharpe Ratio 분석 (리스크 조정 수익률)")
    print("-" * 50)

    sharpe_analysis = (
        df.groupby("strategy")["sharpe"].agg(["mean", "max", "min", "std"]).round(3)
    )
    print("전략별 Sharpe 통계:")
    print(sharpe_analysis)

    # 기간별 Sharpe
    sharpe_by_period = df.groupby("holding_days")["sharpe"].mean().round(3)
    print("\n기간별 평균 Sharpe:")
    print(sharpe_by_period)

    # Sharpe 등급 분류
    def classify_sharpe(x):
        if x >= 1.0:
            return "⭐ 매우 우수"
        elif x >= 0.5:
            return "✅ 우수"
        elif x >= 0.0:
            return "⚠️ 보통"
        elif x >= -0.5:
            return "❌ 저조"
        else:
            return "💀 매우 저조"

    df["sharpe_grade"] = df["sharpe"].apply(classify_sharpe)
    print("\nSharpe 등급 분포:")
    grade_counts = df.groupby(["strategy", "sharpe_grade"]).size().unstack(fill_value=0)
    print(grade_counts)

    # 2. CAGR 분석
    print("\n💰 2. CAGR 분석 (연복리 수익률)")
    print("-" * 50)

    cagr_analysis = (
        df.groupby("strategy")["cagr(%)"].agg(["mean", "max", "min"]).round(2)
    )
    print("전략별 CAGR 통계:")
    print(cagr_analysis)

    # CAGR 등급 분류
    def classify_cagr(x):
        if x >= 10:
            return "⭐ 매우 우수"
        elif x >= 5:
            return "✅ 우수"
        elif x >= 0:
            return "⚠️ 보통"
        elif x >= -5:
            return "❌ 저조"
        else:
            return "💀 매우 저조"

    df["cagr_grade"] = df["cagr(%)"].apply(classify_cagr)
    print("\nCAGR 등급 분포:")
    cagr_grades = df.groupby(["strategy", "cagr_grade"]).size().unstack(fill_value=0)
    print(cagr_grades)

    # 3. MDD 분석
    print("\n📉 3. MDD 분석 (최대 낙폭)")
    print("-" * 50)

    mdd_analysis = df.groupby("strategy")["mdd(%)"].agg(["mean", "max", "min"]).round(2)
    print("전략별 MDD 통계 (절대값):")
    print(mdd_analysis)

    # MDD 등급 (낮을수록 좋음)
    def classify_mdd(x):
        x = abs(x)  # 절대값으로 변환
        if x <= 5:
            return "⭐ 매우 우수"
        elif x <= 10:
            return "✅ 우수"
        elif x <= 15:
            return "⚠️ 보통"
        elif x <= 20:
            return "❌ 저조"
        else:
            return "💀 매우 저조"

    df["mdd_grade"] = df["mdd(%)"].apply(classify_mdd)
    print("\nMDD 등급 분포:")
    mdd_grades = df.groupby(["strategy", "mdd_grade"]).size().unstack(fill_value=0)
    print(mdd_grades)

    # 4. Calmar Ratio 분석
    print("\n🏆 4. Calmar Ratio 분석 (MDD 조정 Sharpe)")
    print("-" * 50)

    calmar_analysis = (
        df.groupby("strategy")["calmar"].agg(["mean", "max", "min"]).round(3)
    )
    print("전략별 Calmar 통계:")
    print(calmar_analysis)

    # 5. Hit Ratio 분석
    print("\n🎯 5. Hit Ratio 분석 (승률)")
    print("-" * 50)

    hit_analysis = (
        df.groupby("strategy")["hit_ratio(%)"].agg(["mean", "max", "min"]).round(1)
    )
    print("전략별 Hit Ratio 통계:")
    print(hit_analysis)

    # 6. Turnover 분석
    print("\n🔄 6. Turnover 분석 (포트폴리오 회전율)")
    print("-" * 50)

    turnover_analysis = (
        df.groupby("strategy")["avg_turnover"].agg(["mean", "max", "min"]).round(3)
    )
    print("전략별 Turnover 통계:")
    print(turnover_analysis)

    # Turnover 등급 (낮을수록 좋음)
    def classify_turnover(x):
        if x <= 0.2:
            return "⭐ 매우 효율적"
        elif x <= 0.4:
            return "✅ 효율적"
        elif x <= 0.6:
            return "⚠️ 보통"
        elif x <= 0.8:
            return "❌ 비효율적"
        else:
            return "💀 매우 비효율적"

    df["turnover_grade"] = df["avg_turnover"].apply(classify_turnover)
    print("\nTurnover 등급 분포:")
    turnover_grades = (
        df.groupby(["strategy", "turnover_grade"]).size().unstack(fill_value=0)
    )
    print(turnover_grades)

    # 7. Profit Factor 분석
    print("\n💹 7. Profit Factor 분석 (손익비)")
    print("-" * 50)

    pf_analysis = (
        df.groupby("strategy")["profit_factor"].agg(["mean", "max", "min"]).round(3)
    )
    print("전략별 Profit Factor 통계:")
    print(pf_analysis)

    # Profit Factor 등급 (1 이상이면 수익)
    def classify_pf(x):
        if x >= 2.0:
            return "⭐ 매우 우수"
        elif x >= 1.5:
            return "✅ 우수"
        elif x >= 1.0:
            return "⚠️ 수익"
        elif x >= 0.8:
            return "❌ 손실"
        else:
            return "💀 큰 손실"

    df["pf_grade"] = df["profit_factor"].apply(classify_pf)
    print("\nProfit Factor 등급 분포:")
    pf_grades = df.groupby(["strategy", "pf_grade"]).size().unstack(fill_value=0)
    print(pf_grades)

    return df


def create_strategy_comparison(df):
    """전략별 종합 비교"""

    print("\n" + "=" * 100)
    print("🏁 전략별 종합 성과 비교")
    print("=" * 100)

    # 전략별 평균 성과
    strategy_avg = (
        df.groupby("strategy")
        .agg(
            {
                "sharpe": "mean",
                "cagr(%)": "mean",
                "mdd(%)": "mean",
                "calmar": "mean",
                "hit_ratio(%)": "mean",
                "avg_turnover": "mean",
                "profit_factor": "mean",
            }
        )
        .round(3)
    )

    print("전략별 평균 성과:")
    print(strategy_avg)

    # 최고 성과 케이스
    print("\n🏆 최고 성과 케이스:")
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    best_cagr = df.loc[df["cagr(%)"].idxmax()]
    best_calmar = df.loc[df["calmar"].idxmax()]

    print(
        f"최고 Sharpe: {best_sharpe['strategy']} {best_sharpe['holding_days']}일 - {best_sharpe['sharpe']:.3f}"
    )
    print(
        f"최고 CAGR: {best_cagr['strategy']} {best_cagr['holding_days']}일 - {best_cagr['cagr(%)']:.1f}%"
    )
    print(
        f"최고 Calmar: {best_calmar['strategy']} {best_calmar['holding_days']}일 - {best_calmar['calmar']:.3f}"
    )

    return strategy_avg


def create_period_analysis(df):
    """기간별 분석"""

    print("\n" + "=" * 100)
    print("⏰ 기간별 성과 분석")
    print("=" * 100)

    # 기간별 평균
    period_avg = (
        df.groupby("holding_days")
        .agg(
            {
                "sharpe": ["mean", "max", "min"],
                "cagr(%)": "mean",
                "mdd(%)": "mean",
                "hit_ratio(%)": "mean",
            }
        )
        .round(3)
    )

    print("기간별 평균 성과:")
    print(period_avg)

    # 80일 전환점 분석
    short_term = df[df["holding_days"] <= 60]
    long_term = df[df["holding_days"] >= 80]

    print("\n📊 단기(≤60일) vs 장기(≥80일) 비교:")
    print(f"단기 평균 Sharpe: {short_term['sharpe'].mean():.3f}")
    print(f"장기 평균 Sharpe: {long_term['sharpe'].mean():.3f}")
    print(f"단기 평균 CAGR: {short_term['cagr(%)'].mean():.2f}%")
    print(f"장기 평균 CAGR: {long_term['cagr(%)'].mean():.2f}%")

    return period_avg


def create_practical_insights(df):
    """실무적 인사이트"""

    print("\n" + "=" * 100)
    print("💼 실무적 인사이트 및 권장사항")
    print("=" * 100)

    # 투자 가능 전략 식별
    investable = df[
        (df["sharpe"] > 0) & (df["cagr(%)"] > 0) & (df["profit_factor"] > 1)
    ]
    print(
        f"✅ 투자 가능 케이스: {len(investable)}/{len(df)} ({len(investable)/len(df)*100:.1f}%)"
    )

    if len(investable) > 0:
        print("\n투자 가능 전략:")
        for _, row in investable.iterrows():
            print(
                f"{row['strategy']} {row['holding_days']}일: Sharpe {row['sharpe']:.3f}, CAGR {row['cagr(%)']:.1f}%"
            )
    # 전략별 강점 분석
    print("\n🎯 전략별 강점 분석:")
    for strategy in df["strategy"].unique():
        strat_data = df[df["strategy"] == strategy]
        best_period = strat_data.loc[strat_data["sharpe"].idxmax(), "holding_days"]

        print(f"\n{strategy}:")
        print(f"  - 최적 기간: {best_period}일")
        print(".3f")
        print(".2f")
        print(".1f")

    # 비용 효율성 분석
    print("\n💰 비용 효율성 분석:")
    cost_eff = df[df["avg_turnover"] <= 0.4]  # 낮은 turnover
    profitable = cost_eff[cost_eff["sharpe"] > 0.3]

    if len(profitable) > 0:
        print(f"저비용 고효율 전략: {len(profitable)}개 케이스")
        for _, row in profitable.iterrows():
            print(
                f"{row['strategy']} {row['holding_days']}일: Sharpe {row['sharpe']:.3f}, Turnover {row['avg_turnover']:.3f}"
            )
    return investable


def save_analysis_report(df, investable):
    """분석 보고서 저장"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        project_root / "results" / f"comprehensive_performance_analysis_{timestamp}.csv"
    )

    # 등급 정보 추가
    df.to_csv(output_file, index=False)
    print(f"\n💾 종합 분석 보고서 저장: {output_file}")

    # 투자 가능 전략만 별도 저장
    if len(investable) > 0:
        investable_file = (
            project_root / "results" / f"investable_strategies_{timestamp}.csv"
        )
        investable.to_csv(investable_file, index=False)
        print(f"💾 투자 가능 전략 보고서 저장: {investable_file}")

    return output_file


def main():
    """메인 실행"""

    print("🚀 전체 18개 케이스 성과지표 종합 분석 시작")

    # 결과 로드
    df = load_final_results()
    if df.empty:
        return

    # 각 지표별 분석
    df = analyze_performance_metrics(df)

    # 전략별 비교
    strategy_avg = create_strategy_comparison(df)

    # 기간별 분석
    period_avg = create_period_analysis(df)

    # 실무적 인사이트
    investable = create_practical_insights(df)

    # 보고서 저장
    output_file = save_analysis_report(df, investable)

    print("\n🎉 성과지표 분석 완료!")
    print(f"📁 분석 결과: {output_file}")
    print(f"✅ 투자 가능 전략: {len(investable)}개")


if __name__ == "__main__":
    main()
