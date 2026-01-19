#!/usr/bin/env python3
"""
동적 기간 백테스트 결과 분석 스크립트
"""

from pathlib import Path

import pandas as pd


def analyze_dynamic_period_results():
    """동적 기간 백테스트 결과 분석"""

    # 최신 결과 파일 찾기
    results_dir = Path("results")
    csv_files = list(results_dir.glob("dynamic_period_backtest_results_*.csv"))

    if not csv_files:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        return

    # 최신 파일 선택
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 분석할 파일: {latest_file}")

    # 데이터 로드
    df = pd.read_csv(latest_file)
    print(f"📈 총 {len(df)}개 결과 로드됨")
    print()

    # 전략별 기간별 피벗 테이블 생성
    metrics = ["sharpe", "cagr", "mdd", "total_return", "hit_ratio"]

    print("🎯 단기/장기/통합 전략 성과 비교")
    print("=" * 80)

    for metric in metrics:
        if metric in df.columns:
            print(f"\n📊 {metric.upper()} 비교표:")
            pivot = df.pivot_table(
                index="strategy_name",
                columns="holding_days",
                values=metric,
                aggfunc="first",
            ).round(4)
            print(pivot)

    # 전략별 최고 성과 분석
    print("\n🏆 전략별 최고 성과:")
    print("-" * 50)

    for strategy in df["strategy_name"].unique():
        strategy_data = df[df["strategy_name"] == strategy]

        best_sharpe = strategy_data.loc[strategy_data["sharpe"].idxmax()]
        best_cagr = strategy_data.loc[strategy_data["cagr"].idxmax()]
        best_stability = strategy_data.loc[
            strategy_data["mdd"].idxmin()
        ]  # MDD가 가장 낮은 것

        print(f"\n{strategy} 전략:")
        print(".4f")
        print(".4f")
        print(".4f")
    # 기간별 평균 성과
    print("\n📅 기간별 평균 성과:")
    print("-" * 50)

    period_avg = df.groupby("holding_days")[["sharpe", "cagr", "mdd"]].mean().round(4)
    print(period_avg)

    # 전략별 평균 성과
    print("\n🎯 전략별 평균 성과:")
    print("-" * 50)

    strategy_avg = (
        df.groupby("strategy_name")[["sharpe", "cagr", "mdd"]].mean().round(4)
    )
    print(strategy_avg)

    print("\n✅ 분석 완료!")


if __name__ == "__main__":
    analyze_dynamic_period_results()
