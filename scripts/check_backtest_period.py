#!/usr/bin/env python3
"""
백테스트 산출 기간 확인
"""

from pathlib import Path

import pandas as pd


def check_backtest_period():
    """백테스트에서 사용된 데이터 기간 확인"""

    print("📅 백테스트 산출 기간 확인")
    print("=" * 50)

    # 1. 설정된 기간
    config_path = Path("configs/redesigned_backtest_params.yaml")
    if config_path.exists():
        import yaml

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print("⚙️  설정된 기간:")
        print(f"   시작일: {config['params']['start_date']}")
        print(f"   종료일: {config['params']['end_date']}")
        print()

    # 2. L6 데이터 실제 기간
    baseline_dir = Path("baseline_20260112_145649")
    l6_path = baseline_dir / "data" / "interim" / "rebalance_scores_corrected.parquet"

    if l6_path.exists():
        df = pd.read_parquet(l6_path)
        # 날짜를 datetime으로 변환
        df["date"] = pd.to_datetime(df["date"])

        print("📊 L6 데이터 실제 기간:")
        min_date = df["date"].min()
        max_date = df["date"].max()
        total_days = (max_date - min_date).days
        print(f"   시작일: {min_date.strftime('%Y-%m-%d')}")
        print(f"   종료일: {max_date.strftime('%Y-%m-%d')}")
        print(f"   총 기간: {total_days}일 ({total_days/365:.1f}년)")
        print()

        # phase별 기간
        print("📋 Phase별 기간:")
        for phase in sorted(df["phase"].unique()):
            phase_data = df[df["phase"] == phase]
            start_date = phase_data["date"].min()
            end_date = phase_data["date"].max()
            days = (end_date - start_date).days
            print(
                f"   {phase.upper()}: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days}일)"
            )
        print()

    # 3. 백테스트 결과에서 확인 가능한 기간
    results_path = Path("results/dynamic_period_backtest_clean_20260113_214547.csv")
    if results_path.exists():
        df_results = pd.read_csv(results_path)
        print("🎯 백테스트 결과 요약:")
        print(f"   총 케이스: {len(df_results)}개")
        print(f"   전략 수: {df_results['strategy'].nunique()}개")
        print(f"   기간 수: {df_results['holding_days'].nunique()}개")
        print("   기간 범위: 20~120일")
        print()

    # 4. 실제 백테스트 수행 기간 추정
    print("📈 백테스트 수행 기간:")
    print("   • 데이터 준비: 2016-01-01 ~ 2024-12-31 (설정)")
    print("   • 실제 사용: 2016년 5월 ~ 2024년 11월 (L6 데이터 기준)")
    print("   • CV 분할: Dev/ Holdout 구간으로 분할")
    print("   • 리밸런싱: 월별 리밸런싱 (약 80-100회)")
    print("   • 평가 기간: 각 리밸런싱 후 holding_days 기간 수익률")


if __name__ == "__main__":
    check_backtest_period()
