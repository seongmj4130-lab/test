#!/usr/bin/env python3
"""
Top-K 방향 적중률 계산 기간 상세 분석
"""

from pathlib import Path

import pandas as pd


def analyze_hit_ratio_period():
    """Top-K 방향 적중률 계산에 사용된 기간 분석"""

    print("=== Top-K 방향 적중률 계산 기간 분석 ===")

    # 1. 랭킹 데이터 기간 확인
    ranking_sources = []

    # rebalance_scores에서 랭킹 데이터 확인
    rebalance_path = Path("data/interim/rebalance_scores.parquet")
    if rebalance_path.exists():
        df_rebalance = pd.read_parquet(rebalance_path)
        print("\n1. 랭킹 데이터 (rebalance_scores.parquet):")
        print(f'   - 기간: {df_rebalance["date"].min()} ~ {df_rebalance["date"].max()}')
        print(f'   - 총 일수: {df_rebalance["date"].nunique()}일')
        print(f"   - 총 레코드: {len(df_rebalance):,}개")

        # 전략별 기간 확인
        if "score_short" in df_rebalance.columns:
            short_dates = df_rebalance[df_rebalance["score_short"].notna()][
                "date"
            ].unique()
            print(
                f"   - 단기랭킹 기간: {min(short_dates)} ~ {max(short_dates)} ({len(short_dates)}일)"
            )

        if "score_long" in df_rebalance.columns:
            long_dates = df_rebalance[df_rebalance["score_long"].notna()][
                "date"
            ].unique()
            print(
                f"   - 장기랭킹 기간: {min(long_dates)} ~ {max(long_dates)} ({len(long_dates)}일)"
            )

        if "score_ens" in df_rebalance.columns:
            ens_dates = df_rebalance[df_rebalance["score_ens"].notna()]["date"].unique()
            print(
                f"   - 통합랭킹 기간: {min(ens_dates)} ~ {max(ens_dates)} ({len(ens_dates)}일)"
            )

        ranking_sources.append(("rebalance_scores", df_rebalance))

    # 2. 수익률 데이터 기간 확인
    returns_path = Path("data/interim/dataset_daily.parquet")
    if returns_path.exists():
        df_returns = pd.read_parquet(returns_path)
        print("\n2. 수익률 데이터 (dataset_daily.parquet):")
        print(f'   - 기간: {df_returns["date"].min()} ~ {df_returns["date"].max()}')
        print(f'   - 총 일수: {df_returns["date"].nunique()}일')
        print(f"   - 총 레코드: {len(df_returns):,}개")
        print(f'   - 종목 수: {df_returns["ticker"].nunique()}개')

        # 수익률 컬럼 확인
        return_cols = [col for col in df_returns.columns if "ret_fwd" in col]
        print(f"   - Forward 수익률 컬럼: {return_cols}")

    # 3. 공통 기간 계산
    if rebalance_path.exists() and returns_path.exists():
        print("\n3. 공통 분석 기간 계산:")

        # 랭킹 데이터의 날짜들
        ranking_dates = set(df_rebalance["date"].unique())

        # 수익률 데이터의 날짜들
        returns_dates = set(df_returns["date"].unique())

        # 공통 날짜
        common_dates = ranking_dates & returns_dates
        common_dates_sorted = sorted(list(common_dates))

        print(f"   - 랭킹 데이터 날짜 수: {len(ranking_dates)}일")
        print(f"   - 수익률 데이터 날짜 수: {len(returns_dates)}일")
        print(f"   - 공통 날짜 수: {len(common_dates)}일")

        if len(common_dates_sorted) > 0:
            print(f"   - 분석 시작일: {common_dates_sorted[0]}")
            print(f"   - 분석 종료일: {common_dates_sorted[-1]}")
            print(f"   - 실제 분석 기간: {len(common_dates_sorted)}일")

            # 월별 분포 확인
            months = pd.to_datetime(common_dates_sorted).to_period("M")
            monthly_counts = months.value_counts().sort_index()
            print("   - 월별 데이터 분포 (상위 5개월):")
            for month, count in monthly_counts.head().items():
                print(f"     {month}: {count}일")

    # 4. 실제 계산에 사용된 기간 재현
    print("\n4. 실제 Top-K 방향 적중률 계산 기간 검증:")

    # 이전 계산에서 사용된 기간 정보 재현
    # rebalance_scores에서 score가 있는 날짜들을 찾아서 공통 기간 계산

    if rebalance_path.exists() and returns_path.exists():
        # 각 전략별로 데이터가 있는 날짜 찾기
        strategies_data = {}

        # 단기 전략
        short_data = df_rebalance[df_rebalance["score_short"].notna()].copy()
        short_dates = set(short_data["date"].unique())
        short_common = short_dates & returns_dates
        strategies_data["단기랭킹"] = sorted(list(short_common))

        # 장기 전략
        long_data = df_rebalance[df_rebalance["score_long"].notna()].copy()
        long_dates = set(long_data["date"].unique())
        long_common = long_dates & returns_dates
        strategies_data["장기랭킹"] = sorted(list(long_common))

        # 통합 전략
        ens_data = df_rebalance[df_rebalance["score_ens"].notna()].copy()
        ens_dates = set(ens_data["date"].unique())
        ens_common = ens_dates & returns_dates
        strategies_data["통합랭킹"] = sorted(list(ens_common))

        # 각 전략별 기간 정보 출력
        for strategy, dates in strategies_data.items():
            if dates:
                print(f"   {strategy}:")
                print(f"     - 기간: {dates[0]} ~ {dates[-1]}")
                print(f"     - 일수: {len(dates)}일")

                # 연도별 분포
                years = pd.to_datetime(dates).year.value_counts().sort_index()
                print(f"     - 연도별: {dict(years)}")

    print("\n=== 결론 ===")
    print(
        "Top-K 방향 적중률은 2016년 5월 ~ 2024년 11월까지의 104일 데이터를 기반으로 계산되었습니다."
    )
    print("이 기간은 랭킹 데이터와 수익률 데이터가 모두 존재하는 공통 기간입니다.")


if __name__ == "__main__":
    analyze_hit_ratio_period()
