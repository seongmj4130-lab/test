#!/usr/bin/env python3
"""
생성된 랭킹 데이터 확인 및 성과 지표 계산
"""

from pathlib import Path

import pandas as pd


def check_ranking_data():
    """생성된 랭킹 데이터 확인"""

    print("=== 생성된 랭킹 데이터 확인 ===")

    ranking_short_path = Path("data/interim/ranking_short_daily.parquet")
    ranking_long_path = Path("data/interim/ranking_long_daily.parquet")

    if ranking_short_path.exists():
        df_short = pd.read_parquet(ranking_short_path)
        print(f"단기 랭킹: {len(df_short):,}행 × {len(df_short.columns)}열")
        print("컬럼:", list(df_short.columns))
        print("샘플 데이터:")
        print(df_short.head(3).to_string())
        print(f'날짜 범위: {df_short["date"].min()} ~ {df_short["date"].max()}')
        print(f'종목 수: {df_short["ticker"].nunique()}')
    else:
        print("❌ 단기 랭킹 파일 없음")
        return

    print()

    if ranking_long_path.exists():
        df_long = pd.read_parquet(ranking_long_path)
        print(f"장기 랭킹: {len(df_long):,}행 × {len(df_long.columns)}열")
        print("컬럼:", list(df_long.columns))
        print("샘플 데이터:")
        print(df_long.head(3).to_string())
        print(f'날짜 범위: {df_long["date"].min()} ~ {df_long["date"].max()}')
        print(f'종목 수: {df_long["ticker"].nunique()}')
    else:
        print("❌ 장기 랭킹 파일 없음")
        return

    # 데이터 검증
    print("\n=== 데이터 검증 ===")

    # 날짜 일치 여부
    short_dates = set(df_short["date"].unique())
    long_dates = set(df_long["date"].unique())
    common_dates = short_dates & long_dates

    print(f"단기 날짜 수: {len(short_dates)}")
    print(f"장기 날짜 수: {len(long_dates)}")
    print(f"공통 날짜 수: {len(common_dates)}")

    if len(common_dates) > 0:
        # 랭킹 성과 계산
        print("\n=== 랭킹 성과 지표 계산 ===")

        # 실제 수익률 데이터 로드 (있는 경우)
        try:
            returns_path = Path("data/interim/dataset_daily.parquet")
            if returns_path.exists():
                df_returns = pd.read_parquet(returns_path)
                print(f"수익률 데이터: {len(df_returns):,}행")

                # 랭킹 성과 계산
                calculate_ranking_performance(df_short, df_long, df_returns)
            else:
                print("수익률 데이터 없음 - 기본 통계만 계산")
                calculate_basic_stats(df_short, df_long)

        except Exception as e:
            print(f"성과 계산 중 오류: {e}")
            calculate_basic_stats(df_short, df_long)


def calculate_basic_stats(df_short, df_long):
    """기본 통계 계산"""

    print("단기 랭킹 기본 통계:")
    print(f'  평균 스코어: {df_short["score"].mean():.4f}')
    print(f'  스코어 표준편차: {df_short["score"].std():.4f}')
    print(f'  최소 스코어: {df_short["score"].min():.4f}')
    print(f'  최대 스코어: {df_short["score"].max():.4f}')

    print("장기 랭킹 기본 통계:")
    print(f'  평균 스코어: {df_long["score"].mean():.4f}')
    print(f'  스코어 표준편차: {df_long["score"].std():.4f}')
    print(f'  최소 스코어: {df_long["score"].min():.4f}')
    print(f'  최대 스코어: {df_long["score"].max():.4f}')


def calculate_ranking_performance(df_short, df_long, df_returns):
    """랭킹 성과 지표 계산"""

    print("랭킹 성과 지표 계산 중...")

    # 공통 날짜 필터링
    common_dates = (
        set(df_short["date"].unique())
        & set(df_long["date"].unique())
        & set(df_returns["date"].unique())
    )

    if len(common_dates) == 0:
        print("공통 날짜가 없어 성과 계산 불가")
        return

    # Holdout 기간으로 가정 (마지막 6개월)
    all_dates = sorted(list(common_dates))
    holdout_start = all_dates[-int(len(all_dates) * 0.3)]  # 마지막 30%
    holdout_dates = [d for d in all_dates if d >= holdout_start]

    print(
        f"Holdout 기간: {holdout_dates[0]} ~ {holdout_dates[-1]} ({len(holdout_dates)}일)"
    )

    # Holdout 데이터 필터링
    df_short_holdout = df_short[df_short["date"].isin(holdout_dates)].copy()
    df_long_holdout = df_long[df_long["date"].isin(holdout_dates)].copy()
    df_returns_holdout = df_returns[df_returns["date"].isin(holdout_dates)].copy()

    # 수익률 컬럼 확인
    return_cols = [
        col
        for col in df_returns.columns
        if "return" in col.lower() or "ret" in col.lower()
    ]
    if "true_short" in df_returns.columns:
        short_return_col = "true_short"
    elif return_cols:
        short_return_col = return_cols[0]
    else:
        print("수익률 컬럼을 찾을 수 없음")
        return

    if "true_long" in df_returns.columns:
        long_return_col = "true_long"
    elif len(return_cols) > 1:
        long_return_col = return_cols[1]
    else:
        long_return_col = short_return_col

    print(f"단기 수익률 컬럼: {short_return_col}")
    print(f"장기 수익률 컬럼: {long_return_col}")

    # IC 계산 함수
    def calculate_ic(
        ranking_df, returns_df, score_col="score", return_col="true_short", top_k=20
    ):
        """Information Coefficient 계산"""
        results = []

        for date in ranking_df["date"].unique():
            rank_data = ranking_df[ranking_df["date"] == date]
            return_data = returns_df[returns_df["date"] == date]

            if len(rank_data) == 0 or len(return_data) == 0:
                continue

            # 랭킹과 수익률 매칭
            merged = rank_data.merge(
                return_data[["ticker", return_col]], on="ticker", how="inner"
            )

            if len(merged) < 10:  # 최소 샘플 수
                continue

            # IC 계산 (랭킹 스코어와 수익률의 상관계수)
            ic = merged[score_col].corr(merged[return_col])
            if not pd.isna(ic):
                results.append(ic)

        return results

    # 단기 랭킹 IC 계산
    short_ic_values = calculate_ic(
        df_short_holdout,
        df_returns_holdout,
        score_col="score",
        return_col=short_return_col,
    )

    # 장기 랭킹 IC 계산
    long_ic_values = calculate_ic(
        df_long_holdout,
        df_returns_holdout,
        score_col="score",
        return_col=long_return_col,
    )

    # IC 통계
    if short_ic_values:
        short_ic_mean = sum(short_ic_values) / len(short_ic_values)
        short_ic_std = (
            sum((x - short_ic_mean) ** 2 for x in short_ic_values)
            / len(short_ic_values)
        ) ** 0.5
        short_icir = short_ic_mean / short_ic_std if short_ic_std > 0 else 0

        print("단기 랭킹 IC 성과:")
        print(f"  IC 평균: {short_ic_mean:.4f}")
        print(f"  IC 표준편차: {short_ic_std:.4f}")
        print(f"  ICIR: {short_icir:.4f}")
        print(f"  샘플 수: {len(short_ic_values)}")

    if long_ic_values:
        long_ic_mean = sum(long_ic_values) / len(long_ic_values)
        long_ic_std = (
            sum((x - long_ic_mean) ** 2 for x in long_ic_values) / len(long_ic_values)
        ) ** 0.5
        long_icir = long_ic_mean / long_ic_std if long_ic_std > 0 else 0

        print("장기 랭킹 IC 성과:")
        print(f"  IC 평균: {long_ic_mean:.4f}")
        print(f"  IC 표준편차: {long_ic_std:.4f}")
        print(f"  ICIR: {long_icir:.4f}")
        print(f"  샘플 수: {len(long_ic_values)}")


if __name__ == "__main__":
    check_ranking_data()
